# Huggingface Adapters in FMS

## What is it?

`HFModelArchitecture` is a set of base classes which provide huggingface functionality to pytorch-native modules. The benefits
you get from using this method are as follows:

- more flexibility of implementation of your underlying pytorch native classes
- seamless conversions from pytorch native modules to those using a huggingface API (no longer need to write long weight conversion scripts)
- performance improvements built in lower level modules get propogated up to huggingface scripts
- A simple interface to implement, which does not require underlying knowledge of the internals of huggingface utility interactions
- Less code to maintain - these classes act as a wrapper, where very little actual logic exists in their end-user implementations

A `HFModelArchitecture` is composed of:

- an embedding which is passed to its other modules if they require them
- an optional head `lm_head` which will be executed if it exists, if it does not, the `lm_head` will be ignored and
this will be considered a base model.

There are 3 main implementations of `HFModelArchiture` which can be extended by a user:

- `HFDecoderModelArchitecture` - provides the logic for a decoder model architecture (contains a decoder)
- `HFEncoderModelArchitecture` - provides the logic for an encoder model architecture (contains an encoder)
- `HFEncoderDecoderModelArchitecture` - provides the logic for an encoder-decoder model architecture (contains an encoder and a decoder)

In the majority of cases, users will be implementing one of the above 3 classes depending on their own model.

Each one of the respective `HFModelArchitecture` above require the input of an `HFEncoder` or `HFDecoder` (or both).
These classes serve as the mechanism to adapt your underlying models forward arguments to those of huggingface.

Lastly, the above does not take into account implementations of lm heads and computation of loss, for this, we have a
base utility class `LMHeadMixin`. When mixing in this class to any `HFModelArchitecture`, the implementation will now
take on that of the base class, plus some lm head. Current implementations of `LMHeadMixin` are as follows:

- `LMHeadModel` - Provides a decoder model with a `language modeling` head
- `ConditionalGeneration` - Provides an encoder-decoder model with a `language modeling` head
- `SequenceClassification` - Provides a model with sequence classification/regression head

## How to implement

Implementation of an HFModelArchitecture takes a few steps, and is quite minimal. For this section, we will implement
Llama from https://github.com/facebookresearch/llama.

### Adapt your underlying model parameters

Because Llama is a decoder-only model, the underlying architecture requires only a decoder(`HFDecoder`)

#### Creating an HFDecoder

HFDecoder requires implementation of a single method `_adapt`. This method will take in all standard parameters from
huggingface as well as any other parameters specified by the user which are non-standard, and return a huggingface
dataclass. Because HFModelArchitecture completely decouples the encoders/decoders/lm_heads, this class will only
focus on producing the output from the decoder. The following is an example using the Llama Transformer class:

```python
# https://github.com/facebookresearch/llama/blob/7e1b864d574fe6f5ff75fa1d028feb269f7152d2/llama/model.py#L457
def decoder_forward(model: Transformer, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = model.tok_embeddings(tokens)
        freqs_cis = model.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in model.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = model.norm(h)
        return h

class LlamaDecoder(HFDecoder):

    def __init__(self, model: nn.Module, config: PretrainedConfig):
        super().__init__(model, config)

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # this is a llama specific param, but we can get access to it here by simply adding it to the _adapt signature
        start_pos: int = 0,
        *args,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        # caching is part of the state of the TransformerBlock, therefore not required here
        output = decoder_forward(self.model, input_ids, start_pos)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=output)
```

### Creating your model architecture

#### Creating the base model

Once you have implemented the `HFDecoder` for your specific model, it is time to implement your own
HFModelArchitecture. For the current example, Llama, this will mean implementing the base class 
`HFDecoderModelArchitecture`.

```python
class Llama(HFDecoderModelArchitecture):

    # attributes required by HF
    config_class = LlamaConfig
    base_model_prefix = "llama"

    def __init__(
        self,
        config: LlamaConfig,
        decoder: Transformer = None,
        embedding: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        # in the case we have not yet received the decoder/embedding, initialize it here
        if decoder is None or embedding is None:
            model = Transformer(...)
            decoder = model if decoder is None else decoder
            embedding = model.tok_embeddings

        super().__init__(LlamaDecoder(decoder, config), embedding, config, *args, **kwargs)
```

This model can now be used as a base Huggingface Model without an LM head, but in many cases a user would like to add
some lm head to this. For this you will use the lm_head_mixins

#### Adding a head to your model

In many cases, a user would like to have an lm head in their model. Adding an lm head will provide a useful mechanism for
performing certain tasks such as text-generation or sequence-classification. All lm-heads can be easily added to these classes
by extending the base model with an `LMHeadMixin`. The following is an example of adding a `language modeling` head for Llama:

```python
class LlamaForCausalLM(LMHeadModelLMHeadMixin, Llama):

    def __init__(self, config: LlamaConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    @classmethod
    def _hf_model_from_fms(cls, model: Transformer, config: LlamaConfig) -> "LlamaForCausalLM":
        return cls(
            config=config,
            decoder=model.decoder,
            embedding=model.emb,
            lm_head=model.output,
        )
```

## Use your Huggingface model

Perform simple text-generation task:

```python
model: Transformer = Transformer(...)

hf_model: LlamaForCausalLM = LlamaForCausalLM.from_fms_model(model)

prompt = "I believe the meaning of life is"

with torch.no_grad():
    generator_hf = pipeline(
        task="text-generation", 
        model=hf_model, 
        tokenizer=tokenizer, 
        num_beams=3, 
        max_new_tokens=50
    )
    print(generator_hf(prompt))
```

## Future Tasks

- Automatic on-the-fly wrapping of a model with a single function (no implementation required)