# Foundation Model Stack

Foundation Model Stack is a collection of components for development, training,
and tuning of foundation models.

## Installation

```
pip install -e .
```
or
```
python setup.py install
```

There's an example inference script under `./examples`.

## HF Model Support

```python
# fms model
llama: LLaMA = LLaMA(config)

# huggingface model backed by fms internals
llama_hf = LLaMAHFForCausalLM.from_fms_model(llama)

# generate some text
generator = pipeline(task="text-generation", model=llama_hf, tokenizer=tokenizer)
generator("""q: how are you? a: I am good. How about you? q: What is the weather like today? a:""")
```
