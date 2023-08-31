import pytest

from fms.models.hf.llama.configuration_llama_hf import LLaMAHFConfig
from fms.models.hf.llama.modeling_llama_hf import LLaMAHFForCausalLM
from fms.models.llama import LLaMA, LLaMAConfig
from ..base import _case_paths, _test_ids
from ..test_hf_model import AbstractHFModelTest
import torch

from ..utils import ModelSignatureParams, HFModelSignatureParams, compare_model_signatures


class TestLlama(AbstractHFModelTest):
    """
    Model Test Suite for llama
    """

    _model_class = LLaMA
    _config_class = LLaMAConfig
    _hf_model_class = LLaMAHFForCausalLM
    _hf_config_class = LLaMAHFConfig
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    _hf_forward_parameters = ["input_ids", "labels"]

    @pytest.fixture(params=_case_paths("llama"), ids=_test_ids)
    def cases(self, request):
        return request.param

    @pytest.mark.slow
    def test_llama_7b_equivalence(self):
        """Tests llama equivalence with a known implementation. Takes approximately 8:38 on an mbp with M1 chip"""
        from transformers import AutoModelForCausalLM, pipeline

        from transformers import AutoTokenizer
        # for now, this test won't be run, but it has been verified
        # if you would like to try this, set llama_model_path to the huggingface llama2 model path
        llama_model_path = ""
        tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=True)
        hf_model = AutoModelForCausalLM.from_pretrained(llama_model_path)

        config = LLaMAConfig(
            src_vocab_size=tokenizer.vocab_size,
            emb_dim=4096,
            multiple_of=256,
            nheads=32,
            nlayers=32,
            norm_eps=1e-05,
            pad_id=hf_model.config.pad_token_id,
        )
        model = LLaMA(config)
        count_parameters = lambda m: sum(p.numel() for p in m.parameters())
        assert count_parameters(model) == count_parameters(hf_model)

        hf_sd = hf_model.model.state_dict()
        hf_sd = rename_weights_to_fms(hf_sd)
        model.load_state_dict(hf_sd, strict=False)
        model.shared.head.weight = hf_model.lm_head.weight
        model.stack.rot_emb.freqs = hf_model.model.layers[0].self_attn.rotary_emb.inv_freq
        for layer in model.stack.layers:
            q = layer.attn.query.weight.data
            q = q.view(model.config.nheads, 2, -1, q.size(1)).transpose(1, 2).reshape(*q.size())
            layer.attn.query.weight.data = q

            k = layer.attn.key.weight.data
            k = k.view(model.config.nheads, 2, -1, k.size(1)).transpose(1, 2).reshape(*k.size())
            layer.attn.key.weight.data = k

        hf_model_fms = LLaMAHFForCausalLM.from_fms_model(
            model,
            bos_token_id=hf_model.config.bos_token_id,
            eos_token_id=hf_model.config.eos_token_id,
            pad_token_id=hf_model.config.pad_token_id
        )

        assert count_parameters(hf_model_fms) == count_parameters(hf_model)

        model.eval()
        hf_model.eval()
        hf_model_fms.eval()

        hf_model_fms.eval()
        inp = torch.arange(0, 16).unsqueeze(0)

        fms_signature_params = ModelSignatureParams(model=model, params=1, inp=inp)

        hf_fms_signature_params = HFModelSignatureParams(
            model=hf_model_fms,
            params=["input_ids", "labels"],
            other_params={"return_dict": True},
            inp=inp
        )

        hf_signature_params = HFModelSignatureParams(
            model=hf_model,
            params=["input_ids", "labels"],
            other_params={"return_dict": True},
            inp=inp,
        )

        compare_model_signatures(fms_signature_params, hf_fms_signature_params)
        compare_model_signatures(hf_fms_signature_params, hf_signature_params)

        prompt = """q: how are you? a: I am good. How about you? q: What is the weather like today? a:"""

        generator_hf = pipeline(
            task="text-generation", model=hf_model, tokenizer=tokenizer, use_cache=True, num_beams=3, max_new_tokens=20
        )
        generator_hf_fms = pipeline(
            task="text-generation",
            model=hf_model_fms,
            tokenizer=tokenizer,
            use_cache=True,
            num_beams=3,
            max_new_tokens=20,
        )
        output_hf = generator_hf(prompt)
        output_hf_fms = generator_hf_fms(prompt)
        assert output_hf == output_hf_fms
        print(output_hf)
        print(output_hf_fms)


def rename_weights_to_fms(orig_sd):
    import re

    replacements = [
        (r"^embed_tokens.weight", "shared.emb.weight"),
        (r"^norm", "stack.dec_norm"),
        (r"^layers", "stack.layers"),
        (r"self_attn\.k_proj", "attn.key"),
        (r"self_attn\.v_proj", "attn.value"),
        (r"self_attn\.q_proj", "attn.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in orig_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd
