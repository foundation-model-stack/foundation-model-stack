from functools import partial
from fms.models.hf.config_utils.config_registry import ModelConfigRegistry
from fms.models.hf.config_utils.param_builders import *

# Any models that should be initializable from HF params should be added here.
_FMS_MODEL_CONFIG_REGISTRY = ModelConfigRegistry(
    {
        "LlamaForCausalLM": ("llama", build_llama_params),
        "GPTBigCodeForCausalLM": ("gpt_bigcode", build_gpt_bigcode_params),
        "MixtralForCausalLM": ("mixtral", build_mixtral_params),
        "RobertaForMaskedLM": ("roberta", build_roberta_params),
        "RobertaForQuestionAnswering": (
            "roberta_question_answering",
            build_roberta_params,
        ),
        "GraniteForCausalLM": ("granite", build_granite_params),
        "MistralForCausalLM": ("mistral", build_mistral_params),
        "BambaForCausalLM": ("bamba", build_bamba_params),
        "SiglipModel": ("siglip_vision", build_siglip_vision_params),
        "LlavaNextForConditionalGeneration": ("llava_next", build_llava_next_params),
        "MPNetForMaskedLM": ("mpnet", build_mpnet_params),
        "BertForMaskedLM": ("bert", build_bert_params),
        # Classify arches have some extra keys for labels
        "RobertaForSequenceClassification": (
            "roberta_classification",
            partial(build_roberta_params, is_classify=True),
        ),
        "BertForSequenceClassification": (
            "bert_classification",
            partial(build_bert_params, is_classify=True),
        ),
    }
)
