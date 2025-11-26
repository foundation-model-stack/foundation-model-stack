from functools import partial
from fms.models.hf.config_utils.config_registry import ModelConfigRegistry
from fms.models.hf.config_utils import param_builders as pb

# Any models that should be initializable from HF params should be added here.
_FMS_MODEL_CONFIG_REGISTRY = ModelConfigRegistry(
    {
        "LlamaForCausalLM": ("llama", pb.build_llama_params),
        "GPTBigCodeForCausalLM": ("gpt_bigcode", pb.build_gpt_bigcode_params),
        "MixtralForCausalLM": ("mixtral", pb.build_mixtral_params),
        "RobertaForMaskedLM": ("roberta", pb.build_roberta_params),
        "RobertaForQuestionAnswering": (
            "roberta_question_answering",
            pb.build_roberta_params,
        ),
        "GraniteForCausalLM": ("granite", pb.build_granite_params),
        "MistralForCausalLM": ("mistral", pb.build_mistral_params),
        "BambaForCausalLM": ("bamba", pb.build_bamba_params),
        "SiglipModel": ("siglip_vision", pb.build_siglip_vision_params),
        "LlavaNextForConditionalGeneration": ("llava_next", pb.build_llava_next_params),
        "MPNetForMaskedLM": ("mpnet", pb.build_mpnet_params),
        "BertForMaskedLM": ("bert", pb.build_bert_params),
        # Classify arches have some extra keys for labels
        "RobertaForSequenceClassification": (
            "roberta_classification",
            partial(pb.build_roberta_params, is_classify=True),
        ),
        "BertForSequenceClassification": (
            "bert_classification",
            partial(pb.build_bert_params, is_classify=True),
        ),
    }
)
