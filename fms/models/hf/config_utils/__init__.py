"""
This module defines the mapping between HuggingFace and FMS architectures.
It also defines the way in which the kwarg overrides to the FMS models are
created!

Guidelines for registering a new model:
- Add a new key in the ModelConfigRegistry instance below with your
  transformers architecture. It should map to a tuple of two elements,
  where the first is the corresponding architecture string in FMS, and
  the second is a function that takes only the pretrained config as input,
  and returns the dictionary of config params.
- Implement the builder function in the `param_builders`
- If the base config func for the model is already there, and you need
  to extend it a bit to add a capability (e.g., classify's num labels),
  you can add a kwarg with a default value for the common case, and
  wrap it in a `partial` to override the default value to align the
  signature cleanly.

NOTE on current behavior for composite models: Nested subconfig
mappings architectures are hardcoded!! E.g., granite vision
directly creates the ModelConfig instances (*not* kwargs) for the
nested granite LLM & siglip subconfigs. We should abstract this in the
future, and consider extending support to map to the ModelConfig objects
directly here to make the boundary with subconfigs less awkward.
"""

from functools import partial
from fms.models.hf.config_utils.config_registry import ModelConfigRegistry
from fms.models.hf.config_utils import param_builders as pb
from fms.models.hf.config_utils.config_utils_types import RegistryMap

# Any models that should be initializable from HF params should be added here.
# fmt: off
__FMS_MODEL_REGISTRY_MAP: RegistryMap = {
    "LlamaForCausalLM": ("llama", pb.build_llama_params),
    "GPTBigCodeForCausalLM": ("gpt_bigcode", pb.build_gpt_bigcode_params),
    "MixtralForCausalLM": ("mixtral", pb.build_mixtral_params),
    "RobertaForMaskedLM": ("roberta", pb.build_roberta_params),
    "RobertaForQuestionAnswering": ("roberta_question_answering", pb.build_roberta_params),
    "GraniteForCausalLM": ("granite", pb.build_granite_params),
    "GraniteMoeHybridForCausalLM": ("granite_moe_hybrid", pb.build_granite_moe_hybrid_params),
    "MistralForCausalLM": ("mistral", pb.build_mistral_params),
    "BambaForCausalLM": ("bamba", pb.build_bamba_params),
    "SiglipModel": ("siglip_vision", pb.build_siglip_vision_params),
    "LlavaNextForConditionalGeneration": ("llava_next", pb.build_llava_next_params),
    "MPNetForMaskedLM": ("mpnet", pb.build_mpnet_params),
    "BertForMaskedLM": ("bert", pb.build_bert_params),
    # Classify arches have some extra keys for labels
    "RobertaForSequenceClassification": ("roberta_classification", partial(pb.build_roberta_params, is_classify=True)),
    "BertForSequenceClassification": ("bert_classification", partial(pb.build_bert_params, is_classify=True)),
}
# fmt: on

_FMS_MODEL_CONFIG_REGISTRY = ModelConfigRegistry(__FMS_MODEL_REGISTRY_MAP)
