from fms.models.hf.gpt_oss.modeling_gpt_oss_hf import (
    HFAdaptedGptOssForCausalLM,
)

# Note: The convert_to_hf function has been removed as it depends on
# transformers.GptOssConfig and transformers.GptOssForCausalLM which
# don't exist in the transformers library yet.
# If needed in the future, it can be re-added when those classes are available.
