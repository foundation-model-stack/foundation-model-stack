"""
Model specific utils for converting HF PretrainedConfig objects -> FMS kwargs.
"""

from fms.models import list_variants

class ModelConfigRegistry:
    """Wrapper class that handles converting hf config -> FMS kwargs."""
    def __init__(self, registry_map=None):
        self.model_param_builders = {}
        self.model_arch_mappings = {}

        if registry_map is not None:
            # Default registry initialization for all in tree models
            for hf_arch_name, fms_info in registry_map.items():
                self.register_model_arch_info(
                    hf_arch_name, *fms_info
                )

    def register_model_arch_info(self, hf_arch_name, fms_arch_name, param_builder):
        self.model_param_builders[hf_arch_name] = param_builder
        self.model_arch_mappings[hf_arch_name] = fms_arch_name

    def map_hf_to_fms_arch(self, architecture):
        if architecture in self.model_arch_mappings:
            fms_arch = self.model_arch_mappings[architecture]
            return fms_arch
        raise KeyError(f"HF architecture {architecture} is unsupported! Registered architectures: {list(self.model_arch_mappings.keys())}")

    def map_hf_arch_to_fms_params(self, architecture, config):
        # Map HF model config to FMS model config
        if architecture in self.model_arch_mappings:
            param_builder = self.model_param_builders[architecture]
            config_params = param_builder(config)
            return config_params
        raise KeyError(f"HF architecture {architecture} is unsupported! Registered architectures: {list(self.model_arch_mappings.keys())}")

    def hf_config_to_fms_config_params(self, config, model_path):
        architecture = config.architectures[0]

        config_params = self.map_hf_arch_to_fms_params(architecture, config)
        fms_arch = self.map_hf_to_fms_arch(architecture)

        # infer get_model params
        config_params["architecture"] = fms_arch
        config_params["variant"] = list_variants(fms_arch)[0]
        config_params["model_path"] = model_path

        ## infer quantization parameters
        quant_config = getattr(config, "quantization_config", None)
        if quant_config is not None:
            try:
                from fms_mo.aiu_addons import _infer_quantization_config  # type: ignore[import-untyped,import-not-found]
            except ImportError:
                raise RuntimeError(
                    "You need to install fms-model-optimizer to load quantized models"
                )
            linear_config = _infer_quantization_config(quant_config)
            if linear_config:
                config_params["linear_config"] = linear_config
        return config_params
