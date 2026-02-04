"""
Model specific utils for converting HF PretrainedConfig objects -> FMS kwargs.
"""

import logging
from fms.models import list_variants
from fms.models.hf.config_utils.config_utils_types import ParamBuilderFunc, RegistryMap
from transformers import PretrainedConfig
from typing import Optional

logger = logging.getLogger(__name__)


class ModelConfigRegistry:
    """Wrapper class that handles converting hf config -> FMS kwargs."""

    def __init__(self, registry_map: RegistryMap):
        self.model_param_builders: dict[str, ParamBuilderFunc] = {}
        self.model_arch_mappings: dict[str, str] = {}

        # Default registry initialization for all in tree models
        for hf_arch_name, fms_info in registry_map.items():
            self.register_model_arch_info(hf_arch_name, *fms_info)

    def register_model_arch_info(
        self, hf_arch_name: str, fms_arch_name: str, param_builder: ParamBuilderFunc
    ):
        """
        Register the transformers architecture dicts mapping to
        the builder for retrieving the FMS model's config param dict,
        as well as the fms arch name. Note that currently we always use
        the first listed architecture as the key, but in the future, we
        could use a tuple of HF architectures as keys.

        Args:
        hf_arch_name: arch name in HF Transformers.
        fms_arch_name: arch name in FMS.
        param_builder: Func for this model which takes exactly one arg
            (pretrained config) and returns the model's config params dict.
        """
        self.model_param_builders[hf_arch_name] = param_builder
        self.model_arch_mappings[hf_arch_name] = fms_arch_name

    def map_hf_to_fms_arch(self, architecture: str) -> Optional[str]:
        """
        Map the transformers architecture to the FMS architecture.

        Args:
        architecture: transformers architecture; note that if multiple
            are present, we typically pass architectures[0] currently.
        """
        if architecture in self.model_arch_mappings:
            fms_arch = self.model_arch_mappings[architecture]
            return fms_arch
        logger.warning(
            f"HF architecture {architecture} does not map to a registered FMS architecture!"
        )
        return None

    def map_hf_arch_to_fms_params(
        self, architecture: str, config: PretrainedConfig
    ) -> Optional[dict]:
        """
        Map the transformers architecture to the callable that produces
        the config params dict to be passed as additional kwargs when
        initializing the FMS model.

        Args:
        architecture: transformers architecture; note that if multiple
            are present, we typically pass architectures[0] currently.
        config: the HF transformers PretrainedConfig to be mapped.
        """
        # Map HF model config to FMS model config
        if architecture in self.model_param_builders:
            param_builder = self.model_param_builders[architecture]
            config_params = param_builder(config)
            return config_params
        logger.warning(
            f"HF architecture {architecture} does not map to a registered model param builder!"
        )
        return None

    def hf_config_to_fms_config_params(
        self, config: PretrainedConfig, model_path: Optional[str]
    ) -> dict:
        """
        Given a transformers config and the path to the model, build the params dict to be
        passed to the initializer of the FMS model.

        Args:
        config: The model's transformers config.
        model_path: The path to the model, or None if the weights are to be downloaded.
        """
        if config.architectures is None:
            raise ValueError("HF Config must contain a model architecture")

        if len(config.architectures) > 1:
            logger.warning(
                "The provided HF config supports multiple architectures; only the first, "
                "%s, will be used in building the FMS model's config params"
                % config.architectures[0]
            )

        architecture = config.architectures[0]
        config_params = self.map_hf_arch_to_fms_params(architecture, config)
        fms_arch = self.map_hf_to_fms_arch(architecture)

        if config_params is None or fms_arch is None:
            raise KeyError(
                f"The HF architecture {architecture} must map to both a config params builder and an fms architecture in the model config registry!"
            )

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
