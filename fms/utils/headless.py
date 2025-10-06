import warnings
import torch


def gather_outputs(
    base_model_output: torch.Tensor, last_n_tokens: int = 0, only_last_token=False, **_
) -> torch.Tensor:
    """
    General logic to gather the outputs of the base headless model for decoder models
    """

    # added for deprecation (returns a 2d tensor for backwards compatibilty)
    if last_n_tokens == 0 and only_last_token:
        # add tracing check here as the warnings will cause re-compilations
        is_compiling = torch.compiler.is_compiling()
        if not is_compiling:
            warnings.warn(
                "only_last_token will be deprecated in future versions, use last_n_tokens instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        base_model_output = base_model_output[:, -1, :]
    # this is the base case
    elif last_n_tokens > 0 and base_model_output.shape[1] >= last_n_tokens:
        base_model_output = base_model_output[:, -last_n_tokens:, :]

    return base_model_output
