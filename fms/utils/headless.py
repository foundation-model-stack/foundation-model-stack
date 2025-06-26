import warnings

import torch


def gather_outputs(f):
    def _headless_execution(self, *args, **kwargs):
        output, cache = f(self, *args, **kwargs)

        index = kwargs.get("index", None)

        # added for deprecation
        if "only_last_token" in kwargs:
            # add tracing check here as the warnings will cause re-compilations
            is_compiling = torch.compiler.is_compiling()
            if not is_compiling:
                warnings.warn(
                    "only_last_token will be deprecated in future versions, use index instead",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if index is None:
                index = -1 if kwargs["only_last_token"] else None
            elif not is_compiling:
                warnings.warn("ignoring only_last_token as index is set")

        if index is not None:
            if isinstance(index, int):
                output = output[:, index, :]
            else:
                output = output.gather(1, index)
        
        return output, cache
    return _headless_execution