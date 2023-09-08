import os
from typing import List

import numpy as np
import torch
import pytest

from fms.testing._internal.common_model import ModelFixtureMixin
from fms.testing._internal.common_path import AbstractResourcePath
from fms.testing.model_utils import get_signature

_FAILED_MODEL_SIGNATURE_OUTPUT_MSG = """
Failed consistency of signature. This could fail for one of 2 reasons:

1. either there was a change in the model architecture which caused a difference in model output
2. a bug was fixed which is causing a different model output and that is expected

If (2) then please re-run fms.tests.models.generate_small_model_tests with --generate_weights --generate_signature

Please provide a justification for re-running generate_small_model_tests in a PR
"""


class ModelConsistencyTestMixin(AbstractResourcePath, ModelFixtureMixin):
    """All tests related to model consistency will be part of this mixin"""

    @pytest.fixture(scope="class", autouse=True)
    def signature(self, resource_path) -> List[float]:
        """retrieve the signature from the test case directory

        Parameters
        ----------
        resource_path: str
            path to the specific test case directory specified in resource_path fixture

        Returns
        -------
        List[float]
            the signature stored in the test case directory that was created when generate_small_model_tests was called
            for this specific test model
        """
        return torch.load(os.path.join(resource_path, "signature.pth"))

    def test_model_output(self, model, signature):
        """test consistency of model output with signature"""

        actual = get_signature(model, params=self._forward_parameters)
        assert np.allclose(
            np.array(actual), np.array(signature)
        ), _FAILED_MODEL_SIGNATURE_OUTPUT_MSG
