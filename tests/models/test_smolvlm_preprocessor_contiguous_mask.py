import pytest
import torch

import fms.models.idefics3 as idefics3_mod


def test_smolvlm_preprocess_expanded_pixel_attention_mask_is_contiguous(monkeypatch):
    b, n, c, h, w = 2, 3, 3, 32, 32

    class DummyProcessor:
        def __call__(self, images=None, **kwargs):
            return {
                "pixel_values": torch.zeros(b, n, c, h, w),
                "pixel_attention_mask": torch.ones(b, 1, h, w, dtype=torch.long),
            }

    monkeypatch.setattr(
        idefics3_mod,
        "load_smolvlm_processor",
        lambda *args, **kwargs: DummyProcessor(),
    )

    with pytest.warns(FutureWarning):
        processor = idefics3_mod.load_smolvlm_preprocessor()

    out = processor.preprocess(images=[object()])
    pam = out["pixel_attention_mask"]

    assert pam.shape == (b, n, h, w)
    assert pam.is_contiguous()

    # This would fail if the mask were a non-contiguous expand() view.
    _ = pam.view(b * n, h, w)
