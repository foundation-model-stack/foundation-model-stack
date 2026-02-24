import torch


def pack_image_embeddings(
    input_ids: torch.Tensor,  # (B, T)
    inputs_embeds: torch.Tensor,  # (B, T, D)
    image_features: torch.Tensor,  # (B, L, D) or (B, N, L, D)
    image_token_id: int,  # e.g., 49190
    expected_L: int,  # e.g., 64  (512/16 => 32; 32*32/(4*4)=64)
) -> torch.Tensor:
    """
    masked replace (no length change) + HARD span asserts:
    Precedence of errors:
      - If there is exactly 1 span -> prefer per-span length error
      - If there are multiple spans -> prefer total count error (message contains 'expected')
    Supports image_features of shape (B,L,D) or (B,N,L,D).
    """
    B, T, D = inputs_embeds.shape

    # Normalize image_features to (B, N, L, D)
    if image_features.ndim == 3:
        Bf, L, D_img = image_features.shape
        if Bf != B:
            raise ValueError(f"B mismatch: embeds={B} vs img={Bf}")
        image_features = image_features.unsqueeze(1)  # (B,1,L,D)
    elif image_features.ndim == 4:
        Bf, Nf, L, D_img = image_features.shape
        if Bf != B:
            raise ValueError(f"B mismatch: embeds={B} vs img={Bf}")
    else:
        raise ValueError(
            f"image_features must be (B,L,D) or (B,N,L,D), got {tuple(image_features.shape)}"
        )

    Bf, N, L, D_img = image_features.shape
    if D != D_img:
        raise ValueError(f"D mismatch: embeds={D}, img={D_img}")
    if L != expected_L:
        raise ValueError(f"image_seq_len mismatch: got {L}, expected {expected_L}")

    if len(input_ids) != len(image_features):
        raise ValueError("input_ids and image_features must have the same batch size")
    mask = input_ids == int(image_token_id)  # (B, T)
    out = inputs_embeds.clone()

    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue  # no images for this sample

        # contiguous spans
        runs: list[tuple[int, int]] = []
        s = prev = int(idx[0].item())
        for i in idx[1:].tolist():
            i = int(i)
            if i == prev + 1:
                prev = i
            else:
                runs.append((s, prev))
                s = prev = i
        runs.append((s, prev))

        total = idx.numel()
        expected_total = len(runs) * L
        if len(runs) > N:
            raise ValueError(
                f"[PackError] batch={b}: found {len(runs)} image spans but only {N} image(s) provided"
            )

        # ----- precedence: single span -> per-span first; multiple spans -> total first
        if len(runs) == 1:
            # single span: prefer span length error first
            a, z = runs[0]
            span_len = z - a + 1
            if span_len != L:
                raise ValueError(
                    f"[PackError] batch={b}: span 0 len {span_len}, expected {L} (a={a}, z={z})"
                )
            # then enforce total (should match if span len matched)
            if total != expected_total:
                raise ValueError(
                    f"[PackError] batch={b}: total {total} != expected {expected_total}"
                )
        else:
            # multiple spans: prefer total mismatch error first
            if total != expected_total:
                raise ValueError(
                    f"[PackError] batch={b}: total {total} != expected {expected_total}"
                )
            # then validate each span length
            for k, (a, z) in enumerate(runs):
                span_len = z - a + 1
                if span_len != L:
                    raise ValueError(
                        f"[PackError] batch={b}: span {k} len {span_len}, expected {L} (a={a}, z={z})"
                    )

        # masked replace
        for img_i, (a, z) in enumerate(runs):
            out[b, a : z + 1, :] = image_features[b, img_i, :, :]

    return out
