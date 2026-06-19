# Enable Vision-Only Model Loading in FMS

## Motivation

When running a multimodal model via sendnn-inference, we want to enable separate vision encoder execution to allow parallel and non-blocking invocation of encoder model, separate from rest of the decoder model. To implement this capability, we need a way to only load vision part of multimodal model, for example, load `vision_tower` for Ministral3. This work is to add an optional `vision_only` flag when we load model via FMS's `get_model` function.


**Example use-case**:

```python
fms_model = get_model("hf_pretrained", model_path=..., vision_only=True)
```

This will load and allocate only the three vision-relevant components, skipping the LLM decoder stack entirely. These three components would be:
- `vision_tower`
- `multi_modal_projector`
- `language_model.base_model.embedding` (text embedding, needed to merge image tokens into text embedding space)

## Design

### Overview

The feature is implemented as a single optional `vision_only: bool = False` flag threaded through four layers: the public `get_model` entry point, the model constructors, the weight-loading loop, and the forward helpers. All existing callers are unaffected because the flag defaults to `False`.

### Model construction (`Mistral3` / `Ministral3`)

When `vision_only=True` the model constructor skips building the full LLM decoder stack (`language_model`). Instead it allocates only:

- `vision_tower` — the PixtralVisionModel image encoder
- `multi_modal_projector` — the linear projection from vision to text space
- `text_embedding` — a standalone `nn.Embedding` that replaces `language_model.base_model.embedding`

The standalone embedding is necessary because `prepare_inputs_for_generation` must merge image feature tokens into the text embedding space even when there is no decoder. A `_vision_only` instance flag is stored so that `_get_text_embeddings`, `reset_parameters`, and `post_init` can conditionally route through the standalone embedding rather than the full language model.

`Ministral3` overrides `__init__` directly, so it receives the same treatment independently. The shared helpers (`_get_text_embeddings`, `reset_parameters`, `post_init`) are defined on `Mistral3` and inherited, so the guard logic is written once.

### Weight loading (`load_state_dict_into_model`)

A new `key_prefix_filter` parameter accepts a tuple of checkpoint key prefixes. When provided, any key in the state dict that does not start with one of those prefixes is skipped **before** the HF→FMS adapter runs. Because the state dict is a `LazySafetensorsDict`, filtered keys are never read from disk — their shard files are never opened, which is the primary memory and I/O saving.

The filter operates on HF-side key names (before the adapter renames them) because the adapter has not been applied at the point of filtering.

### Entry point (`get_model`)

`get_model` extracts `vision_only` from its `**kwargs` before passing the remainder to `__maybe_infer_model_variant`, so the flag is not misinterpreted as a model configuration field. After variant inference, it re-injects `vision_only` into the `extra_args` dict that is forwarded to the model constructor. It also selects the appropriate `key_prefix_filter` value and passes it to `load_state_dict_into_model`.

### Data flow summary

```
get_model(..., vision_only=True)
    │
    ├─ pops vision_only from kwargs
    ├─ infers architecture/variant as normal
    ├─ passes vision_only → model constructor
    │       └─ skips language_model; creates text_embedding instead
    │
    └─ passes key_prefix_filter → load_state_dict_into_model
            └─ skips LLM decoder keys before adapter; tensor files never opened
```

---

## Key Prefix Mapping Reference

| Component | HF checkpoint prefix | FMS prefix after adapter |
|---|---|---|
| Vision tower | `vision_tower.` | `vision_tower.` |
| Projector | `multi_modal_projector.` | `multi_modal_projector.` |
| Text embedding | `language_model.model.embed_tokens.` | `language_model.base_model.embedding.` |
| LLM decoder (skipped) | `language_model.model.layers.` | `language_model.base_model.layers.` |
| LLM head (skipped) | `language_model.lm_head.` | `language_model.head.` |

---

## Implications for Existing Models

### `LlavaNext` (`fms/models/llava_next.py`)

`LlavaNext` has the same three-component structure as `Mistral3`:

- `language_model` (Granite LLM)
- `vision_tower` (SiglipVision)
- `multi_modal_projector` (LlavaNextMultiModalProjector)

Text embeddings are accessed in `prepare_inputs_for_generation` directly as
`self.language_model.base_model.embedding(input_ids)` — the same pattern as
`Mistral3`. Completing `vision_only` support here would require: a `vision_only` flag in
`__init__`, a conditional `text_embedding`, guarded `reset_parameters` /
`post_init` calls, and an extraction of the inline embedding accesses in
`prepare_inputs_for_generation` into a `_get_text_embeddings` helper.

`LlavaNext` already carries a `VISION_ONLY_HF_PREFIXES` class attribute
(added alongside `Mistral3`), so `get_model` will pick up the right prefix
filter automatically once the constructor-side changes are made. The HF prefix
names happen to be identical to `Mistral3` because both architectures use the
same top-level key naming convention (`vision_tower.`, `multi_modal_projector.`,
`language_model.model.embed_tokens.`).

The constructor-side changes are not included in this work; the model is noted
here as the natural next candidate.

---

## Guide for Future Multimodal Models

Any FMS multimodal model that wants to support `vision_only=True` needs to
implement the following four things consistently.

### 1. Model constructor

Add a `vision_only: bool = False` parameter. Store it as `self._vision_only`.
Conditionally construct the LLM:

```python
self._vision_only = vision_only
if not vision_only:
    self.language_model = <LLMClass>(...)
else:
    # Minimal embedding needed to merge image tokens into text space
    self.text_embedding = nn.Embedding(
        self.config.text_config.src_vocab_size,
        self.config.text_config.emb_dim,
    )
self.vision_tower = <VisionEncoder>(...)
self.multi_modal_projector = <Projector>(...)
```

Always construct `vision_tower` and `multi_modal_projector` unconditionally.

### 2. Text embedding helper

If the model has a `_get_text_embeddings` (or equivalent inline logic in
`prepare_inputs_for_generation`), route through the standalone embedding when
in vision-only mode:

```python
def _get_text_embeddings(self, input_ids, input_embeds):
    if input_embeds is not None:
        return input_embeds
    if self._vision_only:
        return self.text_embedding(input_ids)
    return self.language_model.base_model.embedding(input_ids)
```

If the embedding access is inlined (as in `LlavaNext`), extract it into a
helper first so the guard can be written in one place.

### 3. `reset_parameters` and `post_init`

Guard any call that touches `language_model`:

```python
def reset_parameters(self):
    if not self._vision_only:
        self.language_model.reset_parameters()
    self.vision_tower.reset_parameters()

def post_init(self):
    if not self._vision_only:
        self.language_model.post_init()
    self.vision_tower.post_init()
```

### 4. `VISION_ONLY_HF_PREFIXES` class attribute

Define a class-level tuple of HF-side checkpoint prefixes that covers only the
vision components and the text embedding for the architecture. This lives on
the model class itself, not in `__init__.py`, so each model owns its own
prefix list.

```python
class MyMultimodalModel(nn.Module):
    VISION_ONLY_HF_PREFIXES: tuple = (
        "<vision_tower_hf_prefix>.",
        "<projector_hf_prefix>.",
        "<embed_tokens_hf_prefix>.",   # HF name for the text token embedding
        "<embed_tokens_fms_prefix>.",  # FMS name, in case checkpoint is already FMS-formatted
    )
```

`get_model` discovers the constant via `getattr(fms_model, "VISION_ONLY_HF_PREFIXES", None)`
and passes it as `key_prefix_filter` to `load_state_dict_into_model`. If the
attribute is absent (model does not support `vision_only`), `get_model` will
pass `None` and all keys are loaded as normal — no error, no silent data loss.

### Checklist

| Step | What to do |
|---|---|
| Model `__init__` | Add `vision_only` param; conditionally skip LLM; create `text_embedding` |
| `_get_text_embeddings` | Route through `self.text_embedding` when `_vision_only` |
| `reset_parameters` / `post_init` | Guard calls to `language_model` |
| `__init__.py` constant | Define `_<ARCH>_VISION_ONLY_HF_PREFIXES` with correct HF key prefixes |
| `get_model` | Select and pass the right prefix constant for the architecture |
