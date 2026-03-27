import logging
import os
from typing import Any, List, MutableMapping, Optional, Tuple

import torch
import torch.nn.functional as F
import tqdm
from lm_eval.api.instance import Instance  # type: ignore
from lm_eval.api.model import LM  # type: ignore
from lm_eval.api.registry import register_model  # type: ignore
from torch import nn

from fms.utils import tokenizers
from fms.utils.generation import _make_cache_contiguous, _make_cache_dynamic
from fms.utils.generation import __update_padding_kwargs as _update_padding_kwargs

# silence HF warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


@register_model("fms")
class FMSEvalHarnessLM(LM):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: tokenizers.BaseTokenizer,
        use_cache: bool = False,
        device="cpu",
        rank=0,
        world_size=1,
    ):
        self.wrapped_model = model
        self.tokenizer = tokenizer
        self.use_cache = use_cache
        self._rank = rank
        self._world_size = world_size
        self.device = device

        # workaround for https://github.com/EleutherAI/lm-evaluation-harness/issues/1333
        # until the fix is in a release
        def generic_object():
            return None

        self.model = generic_object
        self.model.config = generic_object  # type: ignore
        self.model.config._name_or_path = "FMSEvalHarnessLM"  # type: ignore

    def loglikelihood_one(self, context: str, continuation: str) -> Tuple[float, bool]:
        context_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(context)
        )
        if not len(context_ids):
            context_ids = [self.tokenizer.bos_token_id]

        continuation_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(continuation)
        )
        input_ids = context_ids + continuation_ids[:-1]
        input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        logits = F.log_softmax(self.wrapped_model(input_ids)[0], -1)
        continuation_probs = logits[len(context_ids) - 1 :]
        loglikelihood = torch.gather(
            continuation_probs,
            1,
            torch.tensor(continuation_ids, device=self.device).unsqueeze(1),
        ).squeeze()
        predicted = torch.argmax(continuation_probs, -1).tolist()
        greedy = predicted == continuation_ids
        return loglikelihood.sum().cpu().item(), greedy

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        result = []
        for request in requests:
            context, continuation = request.args
            result.append(self.loglikelihood_one(context, continuation))
        return result

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("not implemented yet")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Greedy decoding, stopping on:
          - tokenizer.eos_token_id (if present)
          - first occurrence of any string in `until` (Instance.args[1]['until'])
          - `max_gen_toks` cap (Instance.args[1]['max_gen_toks'])

        Notes:
          * Ignores sampling params (temperature/top_p/top_k) for determinism.
          * Returns only the generated completion (not the prompt).
        """
        result: List[str] = []
        kwargs: MutableMapping[str, Any] = dict()

        # KV caching settings
        kwargs["use_cache"] = self.use_cache
        if kwargs["use_cache"]:
            logger.info("KV caching enabled")
            kwargs["contiguous_cache"] = True
        else:
            logger.info("KV caching disabled")

        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        for idx, request in tqdm.tqdm(enumerate(requests), total=len(requests)):
            # lm-eval passes: args = (prompt: str, gen_kwargs: dict)
            if isinstance(request.args, tuple):
                prompt, gen_kwargs = request.args
            else:
                # Fallback if a different shape is ever seen
                prompt, gen_kwargs = request.args, {}

            until: List[str] = gen_kwargs.get("until", []) or []

            max_gen_toks_value = gen_kwargs.get("max_gen_toks")
            if max_gen_toks_value is None:
                if idx == 0:  # only emit the warning once
                    logger.warning("max_gen_toks not provided, using default 256")
                max_gen_toks = 256
            else:
                max_gen_toks = int(max_gen_toks_value)

            if kwargs["use_cache"]:
                # reset KV cache
                kwargs["past_key_value_states"] = None

            # Tokenize the prompt
            context_tokens = self.tokenizer.tokenize(prompt)
            context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
            if not len(context_ids):
                context_ids = [self.tokenizer.bos_token_id]

            # Build input_ids tensor
            input_ids = torch.tensor(context_ids, dtype=torch.long, device=self.device)

            # model requires batch dimension
            if not len(input_ids.shape) > 1:
                input_ids = input_ids.unsqueeze(0)

            # (maybe) TODO: respect the max model length
            # max_gen_toks = min(max_gen_toks, max_model_len - input_ids.shape[-1])

            generated_ids: List[int] = []
            stop_text: Optional[str] = None

            for i in range(max_gen_toks):
                # prepare any padding keyword arguments
                # iteration 0 is the prefill step (cache has not been filled yet), so no need to extend the mask/position_ids
                if i > 0:
                    kwargs = _update_padding_kwargs(kwargs["use_cache"], kwargs)

                # Forward pass; models may return (logits,) or logits directly
                out = self.wrapped_model(input_ids, **kwargs)

                if kwargs["use_cache"]:
                    logits, past_key_value_states = out
                    # TODO: this should go away when reduce-overhead issues are fixed, or
                    # maybe could be moved into model code to be more portable.
                    kwargs["past_key_value_states"] = past_key_value_states
                    if kwargs["contiguous_cache"]:
                        kwargs["past_key_value_states"] = _make_cache_contiguous(
                            kwargs["past_key_value_states"]
                        )
                    if torch._dynamo.config.dynamic_shapes:
                        kwargs["past_key_value_states"] = _make_cache_dynamic(
                            kwargs["past_key_value_states"]
                        )
                else:
                    logits = out

                # Handle both 3D and 2D logits
                if logits.dim() == 3:
                    # (batch, seq, vocab) -> take last position for batch 0
                    next_token_logits = logits[:, -1, :]  # (1, vocab)
                    next_id = int(torch.argmax(next_token_logits, dim=-1).item())
                elif logits.dim() == 2:
                    # (seq, vocab) -> take last position
                    next_token_logits = logits[-1, :]  # (vocab,)
                    next_id = int(torch.argmax(next_token_logits).item())
                else:
                    raise RuntimeError(
                        f"Unexpected logits shape {tuple(logits.shape)}; "
                        "expected (batch, seq, vocab) or (seq, vocab)."
                    )

                # EOS check
                if eos_id is not None and next_id == eos_id:
                    break

                # Append and continue
                generated_ids.append(next_id)
                next_id_tensor = torch.tensor(
                    [[next_id]], dtype=torch.long, device=self.device
                )

                if kwargs["use_cache"]:
                    input_ids = next_id_tensor
                else:
                    input_ids = torch.cat([input_ids, next_id_tensor], dim=1)

                # Stop-string check (string-level)
                if until:
                    gen_tokens = self.tokenizer.convert_ids_to_tokens(
                        torch.tensor(generated_ids, dtype=torch.long).cpu()  # type: ignore[arg-type]
                    )
                    gen_text = self.tokenizer.convert_tokens_to_string(gen_tokens)

                    # Find earliest occurrence of any stop string
                    stop_pos = None
                    for s in until:
                        idx = gen_text.find(s)
                        if idx != -1:
                            stop_pos = idx if stop_pos is None else min(stop_pos, idx)

                    if stop_pos is not None:
                        # Trim at the earliest stop boundary and finish
                        stop_text = gen_text[:stop_pos]
                        break

            # Final decode if we didn't break on 'until'
            if stop_text is None:
                gen_tokens = self.tokenizer.convert_ids_to_tokens(
                    torch.tensor(generated_ids, dtype=torch.long).cpu()  # type: ignore[arg-type]
                )
                stop_text = self.tokenizer.convert_tokens_to_string(gen_tokens)

            result.append(stop_text)

        return result
