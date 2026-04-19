import os
from typing import Any, List, MutableMapping, Optional, Tuple

import functools
import logging
import time
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


# Module-level cache for tokenization to avoid self being part of cache key
@functools.lru_cache(maxsize=None)
def _tokenize_cached(
    tokenizer: tokenizers.BaseTokenizer, context: str, continuation: str
) -> Tuple[List[int], List[int], List[int]]:
    """Tokenize context and continuation strings - cached implementation."""
    context_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
    if not len(context_ids):
        context_ids = [tokenizer.bos_token_id]

    continuation_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(continuation))
    input_ids = context_ids + continuation_ids[:-1]

    return context_ids, continuation_ids, input_ids


logger = logging.getLogger(__name__)


# Module-level cache for tokenization to avoid self being part of cache key
@functools.lru_cache(maxsize=None)
def _tokenize_cached(
    tokenizer: tokenizers.BaseTokenizer, context: str, continuation: str
) -> Tuple[List[int], List[int], List[int]]:
    """Tokenize context and continuation strings - cached implementation."""
    context_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
    if not len(context_ids):
        context_ids = [tokenizer.bos_token_id]

    continuation_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(continuation))
    input_ids = context_ids + continuation_ids[:-1]

    return context_ids, continuation_ids, input_ids


@register_model("fms")
class FMSEvalHarnessLM(LM):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: tokenizers.BaseTokenizer,
        use_cache: bool = False,
        batch_size: int = 1,
        device="cpu",
        rank=0,
        world_size=1,
    ):
        self.wrapped_model = model
        self.tokenizer = tokenizer
        self.use_cache = use_cache
        self.batch_size = batch_size
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

    def _tokenize(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """Tokenize context and continuation strings.

        Cached to avoid redundant tokenization when sorting requests by length.
        """
        return _tokenize_cached(self.tokenizer, context, continuation)

    def loglikelihood(
        self,
        requests: List[Instance],
        sorting: bool = True,
    ) -> List[Tuple[float, bool]]:
        if self.batch_size > 1:
            if not sorting:
                logger.info(
                    "Sorting can reduce padding and therefore increase throughput considerably."
                )
        else:
            # Sorting with batch size 1 has no effect, overriding sorting = False
            sorting = False

        # attach original indices and sort by length
        indexed_requests = list(enumerate(requests))

        if sorting:

            def _req_len(x):
                return len(self.tokenizer.tokenize(x[1].args[0])) + len(
                    self.tokenizer.tokenize(x[1].args[1])
                )

            start_time = time.time()
            indexed_requests.sort(key=_req_len)
            logger.info(f"Sorting of requests took {time.time() - start_time:.3f}s")

        results_with_idx: List[Tuple[int, Tuple[float, bool]]] = []

        # looping over batches
        for start in tqdm.tqdm(range(0, len(indexed_requests), self.batch_size)):
            batch = indexed_requests[start : start + self.batch_size]

            context_lens: List[int] = []
            continuation_ids_list: List[List[int]] = []
            input_ids_list: List[torch.Tensor] = []
            orig_indices: List[int] = []

            # tokenize batch
            for orig_idx, req in batch:
                context, continuation = req.args
                context_ids, continuation_ids, input_ids_raw = self._tokenize(
                    context, continuation
                )
                context_lens.append(len(context_ids))
                continuation_ids_list.append(continuation_ids)
                input_ids_list.append(torch.tensor(input_ids_raw, dtype=torch.long))
                orig_indices.append(orig_idx)

            # pad input ids with token id 0 to create a rectangular batch tensor
            # Note: we can use any token id here as the padding part of the logits
            # will be cut during post-processing before computing the log likelihood
            max_len = max(x.size(0) for x in input_ids_list)

            input_ids_batch: torch.Tensor = torch.full(
                (len(batch), max_len),
                0,
                dtype=torch.long,
                device=self.device,
            )

            for i, ids in enumerate(input_ids_list):
                input_ids_batch[i, : ids.size(0)] = ids.to(self.device)

            # forward
            with torch.no_grad():
                logits = self.wrapped_model(input_ids_batch)
                log_probs = F.log_softmax(logits, dim=-1)

            # post-process per sample
            for i in range(len(batch)):
                context_len = context_lens[i]
                continuation_ids = continuation_ids_list[i]
                continuation_probs = log_probs[
                    i, context_len - 1 : context_len - 1 + len(continuation_ids)
                ]
                loglikelihood = continuation_probs.gather(
                    1,
                    torch.tensor(continuation_ids, device=self.device).unsqueeze(1),
                ).squeeze(1)
                predicted = torch.argmax(continuation_probs, -1).tolist()
                greedy = predicted == continuation_ids
                results_with_idx.append(
                    (orig_indices[i], (loglikelihood.sum().item(), greedy))
                )

        # restore original request order
        if sorting:
            results_with_idx.sort(key=lambda x: x[0])

        return [r for _, r in results_with_idx]

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
