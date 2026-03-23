from typing import List, Tuple

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

logger = logging.getLogger(__name__)


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
        self,
        context: str,
        continuation: str
    ) -> Tuple[List[int], List[int], List[int]]:

        context_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(context)
        )
        if not len(context_ids):
            context_ids = [self.tokenizer.bos_token_id]

        continuation_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(continuation)
        )
        input_ids = context_ids + continuation_ids[:-1]

        return context_ids, continuation_ids, input_ids

    def loglikelihood(
        self,
        requests: List[Instance],
        sorting: bool = True,
    ) -> List[Tuple[float, bool]]:

        if self.batch_size > 1:
            if not sorting:
                logger.info('Sorting can reduce padding and therefore increase throughput considerably.')
        else:
            # Sorting with batch size 1 has no effect, overriding sorting = False
            sorting = False
        
        # attach original indices and sort by length
        indexed_requests = list(enumerate(requests))

        _req_len = lambda x: (
            len(self.tokenizer.tokenize(x[1].args[0]))
            + len(self.tokenizer.tokenize(x[1].args[1]))
        )

        if sorting:
            start_time = time.time()
            indexed_requests.sort(key=_req_len)
            logger.info(f'Sorting of requests took {time.time() - start_time:.3f}s')

        results_with_idx: List[Tuple[int, Tuple[float, bool]]] = []

        # getting the pad token id and default to EOS token id if no pad id found
        # Note: is safe because padding tokens are never attended since masked out
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            logger.warning('pad_token_id not provided for this tokenizer, defaulting to eos_token_id.')
            pad_id = getattr(self.tokenizer, "eos_token_id")

        # looping over batches
        for start in tqdm.tqdm(range(0, len(indexed_requests), self.batch_size)):
            batch = indexed_requests[start : start + self.batch_size]

            context_lens = []
            continuation_ids_list = []
            input_ids_list = []
            orig_indices = []

            # tokenize batch
            for orig_idx, req in batch:
                context, continuation = req.args
                context_ids, continuation_ids, input_ids = self._tokenize(context, continuation)
                context_lens.append(len(context_ids))
                continuation_ids_list.append(continuation_ids)
                input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
                orig_indices.append(orig_idx)

            # pad inputs ids
            max_len = max(x.size(0) for x in input_ids_list)

            input_ids = torch.full(
                (len(batch), max_len),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )

            for i, ids in enumerate(input_ids_list):
                input_ids[i, : ids.size(0)] = ids.to(self.device)

            # forward
            with torch.no_grad():
                logits = self.wrapped_model(input_ids)
                log_probs = F.log_softmax(logits, dim=-1)

            # post-process per sample
            for i in range(len(batch)):
                context_len = context_lens[i]
                continuation_ids = continuation_ids_list[i]
                continuation_probs = log_probs[i, context_len - 1 : context_len - 1 + len(continuation_ids)]
                loglikelihood = continuation_probs.gather(
                    1, torch.tensor(continuation_ids, device=self.device).unsqueeze(1),
                ).squeeze(1)
                predicted = torch.argmax(continuation_probs, -1).tolist()
                greedy = predicted == continuation_ids
                results_with_idx.append((orig_indices[i], (loglikelihood.sum().item(), greedy)))

        # restore original request order
        if sorting:
            results_with_idx.sort(key=lambda x: x[0])
        
        return [r for _, r in results_with_idx]

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("not implemented yet")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("not implemented yet")
