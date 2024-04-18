import torch
import torch.nn.functional as F
from torch import nn

from fms.utils import evaluation
from fms.utils.tokenizers import get_tokenizer

from lm_eval.api.instance import Instance


class ModelMock(nn.Module):
    def __init__(self, next_tokens):
        super().__init__()
        self.next_tokens = next_tokens

    def forward(self, inputs, **kwargs):
        results = []
        for i in range(inputs.shape[0]):
            results.append(self.forward_one(inputs[i], **kwargs))
        return torch.stack(results, 0)

    def forward_one(self, inputs, **kwargs):
        inputs = inputs.view(inputs.numel())

        next_token = torch.tensor([self.next_tokens[0]])
        self.next_tokens = self.next_tokens[1:]

        inputs = torch.cat((inputs[1:], next_token), -1)
        inputs = F.one_hot(inputs, 256).float()
        return inputs


def test_eval():
    tokenizer = get_tokenizer("char_tokenizer")
    model = ModelMock([ord("a"), ord("d")])

    lm_eval = evaluation.FMSEvalHarnessLM(model, tokenizer, "cpu")
    instance = Instance(
        request_type="loglikelihood", doc={}, arguments=("hello", "world"), idx=0
    )

    # first time we predict 'a' as last letter, incorrect
    result = lm_eval.loglikelihood([instance])
    assert not result[0][1]

    # then predict d, correct
    result = lm_eval.loglikelihood([instance])
    assert result[0][1]
