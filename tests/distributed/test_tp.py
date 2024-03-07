import torch
import torch.nn

from fms.distributed.tensorparallel import (
    apply_colwise_tp,
    apply_embedding_tp,
    apply_rowwise_tp,
)


def test_apply_colwise_tp():
    linear_mod1 = torch.nn.Linear(1000, 1000, bias=False)
    tp_linear_mod1 = torch.nn.Linear(1000, 500, bias=False)
    linear_mod2 = torch.nn.Linear(1000, 1000, bias=True)
    tp_linear_mod2 = torch.nn.Linear(1000, 500, bias=True)

    apply_colwise_tp(tp_linear_mod1, linear_mod1, 0, 2)
    apply_colwise_tp(tp_linear_mod2, linear_mod2, 0, 2)

    torch.testing.assert_close(linear_mod1.weight[:500, :], tp_linear_mod1.weight)
    torch.testing.assert_close(linear_mod2.bias[:500], tp_linear_mod2.bias)


def test_apply_rowwise_tp():
    linear_mod1 = torch.nn.Linear(1000, 1000, bias=False)
    tp_linear_mod1 = torch.nn.Linear(500, 1000, bias=False)
    linear_mod2 = torch.nn.Linear(1000, 1000, bias=True)
    tp_linear_mod2 = torch.nn.Linear(500, 1000, bias=True)

    apply_rowwise_tp(tp_linear_mod1, linear_mod1, 1, 2)
    apply_rowwise_tp(tp_linear_mod2, linear_mod2, 1, 2)

    torch.testing.assert_close(linear_mod1.weight[:, 500:], tp_linear_mod1.weight)
    torch.testing.assert_close(tp_linear_mod2.bias, torch.zeros((1000,)))


def test_apply_embedding_tp():
    embedding_mod = torch.nn.Embedding(32000, 1000)
    tp_embedding_mod = torch.nn.Embedding(32000, 500)

    apply_embedding_tp(tp_embedding_mod, embedding_mod, 0, 2)

    torch.testing.assert_close(embedding_mod.weight[:, :500], tp_embedding_mod.weight)
