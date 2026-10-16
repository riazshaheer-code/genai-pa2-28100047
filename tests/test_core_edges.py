import math
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from alignment import core
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_exact_next_token_numerics_and_token_weighting():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=2)
    prompts, responses = ['Task: ', 'Q: '], ['a', 'bc']
    expected = []
    log_distribution = model.logit_bias.log_softmax(-1)
    for response in responses:
        ids = tokenizer.encode(response) + [tokenizer.eos_token_id]
        expected.append(log_distribution[ids].sum())
    actual = core.compute_response_logprobs(model, tokenizer, prompts, responses, 30)
    torch.testing.assert_close(actual, torch.stack(expected))
    loss = core.compute_sft_loss(model, tokenizer, prompts, responses, 30)
    torch.testing.assert_close(loss, -torch.stack(expected).sum() / 5)


@pytest.mark.parametrize('length', [1, 3, 4, 5, 6, 7, 20])
def test_truncation_before_and_at_response_and_eos(length):
    tokenizer = ToyTokenizer()
    batch = core.build_sft_batch(tokenizer, ['abc'], ['def'], length)
    ids = (tokenizer.encode('abc') + tokenizer.encode('def') + [1])[:length]
    labels = ([-100] * 3 + tokenizer.encode('def') + [1])[:length]
    assert batch['input_ids'][0].tolist() == ids
    assert batch['labels'][0].tolist() == labels


def test_pad_equal_eos_does_not_mask_real_eos():
    tokenizer = ToyTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'  # Core explicitly right-pads training batches.
    batch = core.build_sft_batch(tokenizer, ['a', 'abc'], ['b', 'def'], 20)
    assert batch['labels'][0, 2].item() == 1
    assert batch['attention_mask'][0, 2].item() == 1
    assert batch['labels'][0, 3:].eq(-100).all()
    assert batch['attention_mask'][0, 3:].eq(0).all()


def test_missing_pad_and_eos_and_empty_strings():
    tokenizer = ToyTokenizer()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = None
    batch = core.build_sft_batch(tokenizer, ['', 'a'], ['', 'b'], 10)
    assert batch['attention_mask'][0].eq(0).all()
    model = ToyCausalLM(tokenizer.vocab_size)
    logps = core.compute_response_logprobs(model, tokenizer, ['', 'a'], ['', 'b'], 10)
    assert logps[0].item() == 0
    assert torch.isfinite(logps).all()


class BigramLM(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.table = nn.Parameter(torch.randn(vocabulary_size, vocabulary_size))

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.table[input_ids])


def test_shift_uses_previous_position_and_row_order():
    torch.manual_seed(9)
    tokenizer = ToyTokenizer()
    model = BigramLM(tokenizer.vocab_size)
    prompts, responses = ['ab', 'cd', 'ef'], ['xyz', 'b', 'cc']
    actual = core.compute_response_logprobs(model, tokenizer, prompts, responses, 30)
    expected = []
    for prompt, response in zip(prompts, responses):
        p = tokenizer.encode(prompt)
        r = tokenizer.encode(response) + [1]
        previous = [p[-1]] + r[:-1]
        expected.append(model.table.log_softmax(-1)[previous, r].sum())
    torch.testing.assert_close(actual, torch.stack(expected))
    reverse = core.compute_response_logprobs(model, tokenizer, prompts[::-1], responses[::-1], 30)
    torch.testing.assert_close(actual, reverse.flip(0))


def test_dpo_numerical_formula_extreme_logits_and_gradients():
    chosen = torch.tensor([10000., -10000., 2.], requires_grad=True)
    rejected = torch.tensor([0., 0., -3.], requires_grad=True)
    reference_chosen = torch.tensor([0., 0., 1.])
    reference_rejected = torch.tensor([0., 0., -1.])
    loss = core.compute_dpo_loss_from_logps(chosen, rejected, reference_chosen, reference_rejected, 0.1)
    expected = torch.nn.functional.softplus(-0.1 * ((chosen - rejected) - (reference_chosen - reference_rejected))).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(chosen.grad <= 0)
    assert torch.all(rejected.grad >= 0)


def test_dpo_small_step_improves_margin_reference_unchanged():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = ToyCausalLM(tokenizer.vocab_size)
    reference_before = {k: v.clone() for k, v in reference.state_dict().items()}
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.)
    batch = core.build_dpo_batch(tokenizer, ['Pick: '], ['a'], ['b'], 20)
    losses, initial_margin, _, _ = core.dpo_forward(policy, reference, batch, 0.1)
    torch.testing.assert_close(losses, torch.tensor([math.log(2)]))
    assert (initial_margin > 0).float().mean().item() == 0
    losses.mean().backward()
    optimizer.step()
    _, updated_margin, _, _ = core.dpo_forward(policy, reference, batch, 0.1)
    assert updated_margin.item() > initial_margin.item()
    assert all(parameter.grad is None for parameter in reference.parameters())
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, reference_before[key])


def test_no_targets_zero_loss_still_backward_safe():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    loss = core.compute_sft_loss(model, tokenizer, ['abcdef'], ['ok'], 2)
    assert loss.item() == 0
    loss.backward()
    assert model.logit_bias.grad.eq(0).all()


@pytest.mark.parametrize('prompts,responses,length', [([], [], 10), (['a'], [], 10), (['a'], ['b'], 0)])
def test_invalid_batches_fail_explicitly(prompts, responses, length):
    with pytest.raises(ValueError):
        core.build_sft_batch(ToyTokenizer(), prompts, responses, length)
