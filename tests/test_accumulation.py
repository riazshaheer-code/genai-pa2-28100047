import copy
import torch
from alignment.core import build_sft_batch, response_logps_from_batch
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_accumulated_gradients_match_one_batch_with_unequal_response_lengths():
    tokenizer = ToyTokenizer()
    torch.manual_seed(7)
    direct = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.0)
    accumulated = copy.deepcopy(direct)
    prompts = ['a', 'ab', 'abc']
    responses = ['x', 'yz', 'xyz']
    batch = build_sft_batch(tokenizer, prompts, responses, 32)
    logps, counts = response_logps_from_batch(direct, batch)
    (-logps.sum() / counts.sum()).backward()

    divisor = 2 * 2 * 32
    target_count = 0
    for start, stop in [(0, 2), (2, 3)]:
        batch = build_sft_batch(tokenizer, prompts[start:stop], responses[start:stop], 32)
        logps, counts = response_logps_from_batch(accumulated, batch)
        (-logps.sum() / divisor).backward()
        target_count += int(counts.sum())
    for parameter in accumulated.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(divisor / target_count)
    for left, right in zip(direct.parameters(), accumulated.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
