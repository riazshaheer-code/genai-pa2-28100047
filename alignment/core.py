import torch
import torch.nn.functional as F


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if len(prompts) == 0:
        raise ValueError('The prompt list is empty.')
    if len(prompts) != len(responses):
        raise ValueError('Each prompt needs one response.')
    if max_length < 1:
        raise ValueError('max_length must be positive.')

    eos_id = getattr(tokenizer, 'eos_token_id', None)
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        if eos_id is not None:
            pad_id = eos_id
        else:
            pad_id = 0

    all_tokens = []
    all_labels = []
    largest_length = 1
    row = 0
    while row < len(prompts):
        if not isinstance(prompts[row], str):
            raise ValueError('Prompts must be strings.')
        if not isinstance(responses[row], str):
            raise ValueError('Responses must be strings.')

        prompt_tokens = tokenizer.encode(prompts[row], add_special_tokens=False)
        response_tokens = tokenizer.encode(responses[row], add_special_tokens=False)
        prompt_tokens = list(prompt_tokens)
        response_tokens = list(response_tokens)
        if eos_id is not None:
            response_tokens.append(eos_id)

        tokens = []
        labels = []
        for token in prompt_tokens:
            tokens.append(token)
            labels.append(-100)
        for token in response_tokens:
            tokens.append(token)
            labels.append(token)

        tokens = tokens[:max_length]
        labels = labels[:max_length]
        all_tokens.append(tokens)
        all_labels.append(labels)
        if len(tokens) > largest_length:
            largest_length = len(tokens)
        row = row + 1

    shape = (len(prompts), largest_length)
    input_ids = torch.full(shape, pad_id, dtype=torch.long)
    attention_mask = torch.zeros(shape, dtype=torch.long)
    labels = torch.full(shape, -100, dtype=torch.long)
    for row in range(len(prompts)):
        length = len(all_tokens[row])
        if length > 0:
            input_ids[row, :length] = torch.tensor(all_tokens[row], dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(all_labels[row], dtype=torch.long)

    batch = {}
    batch['input_ids'] = input_ids
    batch['attention_mask'] = attention_mask
    batch['labels'] = labels
    return batch


def model_device(model):
    for parameter in model.parameters():
        return parameter.device
    device = getattr(model, 'device', 'cpu')
    return torch.device(device)


def move_batch(batch, device):
    moved_batch = {}
    for name in batch:
        moved_batch[name] = batch[name].to(device)
    return moved_batch


def response_logps_from_batch(model, batch):
    device = model_device(model)
    batch = move_batch(batch, device)
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs['logits']

    shifted_labels = batch['labels'][:, 1:]
    response_mask = shifted_labels != -100
    response_counts = response_mask.sum(dim=1)
    response_logps = logits[:, 0, 0] * 0.0

    start = 0
    while start < shifted_labels.shape[1]:
        stop = start + 64
        if stop > shifted_labels.shape[1]:
            stop = shifted_labels.shape[1]
        targets = shifted_labels[:, start:stop]
        scores = logits[:, start:stop, :]
        scores = scores.float()
        flat_scores = scores.reshape(-1, scores.shape[-1])
        flat_targets = targets.reshape(-1)
        token_losses = F.cross_entropy(flat_scores, flat_targets,
                                       reduction='none', ignore_index=-100)
        token_losses = token_losses.reshape(targets.shape)
        chunk_loss = token_losses.sum(dim=1)
        response_logps = response_logps - chunk_loss
        start = stop
    return response_logps, response_counts


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    response_logps, response_counts = response_logps_from_batch(model, batch)
    return response_logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    response_logps, response_counts = response_logps_from_batch(model, batch)
    total_negative_logprob = -response_logps.sum()
    total_tokens = response_counts.sum()
    total_tokens = total_tokens.clamp_min(1)
    loss = total_negative_logprob / total_tokens
    return loss


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    chosen_batch = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    batch = {}
    batch['chosen'] = chosen_batch
    batch['rejected'] = rejected_batch
    return batch


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                               reference_chosen_logps, reference_rejected_logps, beta):
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    improvement = policy_gap - reference_gap
    scaled_improvement = beta * improvement
    log_probabilities = F.logsigmoid(scaled_improvement)
    losses = -log_probabilities
    loss = losses.mean()
    return loss


def dpo_forward(policy_model, reference_model, batch, beta):
    if reference_model is None:
        raise ValueError('DPO needs a reference model.')
    chosen_logps, chosen_counts = response_logps_from_batch(policy_model, batch['chosen'])
    rejected_logps, rejected_counts = response_logps_from_batch(policy_model, batch['rejected'])
    reference_was_training = reference_model.training
    reference_model.eval()
    try:
        with torch.no_grad():
            reference_chosen, unused = response_logps_from_batch(reference_model, batch['chosen'])
            reference_rejected, unused = response_logps_from_batch(reference_model, batch['rejected'])
    finally:
        reference_model.train(reference_was_training)

    reference_chosen = reference_chosen.to(chosen_logps.device)
    reference_rejected = reference_rejected.to(rejected_logps.device)
    chosen_rewards = beta * (chosen_logps - reference_chosen)
    rejected_rewards = beta * (rejected_logps - reference_rejected)
    margins = chosen_rewards - rejected_rewards
    losses = -F.logsigmoid(margins)
    return losses, margins, chosen_counts, rejected_counts


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses,
                     rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    losses, margins, chosen_counts, rejected_counts = dpo_forward(policy_model, reference_model, batch, beta)
    loss = losses.mean()
    return loss
