import contextlib
import functools
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from alignment.core import build_dpo_batch, build_sft_batch, dpo_forward, response_logps_from_batch
from alignment.data import PROMPT_TEMPLATE, capped_prompt, digest, read_rows, write_json


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_metric(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()


def autocast_context(device):
    if str(device).startswith('cuda'):
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def set_dropout_zero(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def collate_rows(rows, tokenizer, config):
    prompts = []
    responses = []
    chosen = []
    rejected = []
    for row in rows:
        prompt = capped_prompt(tokenizer, row['prompt'], config['max_prompt_length'])
        prompts.append(prompt)
        if config['task'] == 'sft':
            responses.append(row['response'])
        else:
            chosen.append(row['chosen'])
            rejected.append(row['rejected'])
    if config['task'] == 'sft':
        batch = build_sft_batch(tokenizer, prompts, responses, config['max_length'])
    else:
        batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, config['max_length'])
    return batch


def forward_batch(policy, reference, batch, config):
    if config['task'] == 'sft':
        logps, counts = response_logps_from_batch(policy, batch)
        if counts.eq(0).any():
            raise ValueError('A training row has no predictable response tokens.')
        return -logps.sum(), int(counts.sum()), {}
    losses, margins, chosen_counts, rejected_counts = dpo_forward(policy, reference, batch, config['beta'])
    if chosen_counts.eq(0).any() or rejected_counts.eq(0).any():
        raise ValueError('A preference row has no predictable response tokens.')
    diagnostics = {'correct': int((margins.detach() > 0).sum()),
                   'margin_sum': float(margins.detach().sum())}
    return losses.sum(), len(losses), diagnostics


def evaluate(policy, reference, loader, config):
    was_training = policy.training
    policy.eval()
    numerator = 0.0
    denominator = 0
    correct = 0
    margin_sum = 0.0
    with torch.no_grad(), autocast_context(config['device']):
        for batch in loader:
            loss_sum, count, diagnostics = forward_batch(policy, reference, batch, config)
            numerator += float(loss_sum)
            denominator += count
            correct += diagnostics.get('correct', 0)
            margin_sum += diagnostics.get('margin_sum', 0.0)
    result = {'loss': numerator / max(1, denominator)}
    if config['task'] == 'dpo':
        result.update(reward_accuracy=correct / max(1, denominator),
                       reward_margin=margin_sum / max(1, denominator))
    else:
        result['perplexity'] = math.exp(min(result['loss'], 80))
    policy.train(was_training)
    return result


def save_model(policy, tokenizer, destination, config):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    write_json(destination / 'run_config.json', config)


def train(config, max_steps=None, train_limit=None, val_limit=None):
    if config['task'] not in ['sft', 'dpo']:
        raise ValueError('task must be sft or dpo.')
    if config['batch_size'] < 1 or config['gradient_accumulation'] < 1:
        raise ValueError('Batch size and accumulation must be positive.')
    if config['epochs'] < 1 or config['learning_rate'] <= 0:
        raise ValueError('Epochs and learning rate must be positive.')
    if config['eval_steps'] < 1 or config['log_steps'] < 1:
        raise ValueError('Logging and evaluation intervals must be positive.')
    if config['scheduler'] not in ['constant', 'cosine']:
        raise ValueError('Unsupported scheduler.')
    if str(config['device']).startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('Enable a GPU before training.')
    if config['task'] == 'dpo':
        if config['model'] != config['reference']:
            raise ValueError('Start the policy and reference from the same SFT checkpoint.')
        if config['beta'] <= 0:
            raise ValueError('DPO beta must be positive.')
    seed_everything(config['seed'])
    output = Path(config['output_dir'])
    if (output / 'COMPLETE.json').exists():
        print('This run is already complete:', output)
        return
    if (output / 'metrics.jsonl').exists():
        raise RuntimeError(f'Partial run exists in {output}. Preserve it and use a new output directory; '
                           'this prevents silently mixing independent training histories.')
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(config['model'], torch_dtype=torch.float32,
                                                 attn_implementation='sdpa').to(config['device'])
    policy.config.use_cache = False
    if config['gradient_checkpointing']:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    reference = None
    if config['task'] == 'dpo':
        reference = AutoModelForCausalLM.from_pretrained(config['reference'], torch_dtype=torch.float32,
                                                        attn_implementation='sdpa').to(config['reference_device'])
        reference.requires_grad_(False)
        reference.eval()
        reference.config.use_cache = False
        set_dropout_zero(policy)
        set_dropout_zero(reference)
    train_rows = read_rows(config['train_path'])
    val_rows = read_rows(config['val_path'])
    if train_limit is not None:
        train_rows = train_rows[:train_limit]
    if val_limit is not None:
        val_rows = val_rows[:val_limit]
    if not train_rows or not val_rows:
        raise ValueError('Both training and validation data must be nonempty.')
    collator = functools.partial(collate_rows, tokenizer=tokenizer, config=config)
    train_loader = DataLoader(train_rows, batch_size=config['batch_size'], shuffle=True,
                               generator=torch.Generator().manual_seed(config['seed']), collate_fn=collator)
    val_loader = DataLoader(val_rows, batch_size=config['batch_size'], shuffle=False, collate_fn=collator)
    actual_config = dict(config)
    actual_config.update(train_examples=len(train_rows), val_examples=len(val_rows),
                          max_steps=max_steps, prompt_template=PROMPT_TEMPLATE,
                          sft_reduction='global mean over unmasked shifted response tokens',
                          reference_precision='fp32 weights with fp16 autocast',
                          checkpoint_precision='fp32', optimizer_state_saved=False,
                          train_data_sha256=digest(train_rows), val_data_sha256=digest(val_rows))
    import datasets
    import transformers
    actual_config['versions'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
                                  'datasets': datasets.__version__}
    actual_config['model_commit'] = getattr(policy.config, '_commit_hash', None)
    try:
        actual_config['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        actual_config['code_commit'] = None
    write_json(output / 'config.json', actual_config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    updates_per_epoch = math.ceil(len(train_loader) / config['gradient_accumulation'])
    total_steps = updates_per_epoch * config['epochs']
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    scheduler = None
    if config['scheduler'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps=math.ceil(total_steps * config['warmup_ratio']),
                    num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0, enabled=str(config['device']).startswith('cuda'))
    backward_divisor = config['batch_size'] * config['gradient_accumulation']
    if config['task'] == 'sft':
        backward_divisor = backward_divisor * config['max_length']
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    successful_updates = 0
    consecutive_skips = 0
    if config['task'] == 'dpo':
        best = -math.inf
    else:
        best = math.inf
    start_time = time.monotonic()
    initial = evaluate(policy, reference, val_loader, config)
    append_metric(output / 'metrics.jsonl', {'split': 'validation', 'step': 0, **initial})
    print('Initial validation:', initial, flush=True)
    for epoch in range(config['epochs']):
        group_loss = 0.0
        group_count = 0
        group_correct = 0
        group_margin = 0.0
        for index, batch in enumerate(train_loader):
            with autocast_context(config['device']):
                loss_sum, count, diagnostics = forward_batch(policy, reference, batch, config)
            if not torch.isfinite(loss_sum):
                raise FloatingPointError('Non-finite loss. Stop and inspect the pilot.')
            scaler.scale(loss_sum / backward_divisor).backward()
            group_loss += float(loss_sum.detach())
            group_count += count
            group_correct += diagnostics.get('correct', 0)
            group_margin += diagnostics.get('margin_sum', 0.0)
            boundary = ((index + 1) % config['gradient_accumulation'] == 0 or index + 1 == len(train_loader))
            if not boundary:
                continue
            scaler.unscale_(optimizer)
            for parameter in policy.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(backward_divisor / group_count)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config['gradient_clip'])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            if skipped:
                consecutive_skips += 1
            else:
                successful_updates += 1
                consecutive_skips = 0
            if consecutive_skips >= 8:
                raise FloatingPointError('Eight consecutive AMP updates were skipped. Stop and inspect gradients.')
            if scheduler is not None and not skipped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if group_count == 0:
                raise RuntimeError('The accumulated group has no targets.')
            metric = {'split': 'train', 'step': step, 'epoch': epoch + 1,
                       'loss': group_loss / group_count, 'learning_rate': optimizer.param_groups[0]['lr'],
                       'gradient_norm': float(grad_norm), 'amp_step_skipped': skipped,
                       'elapsed_seconds': time.monotonic() - start_time}
            if config['task'] == 'dpo':
                metric.update(reward_accuracy=group_correct / group_count,
                               reward_margin=group_margin / group_count)
            append_metric(output / 'metrics.jsonl', metric)
            if step % config['log_steps'] == 0:
                print(metric, flush=True)
            group_loss = 0.0
            group_count = 0
            group_correct = 0
            group_margin = 0.0
            finished = step >= total_steps
            if step % config['eval_steps'] == 0 or index + 1 == len(train_loader) or finished:
                validation = evaluate(policy, reference, val_loader, config)
                append_metric(output / 'metrics.jsonl', {'split': 'validation', 'step': step, **validation})
                if config['task'] == 'dpo':
                    score = validation['reward_accuracy']
                    improved = score > best
                else:
                    score = validation['loss']
                    improved = score < best
                if improved:
                    best = score
                    save_model(policy, tokenizer, output / 'best', actual_config)
                    write_json(output / 'best_metrics.json', {'step': step, **validation})
                print('Validation:', step, validation, 'best:', best, flush=True)
            if finished:
                break
        if step >= total_steps:
            break
    if successful_updates == 0:
        raise FloatingPointError('No optimizer update succeeded. This run must not be marked complete.')
    save_model(policy, tokenizer, output / 'final', actual_config)
    if str(config['device']).startswith('cuda'):
        actual_config['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated(config['device'])
        write_json(output / 'config.json', actual_config)
    write_json(output / 'COMPLETE.json', {'steps': step, 'successful_updates': successful_updates,
                                         'elapsed_seconds': time.monotonic() - start_time,
                                         'best_selection_score': best, 'pilot': max_steps is not None})
