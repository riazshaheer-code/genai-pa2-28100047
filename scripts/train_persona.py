import argparse
import json
import random
import shutil
import time
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import DPOConfig, DPOTrainer
from alignment.data import digest, read_rows, write_json


class SaveMetrics(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = dict(logs)
        row['step'] = state.global_step
        row['epoch'] = state.epoch
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')


def conversational_rows(rows):
    converted = []
    for row in rows:
        for field in ['prompt', 'chosen', 'rejected']:
            if field not in row or not isinstance(row[field], str) or row[field].strip() == '':
                raise ValueError('Invalid persona field: ' + field)
        item = {}
        item['prompt'] = [{'role': 'user', 'content': row['prompt']}]
        item['chosen'] = [{'role': 'assistant', 'content': row['chosen']}]
        item['rejected'] = [{'role': 'assistant', 'content': row['rejected']}]
        converted.append(item)
    return converted


def train_persona(config, pilot=False, cpu=False):
    output = Path(config['output_dir'])
    if (output / 'COMPLETE.json').exists():
        print('Persona run already complete:', output)
        return
    if (output / 'metrics.jsonl').exists():
        raise ValueError('Partial persona run found. Keep it and choose another output directory.')
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config['seed'])
    rows = read_rows(config['dataset'])
    if len(rows) < 2:
        raise ValueError('Persona training needs at least two examples.')
    indices = list(range(len(rows)))
    generator = random.Random(config['seed'])
    generator.shuffle(indices)
    validation_count = max(1, int(len(rows) * config['validation_fraction']))
    if validation_count >= len(rows):
        raise ValueError('The validation split leaves no training data.')
    validation = []
    training = []
    for index in range(len(indices)):
        row = rows[indices[index]]
        if index < validation_count:
            validation.append(row)
        else:
            training.append(row)
    write_json(output / 'train_rows.json', training)
    write_json(output / 'val_rows.json', validation)
    training_dataset = Dataset.from_list(conversational_rows(training))
    validation_dataset = Dataset.from_list(conversational_rows(validation))
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError('Persona training needs a chat template.')
    model = AutoModelForCausalLM.from_pretrained(config['model'], torch_dtype=torch.float32,
                                                attn_implementation='sdpa')
    model.config.use_cache = False
    target_counts = {}
    for target in config['target_modules']:
        target_counts[target] = 0
    for name, module in model.named_modules():
        last_name = name.split('.')[-1]
        if last_name in target_counts:
            target_counts[last_name] += 1
    for target in target_counts:
        if target_counts[target] == 0:
            raise ValueError('LoRA target missing from model: ' + target)
    lora = LoraConfig(r=config['rank'], lora_alpha=config['alpha'],
                      lora_dropout=config['dropout'], target_modules=config['target_modules'],
                      bias='none', task_type='CAUSAL_LM')
    max_steps = -1
    if pilot:
        max_steps = 2
    fp16 = False
    if not cpu and torch.cuda.is_available():
        fp16 = True
    arguments = DPOConfig(
        output_dir=str(output / 'checkpoints'),
        per_device_train_batch_size=config['batch_size'], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config['gradient_accumulation'],
        learning_rate=config['learning_rate'], num_train_epochs=config['epochs'], max_steps=max_steps,
        beta=config['beta'], max_length=config['max_length'],
        max_prompt_length=config['max_prompt_length'], truncation_mode='keep_start',
        fp16=fp16, bf16=False, use_cpu=cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        max_grad_norm=1.0, weight_decay=config['weight_decay'],
        lr_scheduler_type='cosine', warmup_ratio=config['warmup_ratio'], optim='adamw_torch',
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, save_only_model=True,
        load_best_model_at_end=False, metric_for_best_model='rewards/accuracies', greater_is_better=True,
        logging_steps=1, report_to='none', seed=config['seed'], data_seed=config['seed'],
        disable_dropout=True,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=arguments,
                          processing_class=tokenizer, peft_config=lora,
                          train_dataset=training_dataset, eval_dataset=validation_dataset,
                          callbacks=[SaveMetrics(output / 'metrics.jsonl')])
    total_parameters = 0
    trainable_parameters = 0
    for parameter in trainer.model.parameters():
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
    actual_config = dict(config)
    actual_config['train_examples'] = len(training)
    actual_config['val_examples'] = len(validation)
    actual_config['train_sha256'] = digest(training)
    actual_config['val_sha256'] = digest(validation)
    actual_config['target_module_counts'] = target_counts
    actual_config['total_parameters'] = total_parameters
    actual_config['trainable_parameters'] = trainable_parameters
    actual_config['reference'] = 'Original base weights with the LoRA adapter disabled (TRL PEFT reference).'
    actual_config['chat_template'] = tokenizer.chat_template
    actual_config['dropout_note'] = 'TRL disable_dropout=True also disables the requested LoRA dropout.'
    actual_config['training_arguments'] = arguments.to_dict()
    import transformers
    import trl
    import peft
    actual_config['versions'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
                                 'trl': trl.__version__, 'peft': peft.__version__}
    write_json(output / 'config.json', actual_config)
    print('Trainable parameters:', trainable_parameters, '/', total_parameters, flush=True)
    start = time.monotonic()
    trainer.train()
    trainer.save_model(str(output / 'final'))
    tokenizer.save_pretrained(output / 'final')
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint is None:
        raise RuntimeError('TRL did not select a best validation checkpoint.')
    best_path = output / 'best'
    best_path.mkdir(parents=True, exist_ok=True)
    for file in Path(best_checkpoint).iterdir():
        if file.is_file() and file.name.startswith(('adapter_', 'tokenizer', 'special_tokens', 'added_tokens', 'vocab', 'merges')):
            shutil.copy2(file, best_path / file.name)
    tokenizer.save_pretrained(best_path)
    write_json(output / 'trainer_history.json', trainer.state.log_history)
    write_json(output / 'COMPLETE.json', {'steps': trainer.state.global_step,
               'best_checkpoint': best_checkpoint, 'best_metric': trainer.state.best_metric,
               'elapsed_seconds': time.monotonic() - start, 'pilot': pilot})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/persona.json')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.pilot:
        config['output_dir'] += '_pilot'
    train_persona(config, args.pilot, args.cpu)


if __name__ == '__main__':
    main()
