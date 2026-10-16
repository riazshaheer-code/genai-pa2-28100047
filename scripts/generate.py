import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.data import capped_prompt, digest, read_rows, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--config', default='configs/generation.json')
    parser.add_argument('--persona', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch_size must be positive.')
    settings = json.loads(Path(args.config).read_text())
    prompts = read_rows(args.prompts)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('limit must be positive.')
        prompts = prompts[:args.limit]
    seen = set()
    for row in prompts:
        if row['id'] in seen:
            raise ValueError('Duplicate prompt id.')
        seen.add(row['id'])
    metadata = {}
    metadata['model'] = args.model
    metadata['adapter'] = args.adapter
    metadata['generator'] = args.name
    metadata['settings'] = settings
    metadata['persona'] = args.persona
    metadata['prompt_sha256'] = digest(prompts)
    path = Path(args.output)
    metadata_path = path.with_suffix('.meta.json')
    results = []
    if path.exists():
        if not metadata_path.is_file():
            raise ValueError('Existing generations have no settings file.')
        previous = json.loads(metadata_path.read_text())
        if previous != metadata:
            raise ValueError('Generation settings changed. Use another output filename.')
        results = read_rows(path)
        if len(results) > len(prompts):
            raise ValueError('Existing output is longer than the prompt set.')
        for index in range(len(results)):
            if results[index]['id'] != prompts[index]['id']:
                raise ValueError('Saved generation order does not match.')
            if results[index]['instruction'] != prompts[index]['instruction']:
                raise ValueError('Saved generation prompt does not match.')
    if len(results) == len(prompts):
        print('All generations already exist:', path)
        return
    write_json(metadata_path, metadata)
    torch.manual_seed(settings['seed'])
    device = 'cpu'
    dtype = torch.float32
    if torch.cuda.is_available():
        device = 'cuda:0'
        dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('Generation needs a padding token.')
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                attn_implementation='sdpa')
    if args.adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    start = len(results)
    while start < len(prompts):
        rows = prompts[start:start + args.batch_size]
        formatted = []
        for row in rows:
            instruction = row['instruction']
            if args.persona:
                messages = [{'role': 'user', 'content': instruction}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = capped_prompt(tokenizer, instruction, settings['max_prompt_length'])
            formatted.append(text)
        inputs = tokenizer(formatted, return_tensors='pt', padding=True, add_special_tokens=False,
                           return_token_type_ids=False)
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        generation_args = dict(settings)
        generation_args.pop('seed')
        generation_args.pop('max_prompt_length')
        generation_args['pad_token_id'] = tokenizer.pad_token_id
        generation_args['eos_token_id'] = tokenizer.eos_token_id
        with torch.inference_mode():
            sequences = model.generate(**inputs, **generation_args)
        input_width = inputs['input_ids'].shape[1]
        for index in range(len(rows)):
            output_tokens = sequences[index, input_width:]
            text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            item = dict(rows[index])
            item['output'] = text.strip()
            item['generator'] = args.name
            item['generation_settings'] = settings
            item['formatted_prompt'] = formatted[index]
            item['chat_template_used'] = args.persona
            results.append(item)
        write_json(path, results)
        start = len(results)
        print('Saved', start, '/', len(prompts), flush=True)
    print('Generation complete:', path)


if __name__ == '__main__':
    main()
