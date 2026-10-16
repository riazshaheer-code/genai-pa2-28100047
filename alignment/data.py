import hashlib
import json
import random
from pathlib import Path

PROMPT_TEMPLATE = '### Instruction:\n{instruction}\n\n### Response:\n'


def format_prompt(instruction):
    instruction = instruction.strip()
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    return prompt


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    text = json.dumps(value, ensure_ascii=False, indent=2)
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing data file: ' + str(path))
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.jsonl':
        rows = []
        for line in text.splitlines():
            if line.strip() != '':
                row = json.loads(line)
                rows.append(row)
    else:
        rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError('Data must be a list: ' + str(path))
    return rows


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    raw = text.encode('utf-8')
    result = hashlib.sha256(raw).hexdigest()
    return result


def single_turn(messages):
    if not isinstance(messages, list):
        return None
    if len(messages) != 2:
        return None
    if not isinstance(messages[0], dict) or not isinstance(messages[1], dict):
        return None
    if messages[0].get('role') != 'user':
        return None
    if messages[1].get('role') != 'assistant':
        return None
    prompt = messages[0].get('content', '')
    response = messages[1].get('content', '')
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None
    prompt = prompt.strip()
    response = response.strip()
    if prompt == '' or response == '':
        return None
    return prompt, response


def extract_sft(row):
    messages = row.get('messages')
    if not isinstance(messages, list):
        return None
    if len(messages) < 2:
        return None
    first_exchange = messages[:2]
    pair = single_turn(first_exchange)
    if pair is None:
        return None
    result = {}
    result['prompt'] = pair[0]
    result['response'] = pair[1]
    return result


def extract_uf(row):
    chosen = single_turn(row.get('chosen'))
    rejected = single_turn(row.get('rejected'))
    if chosen is None or rejected is None:
        return None
    if chosen[0] != rejected[0]:
        return None
    if chosen[1] == rejected[1]:
        return None
    result = {}
    result['prompt'] = chosen[0]
    result['chosen'] = chosen[1]
    result['rejected'] = rejected[1]
    return result


def extract_hh(row):
    parsed = []
    for side in ['chosen', 'rejected']:
        text = row.get(side, '')
        if not isinstance(text, str):
            return None
        if text.count('\n\nHuman:') != 1:
            return None
        if text.count('\n\nAssistant:') != 1:
            return None
        human, assistant = text.split('\n\nAssistant:', 1)
        prompt = human.removeprefix('\n\nHuman:').strip()
        response = assistant.strip()
        if prompt == '' or response == '':
            return None
        parsed.append((prompt, response))
    if parsed[0][0] != parsed[1][0]:
        return None
    if parsed[0][1] == parsed[1][1]:
        return None
    result = {}
    result['prompt'] = parsed[0][0]
    result['chosen'] = parsed[0][1]
    result['rejected'] = parsed[1][1]
    return result


def choose_rows(source, extractor, count, seed, excluded_prompts=None):
    candidates = []
    seen = set()
    if excluded_prompts is not None:
        for prompt in excluded_prompts:
            seen.add(prompt)
    for row in source:
        item = extractor(row)
        if item is None:
            continue
        prompt = item['prompt']
        if prompt in seen:
            continue
        seen.add(prompt)
        candidates.append(item)
    random_generator = random.Random(seed)
    random_generator.shuffle(candidates)
    if len(candidates) < count:
        raise ValueError('Not enough unique single-turn rows: ' + str(len(candidates)))
    selected = candidates[:count]
    return selected


def capped_prompt(tokenizer, instruction, max_tokens):
    prompt = format_prompt(instruction)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return prompt
    empty_prompt = format_prompt('')
    template_tokens = tokenizer.encode(empty_prompt, add_special_tokens=False)
    available = max_tokens - len(template_tokens) - 2
    if available < 1:
        raise ValueError('The prompt limit cannot fit the template.')
    tokens = tokenizer.encode(instruction.strip(), add_special_tokens=False)
    tokens = tokens[:available]
    while len(tokens) > 0:
        shortened_instruction = tokenizer.decode(tokens, skip_special_tokens=False)
        prompt = format_prompt(shortened_instruction)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) <= max_tokens:
            return prompt
        tokens = tokens[:-1]
    raise ValueError('Could not shorten the prompt enough.')
