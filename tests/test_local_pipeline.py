import json
from pathlib import Path
import pytest
import torch


@pytest.fixture
def tiny_model(tmp_path):
    pytest.importorskip('transformers')
    pytest.importorskip('datasets')
    pytest.importorskip('accelerate')
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    vocabulary = {'<pad>': 0, '<eos>': 1, '<bos>': 2, '<unk>': 3,
                  'hello': 4, 'good': 5, 'bad': 6, 'yes': 7, 'no': 8,
                  'user': 9, 'assistant': 10, ':': 11, 'Instruction': 12,
                  'Response': 13, '###': 14, 'test': 15}
    raw = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>',
                  eos_token='<eos>', bos_token='<bos>', unk_token='<unk>')
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    config = LlamaConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=256, pad_token_id=0, eos_token_id=1, bos_token_id=2)
    model = LlamaForCausalLM(config)
    path = tmp_path / 'tiny'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_real_training_and_generation(tiny_model, tmp_path, monkeypatch):
    from alignment.training import train
    from alignment.data import read_rows, write_json
    from scripts.generate import main as generate
    import sys
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/sft.json').read_text())
    rows = [{'prompt': 'hello', 'response': 'good yes'},
            {'prompt': 'test', 'response': 'yes'},
            {'prompt': 'hello test', 'response': 'good'}]
    data = tmp_path / 'sft.json'
    write_json(data, rows)
    config.update(model=str(tiny_model), device='cpu', reference_device='cpu',
                  train_path=str(data), val_path=str(data), output_dir=str(tmp_path / 'sft'),
                  batch_size=1, gradient_accumulation=2, max_length=32, max_prompt_length=16,
                  eval_steps=1, log_steps=1)
    train(config, max_steps=2)
    assert read_rows(tmp_path / 'sft/metrics.jsonl')[-1]['step'] == 2
    assert (tmp_path / 'sft/best/model.safetensors').is_file()
    assert (tmp_path / 'sft/final/model.safetensors').is_file()
    preferences = tmp_path / 'prefs.json'
    write_json(preferences, [{'prompt': 'hello', 'chosen': 'good', 'rejected': 'bad'},
                            {'prompt': 'test', 'chosen': 'yes', 'rejected': 'no'}])
    checkpoint = str(tmp_path / 'sft/best')
    config.update(task='dpo', model=checkpoint, reference=checkpoint,
                  train_path=str(preferences), val_path=str(preferences),
                  output_dir=str(tmp_path / 'dpo'), gradient_accumulation=1, scheduler='constant')
    train(config, max_steps=2)
    metrics = read_rows(tmp_path / 'dpo/metrics.jsonl')
    assert metrics[0]['reward_accuracy'] == 0
    assert abs(metrics[0]['reward_margin']) < 1e-6
    assert (tmp_path / 'dpo/best/model.safetensors').is_file()
    prompts = tmp_path / 'prompts.json'
    write_json(prompts, [{'id': '0', 'instruction': 'hello', 'dataset': 'test'},
                         {'id': '1', 'instruction': 'test', 'dataset': 'test'}])
    generation_config = tmp_path / 'generation.json'
    write_json(generation_config, {'seed': 42, 'do_sample': False, 'max_new_tokens': 3,
                                  'max_prompt_length': 16, 'repetition_penalty': 1.0})
    arguments = ['generate', '--model', checkpoint, '--prompts', str(prompts), '--output',
                  str(tmp_path / 'generations.json'), '--name', 'tiny', '--config', str(generation_config)]
    monkeypatch.setattr(sys, 'argv', arguments)
    generate()
    results = read_rows(tmp_path / 'generations.json')
    assert [row['id'] for row in results] == ['0', '1']
    generate()
    assert read_rows(tmp_path / 'generations.json') == results


def test_real_trl_persona_loop(tiny_model, tmp_path):
    pytest.importorskip('trl')
    pytest.importorskip('peft')
    from scripts.train_persona import train_persona
    from alignment.data import write_json
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/persona.json').read_text())
    data = tmp_path / 'persona.json'
    rows = []
    for index in range(6):
        rows.append({'prompt': 'hello test ' + str(index), 'chosen': 'good yes', 'rejected': 'bad no'})
    write_json(data, rows)
    config.update(model=str(tiny_model), dataset=str(data), output_dir=str(tmp_path / 'persona'),
                  gradient_accumulation=1, epochs=1, max_length=32, max_prompt_length=16)
    train_persona(config, pilot=True, cpu=True)
    assert (tmp_path / 'persona/best/adapter_model.safetensors').is_file()
    assert (tmp_path / 'persona/final/adapter_model.safetensors').is_file()
    result = json.loads((tmp_path / 'persona/config.json').read_text())
    assert 0 < result['trainable_parameters'] < result['total_parameters']
