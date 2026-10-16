import json
import pytest
from alignment.data import read_rows
from alignment.evaluation import build_pairs, judge_prompt, parse_judgments, summarize
from scripts.judge_pairwise_kaggle import run_judging


def generations(name):
    rows = []
    for index in range(3):
        rows.append({'id': str(index), 'instruction': 'Question ' + str(index),
                     'output': name + ' response', 'generator': name,
                     'generation_settings': {'do_sample': False},
                     'formatted_prompt': 'Question ' + str(index), 'chat_template_used': False})
    return rows


def test_pair_mapping_blinding_and_settings():
    baseline = generations('baseline-checkpoint')
    candidate = generations('candidate-checkpoint')
    pairs = build_pairs(baseline, candidate[::-1], 42)
    assert [row['id'] for row in pairs] == ['0', '1', '2']
    for pair in pairs:
        position = pair['_metadata']['candidate_position'].lower()
        assert pair['response_' + position] == 'candidate-checkpoint response'
        pair['response_a'] = 'plain text A'
        pair['response_b'] = 'plain text B'
    prompt = judge_prompt(pairs, 'general')
    assert 'candidate-checkpoint' not in prompt
    assert 'baseline-checkpoint' not in prompt
    assert '_metadata' not in prompt
    candidate[0]['generation_settings'] = {'do_sample': True}
    with pytest.raises(ValueError, match='mismatch'):
        build_pairs(baseline, candidate, 42)


def test_judge_validation_and_win_denominators():
    pairs = build_pairs(generations('sft'), generations('dpo'), 42)
    raw = []
    for pair in pairs:
        raw.append({'id': pair['id'], 'winner': pair['_metadata']['candidate_position']})
    raw[1]['winner'] = 'tie'
    raw[2]['winner'] = 'invalid'
    parsed = parse_judgments('```json\n' + json.dumps(raw) + '\n```', pairs)
    summary = summarize(pairs, parsed, 'test-judge')
    assert summary['counts'] == {'win': 1, 'loss': 0, 'tie': 1, 'unknown': 1, 'failure': 0}
    assert summary['win_rate_excluding_ties'] == 1
    assert summary['win_rate_ties_half'] == 0.75
    with pytest.raises(ValueError):
        parse_judgments(json.dumps(raw[:-1]), pairs)
    raw[1]['id'] = raw[0]['id']
    with pytest.raises(ValueError):
        parse_judgments(json.dumps(raw), pairs)


def test_judge_retry_delay_resume_and_failure(tmp_path, monkeypatch):
    import scripts.judge_pairwise_kaggle as module
    pairs = build_pairs(generations('sft'), generations('dpo'), 42)
    clock = [0.0]
    calls = []
    def sleep(seconds):
        assert seconds >= 5
        clock[0] += seconds
    monkeypatch.setattr(module.time, 'sleep', sleep)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    def judge(prompt):
        calls.append(clock[0])
        if len(calls) == 1:
            return 'bad JSON'
        data = json.loads(prompt[prompt.index('[{'):])
        return json.dumps([{'id': row['id'], 'winner': 'tie'} for row in data])
    output = tmp_path / 'judged.jsonl'
    result = run_judging(pairs, output, 'fake', judge, batch_size=2, reuse=True)
    assert result['counts']['tie'] == 3
    assert calls == [0, 5, 10]
    run_judging(pairs, output, 'fake', judge, reuse=True)
    assert len(calls) == 3
    assert len(read_rows(output)) == 3
    def broken(prompt):
        raise RuntimeError('offline')
    result = run_judging(pairs, tmp_path / 'failed.jsonl', 'fake', broken, max_retries=2)
    assert result['counts']['failure'] == 3
    assert len(read_rows(tmp_path / 'failed.failures.jsonl')) == 3


def test_kaggle_task_returns_supported_dict_and_caller_returns_text(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from scripts.judge_pairwise_kaggle import kaggle_judge

    def task(**options):
        assert options['store_task'] is False
        assert options['store_run'] is False

        def decorate(function):
            assert function.__annotations__['return'] is dict

            def run(**arguments):
                result = function(**arguments)
                assert isinstance(result, dict)
                assert result['response'] == 'saved judge response'
                return result

            return SimpleNamespace(run=run)

        return decorate

    llm = SimpleNamespace(prompt=lambda prompt: 'saved judge response')
    fake = SimpleNamespace(llms={'judge': llm}, task=task,
                           chats=SimpleNamespace(new=lambda name: nullcontext()))
    monkeypatch.setitem(sys.modules, 'kaggle_benchmarks', fake)
    call = kaggle_judge('judge')
    assert call('first prompt') == 'saved judge response'
    assert call('second prompt') == 'saved judge response'
