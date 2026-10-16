import json
import random
import sys
from types import SimpleNamespace

from alignment.data import digest, read_rows, write_json


def test_missing_subset_uses_saved_revision_and_checks_hash(tmp_path, monkeypatch):
    from scripts import prepare_data

    repositories = {
        'sft': 'HuggingFaceH4/ultrachat_200k',
        'hh': 'Anthropic/hh-rlhf',
        'uf': 'HuggingFaceH4/ultrafeedback_binarized'
    }
    revisions = {}
    for repository in list(repositories.values()) + ['tatsu-lab/alpaca_eval']:
        revisions[repository] = 'saved-revision'
    expected = {}
    for name in repositories:
        for split in ['train', 'val']:
            row = {'prompt': name + split}
            if name == 'sft':
                row['response'] = 'answer'
            else:
                row['chosen'] = 'good'
                row['rejected'] = 'bad'
            expected[name + '_' + split + '.json'] = [row]
    source = []
    for index in range(300):
        source.append({'instruction': 'prompt ' + str(index)})
    source_path = tmp_path / 'alpaca.json'
    write_json(source_path, source)
    evaluation = []
    for index in random.Random(42).sample(range(300), 300):
        evaluation.append({'id': 'alpaca-' + str(index).zfill(4),
                           'instruction': source[index]['instruction'],
                           'dataset': 'alpaca_eval', 'source_index': index})
    expected['alpaca_eval_300_seed42.json'] = evaluation
    output = tmp_path / 'prepared'
    output.mkdir()
    manifest = {'seed': 42, 'dataset_revisions': revisions, 'files': {}}
    for filename in expected:
        rows = expected[filename]
        manifest['files'][filename] = {'rows': len(rows), 'sha256': digest(rows)}
        if filename != 'sft_train.json':
            write_json(output / filename, rows)
    write_json(output / 'manifest.json', manifest)

    def load_dataset(repository, split, revision):
        assert revision == 'saved-revision'
        for name in repositories:
            if repository == repositories[name]:
                subset = 'train'
                if split.startswith('test'):
                    subset = 'val'
                return expected[name + '_' + subset + '.json']
        raise ValueError('Unexpected repository')

    def list_files(repository, repo_type, revision):
        assert revision == 'saved-revision'
        return ['alpaca_eval.json']

    monkeypatch.setattr(prepare_data, 'HfApi', lambda: SimpleNamespace(list_repo_files=list_files))
    monkeypatch.setattr(prepare_data, 'load_dataset', load_dataset)
    monkeypatch.setattr(prepare_data, 'choose_rows', lambda source, extractor, count, seed, **kwargs: source)
    monkeypatch.setattr(prepare_data, 'hf_hub_download', lambda *args, **kwargs: str(source_path))
    monkeypatch.setattr(sys, 'argv', ['prepare_data', '--out', str(output)])
    prepare_data.main()
    assert read_rows(output / 'sft_train.json') == expected['sft_train.json']
    saved = json.loads((output / 'manifest.json').read_text())
    assert saved['dataset_revisions'] == revisions
    assert saved['files']['sft_train.json']['sha256'] == digest(expected['sft_train.json'])
