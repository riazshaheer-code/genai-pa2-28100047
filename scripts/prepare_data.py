import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from alignment.data import choose_rows, digest, extract_hh, extract_sft, extract_uf, read_rows, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/prepared')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError('Use seed 42 for the fixed assignment evaluation subset.')
    output = Path(args.out)
    manifest_path = output / 'manifest.json'
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['seed'] != args.seed:
            raise ValueError('Saved data uses another seed.')
        complete = True
        for filename in manifest['files']:
            information = manifest['files'][filename]
            if not (output / filename).is_file():
                complete = False
                continue
            rows = read_rows(output / filename)
            if digest(rows) != information['sha256']:
                raise ValueError('Saved data changed: ' + filename)
        if complete:
            print('Prepared data already exists. Reusing the saved subsets.')
            return
    api = HfApi()
    sft_id = 'HuggingFaceH4/ultrachat_200k'
    hh_id = 'Anthropic/hh-rlhf'
    uf_id = 'HuggingFaceH4/ultrafeedback_binarized'
    alpaca_id = 'tatsu-lab/alpaca_eval'
    if manifest is not None:
        revisions = manifest['dataset_revisions']
    else:
        revisions = {}
        for repo in [sft_id, hh_id, uf_id, alpaca_id]:
            information = api.dataset_info(repo)
            revisions[repo] = information.sha
    saved = {}
    for name, repo, extractor, train_split, validation_split, count in [
        ('sft', sft_id, extract_sft, 'train_sft', 'test_sft', 30000),
        ('hh', hh_id, extract_hh, 'train', 'test', 15000),
        ('uf', uf_id, extract_uf, 'train_prefs', 'test_prefs', 15000),
    ]:
        validation_source = load_dataset(repo, split=validation_split, revision=revisions[repo])
        validation = choose_rows(validation_source, extractor, 1000, args.seed)
        del validation_source
        training_source = load_dataset(repo, split=train_split, revision=revisions[repo])
        excluded = set()
        for row in validation:
            excluded.add(row['prompt'])
        train = choose_rows(training_source, extractor, count, args.seed, excluded_prompts=excluded)
        for split, rows in [('train', train), ('val', validation)]:
            filename = f'{name}_{split}.json'
            if manifest is not None:
                if digest(rows) != manifest['files'][filename]['sha256']:
                    raise ValueError('Recreated data differs from saved manifest: ' + filename)
            write_json(output / filename, rows)
            saved[filename] = {'rows': len(rows), 'sha256': digest(rows)}
        del training_source
    files = api.list_repo_files(alpaca_id, repo_type='dataset', revision=revisions[alpaca_id])
    candidates = []
    for file in files:
        if file.endswith('alpaca_eval.json'):
            candidates.append(file)
    if len(candidates) != 1:
        raise ValueError(f'Expected one AlpacaEval JSON; found {candidates}. Inspect the dataset repository.')
    filename = hf_hub_download(alpaca_id, candidates[0], repo_type='dataset', revision=revisions[alpaca_id])
    rows = json.loads(Path(filename).read_text(encoding='utf-8'))
    if len(rows) < 300:
        raise ValueError('AlpacaEval dataset has fewer than 300 prompts.')
    indices = random.Random(args.seed).sample(range(len(rows)), 300)
    evaluation = []
    for index in indices:
        item = {}
        item['id'] = f'alpaca-{index:04d}'
        item['instruction'] = rows[index]['instruction']
        item['dataset'] = 'alpaca_eval'
        item['source_index'] = index
        evaluation.append(item)
    if manifest is not None:
        expected = manifest['files']['alpaca_eval_300_seed42.json']['sha256']
        if digest(evaluation) != expected:
            raise ValueError('Recreated AlpacaEval subset differs from saved manifest.')
    write_json(output / 'alpaca_eval_300_seed42.json', evaluation)
    saved['alpaca_eval_300_seed42.json'] = {'rows': len(evaluation), 'sha256': digest(evaluation),
                                           'selection': 'random.Random(seed).sample on original row order'}
    write_json(output / 'manifest.json', {'seed': args.seed, 'dataset_revisions': revisions,
                                         'files': saved,
                                         'filter': 'SFT: first user-assistant exchange; preferences: strictly single-turn; unique prompts'})
    print(json.dumps(saved, indent=2))


if __name__ == '__main__':
    main()
