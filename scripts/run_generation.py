import argparse
import subprocess
import sys
from pathlib import Path
from alignment.data import read_rows
from alignment.evaluation import build_pairs, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_root', default='outputs')
    parser.add_argument('--prepared_root', default='data/prepared')
    parser.add_argument('--output_root', default='outputs/generations')
    parser.add_argument('--phase', choices=['baseline', 'trained', 'all'], default='all')
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    checkpoints = Path(args.checkpoint_root)
    output = Path(args.output_root)
    if args.sanity:
        output = output / 'sanity'
    alpaca = str(Path(args.prepared_root) / 'alpaca_eval_300_seed42.json')
    hh = 'eval_sets/hh_rlhf_eval_300_seed42.json'
    persona_prompts = 'eval_sets/persona_eval.json'
    jobs = []
    if args.phase in ['baseline', 'all']:
        jobs.append(('base_alpaca', 'Qwen/Qwen2.5-0.5B', alpaca, False, None))
        jobs.append(('base_hh', 'Qwen/Qwen2.5-0.5B', hh, False, None))
    if args.phase in ['trained', 'all']:
        jobs.append(('sft_alpaca', str(checkpoints / 'sft/best'), alpaca, False, None))
        jobs.append(('sft_hh', str(checkpoints / 'sft/best'), hh, False, None))
        jobs.append(('dpo_uf_alpaca', str(checkpoints / 'dpo_uf/best'), alpaca, False, None))
        jobs.append(('dpo_hh_hh', str(checkpoints / 'dpo_hh/best'), hh, False, None))
        base = 'HuggingFaceTB/SmolLM2-360M-Instruct'
        jobs.append(('smollm_persona', base, persona_prompts, True, None))
        jobs.append(('lora_persona', base, persona_prompts, True, str(checkpoints / 'persona/best')))
    for name, model, prompts, persona, adapter in jobs:
        command = [sys.executable, '-m', 'scripts.generate', '--model', model,
                   '--prompts', prompts, '--name', name, '--output', str(output / (name + '.json'))]
        if persona:
            command.append('--persona')
        if adapter is not None:
            command.extend(['--adapter', adapter])
        if args.sanity:
            command.extend(['--limit', '3'])
        subprocess.run(command, check=True)
        if args.sanity:
            for row in read_rows(output / (name + '.json')):
                print('\nPROMPT:', row['instruction'])
                print('OUTPUT:', row['output'])
    if not args.sanity and args.phase in ['trained', 'all']:
        comparisons = [('hh', 'sft_hh', 'dpo_hh_hh'),
                       ('uf', 'sft_alpaca', 'dpo_uf_alpaca'),
                       ('persona', 'smollm_persona', 'lora_persona')]
        for track, baseline, candidate in comparisons:
            baseline_rows = read_rows(output / (baseline + '.json'))
            candidate_rows = read_rows(output / (candidate + '.json'))
            pairs = build_pairs(baseline_rows, candidate_rows, 42)
            path = output / 'pairs' / (track + '.jsonl')
            if path.exists() and read_rows(path) != pairs:
                raise ValueError('Existing pairs differ: ' + str(path))
            write_jsonl(path, pairs)
            print('Saved comparison:', path)


if __name__ == '__main__':
    main()
