import argparse
from pathlib import Path
from alignment.data import read_rows
from alignment.evaluation import build_pairs, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    baseline = read_rows(args.baseline)
    candidate = read_rows(args.candidate)
    pairs = build_pairs(baseline, candidate, args.seed)
    destination = Path(args.output)
    if destination.exists():
        if read_rows(destination) != pairs:
            raise ValueError('Existing comparison differs. Use another output path.')
        print('Comparison already saved:', destination)
        return
    write_jsonl(destination, pairs)
    print('Saved', len(pairs), 'pairs:', destination)


if __name__ == '__main__':
    main()
