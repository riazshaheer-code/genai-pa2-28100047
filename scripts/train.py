import argparse
import json
from pathlib import Path
from alignment.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pilot', action='store_true', help='Separate 2-step run on 64 train / 16 validation rows.')
    parser.add_argument('--model')
    parser.add_argument('--reference')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ['model', 'reference', 'output_dir']:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.pilot:
        config['output_dir'] += '_pilot'
        config['eval_steps'] = 1
        config['log_steps'] = 1
        config['warmup_ratio'] = 0.0
        train(config, max_steps=2, train_limit=64, val_limit=16)
    else:
        train(config)


if __name__ == '__main__':
    main()
