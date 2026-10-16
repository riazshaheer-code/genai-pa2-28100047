import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alignment.data import read_rows, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', default='outputs')
    parser.add_argument('--destination', default='outputs/report_assets')
    args = parser.parse_args()
    output = Path(args.outputs)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    configurations = {}
    for name in ['sft', 'dpo_hh', 'dpo_uf', 'persona']:
        path = output / name / 'metrics.jsonl'
        if not path.is_file():
            continue
        rows = read_rows(path)
        metrics = ['loss']
        if name in ['dpo_hh', 'dpo_uf']:
            metrics.extend(['reward_accuracy', 'reward_margin'])
        if name == 'persona':
            metrics.extend(['rewards/accuracies', 'rewards/margins'])
        figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
        for index in range(len(metrics)):
            metric = metrics[index]
            ax = axes[0, index]
            for split in ['train', 'validation']:
                steps = []
                values = []
                for row in rows:
                    key = metric
                    row_split = row.get('split')
                    if name == 'persona':
                        row_split = 'train'
                        if 'eval_loss' in row:
                            row_split = 'validation'
                            key = 'eval_' + metric
                    if row_split == split and key in row:
                        steps.append(row['step'])
                        values.append(row[key])
                if len(values) > 0:
                    ax.plot(steps, values, label=split, marker='.', linewidth=1.1)
            ax.set_title(name + ': ' + metric)
            ax.set_xlabel('Optimizer step')
            ax.set_ylabel(metric)
            ax.grid(alpha=0.2)
            if len(ax.lines) > 0:
                ax.legend()
        figure.suptitle('Raw logged metrics; no smoothing')
        figure.tight_layout()
        figure.savefig(destination / (name + '.png'), dpi=180)
        plt.close(figure)
        config_path = output / name / 'config.json'
        if config_path.is_file():
            import json
            configurations[name] = json.loads(config_path.read_text())
    write_json(destination / 'actual_run_configurations.json', configurations)
    print('Plots and actual configurations:', destination)


if __name__ == '__main__':
    main()
