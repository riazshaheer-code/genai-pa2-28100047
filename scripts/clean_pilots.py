import json
import shutil
from pathlib import Path


def main():
    for run in Path('outputs').glob('*_pilot'):
        marker = run / 'COMPLETE.json'
        if not marker.is_file():
            continue
        information = json.loads(marker.read_text())
        if information.get('pilot') is not True:
            continue
        for name in ['best', 'final', 'checkpoints']:
            path = run / name
            if path.is_dir():
                shutil.rmtree(path)
                print('Removed completed pilot weights:', path)


if __name__ == '__main__':
    main()
