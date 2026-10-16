import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.splitlines(keepends=True)}


def save_notebook(name, cells):
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python', 'version': '3.11'}},
                'nbformat': 4, 'nbformat_minor': 5}
    for index in range(len(cells)):
        cells[index]['id'] = 'cell-' + str(index)
    path = ROOT / 'notebooks' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')


def main():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if not path.is_file():
                continue
            if any(part in ['.git', '.pytest_cache', '__pycache__', 'notebooks', 'outputs', 'assets', 'docs'] for part in relative.parts):
                continue
            if 'prepared' in relative.parts:
                if path.name not in ['manifest.json', 'alpaca_eval_300_seed42.json']:
                    continue
            if relative.name in ['REPORT.md', 'REPORT.tex', 'REPORT_CHECKLIST.md', 'SUBMISSION_CHECKLIST.md', 'verified_results.json', 'source_archives.json']:
                continue
            if path.suffix == '.pdf':
                continue
            archive.write(path, relative.as_posix())
    payload = base64.b64encode(buffer.getvalue()).decode('ascii')
    bootstrap = '''import base64
import io
import os
import sys
import zipfile
from pathlib import Path

project = Path('/kaggle/working/pa2')
project.mkdir(parents=True, exist_ok=True)
source_zip = base64.b64decode(PAYLOAD)
with zipfile.ZipFile(io.BytesIO(source_zip)) as archive:
    for name in archive.namelist():
        destination = (project / name).resolve()
        if not destination.is_relative_to(project.resolve()):
            raise ValueError('Unexpected path in source archive.')
        contents = archive.read(name)
        if destination.is_file():
            if destination.read_bytes() == contents:
                continue
            if (project / 'outputs').exists():
                raise ValueError('Source differs from an existing run. Use a fresh notebook.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
os.chdir(project)
sys.path.insert(0, str(project))
print('Project files ready:', project)
'''
    bootstrap = bootstrap.replace('PAYLOAD', repr(payload))
    install = '''import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-persona.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
'''
    helpers = '''import json
import os
import shutil
import subprocess
import sys
import torch
import transformers
import datasets
from pathlib import Path

print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('datasets:', datasets.__version__)
print('GPU count:', torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
if torch.cuda.device_count() < 2:
    raise RuntimeError('Select GPU T4 x2 in the Kaggle settings.')

def run(arguments, persona=False):
    environment = os.environ.copy()
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    if persona:
        environment['CUDA_VISIBLE_DEVICES'] = '0'
    command = [sys.executable] + arguments
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, check=True, env=environment)

def free_space():
    usage = shutil.disk_usage('/kaggle/working')
    print('Free disk GiB:', round(usage.free / (1024 ** 3), 2))
    if usage.free < 5 * (1024 ** 3):
        raise RuntimeError('Preserve earlier outputs and free disk space before another training run.')
'''
    cells = [markdown('# PA2: training\nSelect **GPU T4 x2** and turn **Internet on**. Run the cells in order. Full runs can take most of an eight- to nine-hour session. The first cell unpacks the included source; no GitHub setup is needed to start.\n\nInspect each three-prompt sanity output before running the following full-generation cell. Save your notebook version and its outputs after training.'),
             code(bootstrap), code(install), code(helpers),
             markdown('## Check the implementation\nThese include the official tests and small local training checks. They do not train the assignment models.'),
             code("run(['-m', 'pytest', '-q', 'tests'])"),
             markdown('## Prepare fixed data\nDownload only the required splits, keep single-turn rows, and save exact subsets and revisions.'),
             code("run(['-m', 'scripts.prepare_data'])"),
             markdown('## Baseline sanity samples\nInspect the responses and generation length before the full baseline run.'),
             code("run(['-m', 'scripts.run_generation', '--phase', 'baseline', '--sanity'])"),
             code("run(['-m', 'scripts.run_generation', '--phase', 'baseline'])")]
    for name in ['sft', 'dpo_hh', 'dpo_uf']:
        cells.append(markdown('## ' + name + '\nRun a separate two-update pilot, preserve its metrics, then remove its temporary weights. The full run keeps best and final checkpoints. Both DPO runs start from SFT best.'))
        pilot = "free_space()\nrun(['-m', 'scripts.train', '--config', 'configs/" + name + ".json', '--pilot'])\nrun(['-m', 'scripts.clean_pilots'])"
        cells.append(code(pilot))
        full = "free_space()\nrun(['-m', 'scripts.train', '--config', 'configs/" + name + ".json'])"
        cells.append(code(full))
    cells.append(markdown('## Persona LoRA\nThis separate experiment uses TRL and SmolLM2 on one GPU. It does not start from Qwen SFT.'))
    cells.append(code("run(['-m', 'scripts.train_persona', '--pilot'], persona=True)\nrun(['-m', 'scripts.clean_pilots'])"))
    cells.append(code("run(['-m', 'scripts.train_persona'], persona=True)"))
    cells.append(code("run(['-m', 'scripts.make_plots'])\nprint('Training outputs:', project / 'outputs')\nprint('Now save this notebook version with its outputs.')"))
    cells.append(markdown('## Next\nSave the training notebook version and outputs. Use `02_generation.ipynb` for trained generations. Attach this training output to that notebook. The final report needs the real results and a GitHub repository URL; neither is supplied by an untrained source bundle.'))
    save_notebook('01_training.ipynb', cells)
    generation_setup = '''import json
import os
import subprocess
import sys
from pathlib import Path

attached_root = None
# Set this to your attached folder if there is more than one saved run.
if attached_root is None:
    matches = []
    for folder, directories, files in os.walk('/kaggle/input', followlinks=True):
        path = Path(folder)
        if path.name == 'best' and path.parent.name == 'sft' and path.parent.parent.name == 'outputs':
            if 'config.json' in files:
                matches.append(path / 'config.json')
    if len(matches) != 1:
        raise ValueError('Attach exactly one training output, or set attached_root explicitly.')
    attached_root = matches[0].parents[3]
attached_root = Path(attached_root)
checkpoints = attached_root / 'outputs'
prepared = attached_root / 'data/prepared'
if not (checkpoints / 'dpo_uf/best/config.json').is_file():
    raise FileNotFoundError('The attached run is missing the UltraFeedback best checkpoint.')
if not (prepared / 'alpaca_eval_300_seed42.json').is_file():
    raise FileNotFoundError('The saved AlpacaEval subset is missing.')
print('Using saved training run:', attached_root)

# Copy the small metrics/configuration files for plots; keep full weights in the attached input.
for name in ['sft', 'dpo_hh', 'dpo_uf', 'persona']:
    for filename in ['metrics.jsonl', 'config.json', 'best_metrics.json', 'COMPLETE.json']:
        source = checkpoints / name / filename
        if source.is_file():
            destination = project / 'outputs' / name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

def run(arguments):
    environment = os.environ.copy()
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    subprocess.run([sys.executable] + arguments, check=True, env=environment)

arguments = ['-m', 'scripts.run_generation', '--phase', 'trained',
             '--checkpoint_root', str(checkpoints), '--prepared_root', str(prepared)]
'''
    cells = [markdown('# PA2: generation\nEnable a GPU and Internet. Attach the saved training notebook output using Add Input. This notebook reads the saved checkpoints and generates responses without retraining.'),
             code(bootstrap), code(install), code(generation_setup),
             markdown('## Sanity check\nInspect all printed samples. Check for empty output, wrong formatting or repeated loops. Use the same decoding settings for both models in each comparison.'),
             code("run(arguments + ['--sanity'])"),
             markdown('## Full generation\nSave the notebook version and outputs when this finishes. Comparisons are saved before any model judging.'),
             code("run(arguments)\nrun(['-m', 'scripts.make_plots'])\nprint('Generation outputs:', project / 'outputs/generations')")]
    save_notebook('02_generation.ipynb', cells)
    evaluation_setup = '''import os
import json
import time
import sys
from pathlib import Path
import kaggle_benchmarks as kbench
from alignment.data import read_rows
from scripts.judge_pairwise_kaggle import kaggle_judge, run_judging

pairs_root = None
# Set this explicitly if you attached several generation outputs.
if pairs_root is None:
    matches = []
    for folder, directories, files in os.walk('/kaggle/input', followlinks=True):
        path = Path(folder)
        if path.name == 'pairs' and path.parent.name == 'generations' and 'hh.jsonl' in files:
            matches.append(path / 'hh.jsonl')
    if len(matches) != 1:
        raise ValueError('Attach one generation output, or set pairs_root explicitly.')
    pairs_root = matches[0].parent
pairs_root = Path(pairs_root)
judge_model = 'google/gemini-3.6-flash'
print('Saved comparisons:', pairs_root)
print('Requested judge:', judge_model)

# Record a different model name here if Kaggle does not provide the requested judge.
# Use a new output directory if you change judges after a partially completed run.
judgment_directory = project / 'outputs/evaluation'
judge = kaggle_judge(judge_model)

def judge_track(track, mode):
    pairs = read_rows(pairs_root / (track + '.jsonl'))
    result = run_judging(pairs, judgment_directory / (track + '_judged.jsonl'),
                         judge_model, judge, batch_size=5, sleep_seconds=5,
                         max_retries=3, mode=mode, reuse=True)
    time.sleep(5)
    return result
'''
    cells = [markdown('# PA2: Kaggle model judging\nCreate a Kaggle benchmark notebook with model access, then import these cells. Attach the saved generation notebook output. A regular GPU notebook without benchmark/model access may not have the proxy credentials. This stage only reads saved generations.\n\nKaggle setup reference: https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md'),
             code(bootstrap), code(evaluation_setup),
             code("judge_track('hh', 'safety')"), code("judge_track('uf', 'general')"),
             code("judge_track('persona', 'persona')"),
             code("for path in sorted(judgment_directory.glob('*.summary.json')):\n    print(path.name)\n    print(json.dumps(json.loads(path.read_text()), indent=2))\nimport shutil\nimport base64\nfrom IPython.display import HTML, display\narchive = shutil.make_archive('/kaggle/working/pa2_judgments_backup', 'zip', root_dir=judgment_directory)\nencoded = base64.b64encode(Path(archive).read_bytes()).decode('ascii')\ndisplay(HTML('<a download=\"pa2_judgments_backup.zip\" href=\"data:application/zip;base64,' + encoded + '\">DOWNLOAD JUDGMENTS BACKUP</a>'))"),
             markdown('## Report\nUse the saved summaries, training plots, actual configurations and representative generated responses. Keep the PDF at most eight pages including references. The report checklist is in `REPORT_CHECKLIST.md`.')]
    save_notebook('03_evaluation.ipynb', cells)
    print('Created all three notebooks.')


if __name__ == '__main__':
    main()
