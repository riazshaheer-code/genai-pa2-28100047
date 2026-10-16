# Advanced GenAI PA2

Repository: https://github.com/riazshaheer-code/genai-pa2-28100047

This project implements SFT and two DPO runs for Qwen2.5-0.5B using PyTorch. The separate persona task uses SmolLM2-360M-Instruct, TRL and LoRA. Both Qwen DPO runs start independently from the same SFT best checkpoint.

The report is [REPORT.pdf](REPORT.pdf). Its editable sources are [REPORT.md](REPORT.md) and [REPORT.tex](REPORT.tex). All reported counts come from the saved outputs in this repository.

## Results

| Candidate and baseline | Prompts | Wins | Losses | Ties | Win rate excluding ties | Ties counted as half |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UF DPO vs SFT, AlpacaEval | 300 | 105 | 133 | 62 | 44.12% | 45.33% |
| HH DPO vs SFT, HH set | 300 | 141 | 106 | 53 | 57.09% | 55.83% |
| Persona LoRA vs original SmolLM2 | 16 | 0 | 0 | 16 | Undefined | 50.00% |

The judge was `google/gemini-3.6-flash`. There were no unknown overall decisions or unresolved failures. The persona style axis had 11 ties and five unknown judgments. Preference validation accuracy and generated-answer quality are discussed separately in the report.

## Files

| Path | Contents |
| --- | --- |
| `submission_adapter.py` | The six grader entry points |
| `alignment/core.py` | Response masking, scoring and loss functions |
| `alignment/training.py` | PyTorch SFT and DPO loops |
| `scripts/train_persona.py` | TRL and PEFT persona training |
| `scripts/prepare_data.py` | Fixed subsets, filters and revision checks |
| `scripts/run_generation.py` | Generation and blinded comparison files |
| `scripts/judge_pairwise_kaggle.py` | Kaggle judge, retries and saved summaries |
| `configs/` | Experiment settings; decoding repetition penalty is 1.1 |
| `data/prepared/manifest.json` | Dataset revisions and selected subset hashes |
| `eval_sets/` | Fixed evaluation prompts |
| `outputs/` | Recorded training metrics, generations and judgments |
| `assets/` | Figures used in the report |
| `tests/` | Supplied adapter tests and additional implementation checks |
| `verify_report_results.py` | Recalculate and check the report results without a GPU |
| `notebooks/` | Separate training, generation and judging workflows |

## Check the recorded results

Run from the repository root:

```bash
python verify_report_results.py
```

This checks comparison IDs and hashes, model mappings, original responses, all judgment counts, win rates and selected training checkpoints. It writes `verified_results.json`. No model download or GPU is needed.

The original training curves are in `outputs/report_assets/`. The report figures use raw logged points without smoothing.

## Tests

Use a PyTorch installation suitable for the machine. The requirements files deliberately do not install or replace PyTorch. On Kaggle, retain its existing CUDA-compatible installation.

```bash
python -m pip install -r requirements-persona.txt
python -m pip install -e . --no-deps
python -m pytest -q tests
```

The public interface tests are in `tests/test_public_adapter_smoke.py`. The additional tests include masking and truncation, EOS handling, frozen-reference gradients, accumulated gradients and small local training/generation runs. Local tests use tiny randomly initialized models and do not train the assignment checkpoints.

## Reproduce the experiments

The published `outputs/` directory contains evidence from completed runs, including completion markers. Large model tensors and downloaded training subsets are stored separately. Reproduction should use a fresh working project rather than training into those evidence directories.

The three notebooks unpack their included implementation and configuration into `/kaggle/working/pa2`, keeping the repository's recorded outputs separate. They copy the saved manifest, and data preparation downloads its pinned dataset revisions and checks the reconstructed subset hashes.

1. Import `notebooks/01_training.ipynb` in Kaggle, enable Internet and select GPU T4 x2. Run its cells in order. Inspect the three baseline sanity samples before full generation. Save a completed notebook version with its outputs.
2. Import `notebooks/02_generation.ipynb`, attach the saved training output, and enable a GPU. Inspect the sanity samples, then generate the full comparisons. Save that notebook version and outputs.
3. Use `notebooks/03_evaluation.ipynb` in a Kaggle benchmark environment with model access. Attach the saved generation output. This notebook judges existing pairs in batches of five, waits at least five seconds between attempts, retries failures, and saves a downloadable results backup.

The saved training output used for this report is [notebook14338171ad](https://www.kaggle.com/code/riazshaheer/notebook14338171ad). The saved generation output is [notebook3553a173df](https://www.kaggle.com/code/riazshaheer/notebook3553a173df). Access to these saved outputs is controlled by their Kaggle visibility settings. The small evidence files needed to check the report are included here.

The Qwen template is:

```text
### Instruction:
{instruction}

### Response:
```

Qwen prompts are capped at 256 tokens and training sequences at 512. Response EOS is appended before truncation. SFT averages over predictable response tokens; DPO sums their log probabilities. Persona uses the native SmolLM2 chat template. The LoRA rank is 8, alpha 16, dropout 0, with query/key/value/output attention projections as targets.

Completed training is skipped when using its original saved output. Interrupted training requires a new output directory because optimizer state is not saved. Generation and judging support resuming saved outputs when their metadata match.

## Build the report

From the repository root, with a standard TeX Live or Overleaf setup:

```bash
pdflatex REPORT.tex
pdflatex REPORT.tex
```

The PDF must stay within eight pages including figures and references.

## Sources

The course starter supplied the public adapter contract, tests and preference/evaluation data: https://github.com/ahmad903501/small-model-alignment-starter

The report cites the DPO paper, model cards, TRL and PEFT documentation. Saved run configurations record the actual library versions. Passing the public tests does not establish private-test results or model quality.
