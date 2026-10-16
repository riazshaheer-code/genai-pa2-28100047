# Validation

The final local suite passed all 36 tests. The supplied public adapter file passed all seven of its tests. The supplied public tests and toy model files were not modified.

```text
python -m pytest -q tests
36 passed

python -m pytest -q tests/test_public_adapter_smoke.py
7 passed
```

The local environment used CPU PyTorch 2.6.0, Transformers 4.56.2, Datasets 4.8.5, TRL 0.23.1 and PEFT 0.17.1. The assignment training environment was Kaggle with PyTorch 2.11.0+cu128; its actual recorded settings are retained in outputs/*/config.json.

The suite includes response masking and EOS, token scoring, policy/reference gradients, gradient accumulation with unequal response lengths, small SFT and DPO training runs, generation and reuse, and a small TRL/LoRA run. Regression checks cover the Kaggle task result type and reconstruction from a saved dataset revision manifest. These checks do not establish private grader results.

All three notebook files passed code-cell syntax checks. Their included source was checked for repetition penalty 1.1, the corrected Kaggle task wrapper and the saved data manifest. Full assignment GPU training was not repeated during repository packaging.

The standalone report verifier checked all 616 judgments, comparison hashes, original generated responses, model mappings, win-rate calculations, and selected training metrics. The fixed AlpacaEval file was reconstructed from the saved generation records and matches the original manifest hash.

Large model weights are not included in the GitHub package. Saved Kaggle output links are in the report and README.
