# Contributing

Public code only. Do not add patient images, case identifiers, hospital names, internal usernames, or laboratory absolute paths.

## Checks

```bash
pip install -e ".[dev]"
pytest
python scripts/check_spec.py
python scripts/smoke_synthetic.py
```

`tests/test_leakage.py` fails if Chinese text or internal tokens appear in the tree. Keep documentation in English.

## Constants

`eagle/spec.py` is the locked source. `configs/eagle_v1.yaml` must match it; `scripts/check_spec.py` compares both with Appendix 1.

## Weights and data

Do not commit `*.pt`, `*.pth`, `*.nii.gz`, or case tables. `.gitignore` already blocks those patterns.
