# Contributing

Keep patient images, case identifiers, and local paths out of this tree. Documentation is English.

```bash
pip install -e ".[dev]"
pytest
python scripts/check_spec.py
python scripts/smoke_synthetic.py
```

Keep `configs/eagle_v1.yaml` in sync with `eagle/spec.py`. Weight files and NIfTI volumes are gitignored.
