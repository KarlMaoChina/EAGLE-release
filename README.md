# EAGLE

Official PyTorch implementation of **EAGLE** (Enhanced AI for Gallbladder Lesion Evaluation): a frozen multimodal pipeline that scores gallbladder malignancy risk from a non-contrast CT volume and twelve routine clinical variables.

This repository contains the public methods code only. It does not include patient images, case identifiers, site-specific tables, or trained weights.

## Method

EAGLE v1.0 is five modules with a locked operating point:

1. **Segmentation cleanup** — anatomy-aware post-processing of a gallbladder mask (spatial Gaussian prior, relative volume filter, 20 mm proximity, 2 mm closing).
2. **Dual-view preparation** — standard spacing `0.72 x 0.72 x 1.0 mm` and enlarged spacing `0.9 x 0.9 x 2.0 mm`, both cropped to `96 x 112 x 80` patches, windowed at level 40 HU / width 300 HU, then z-scored, with region-adaptive masks.
3. **Radiomics** — IBSI features via PyRadiomics 3.1.0 from a 5 mm dilated ROI (`1,470` candidates; `32` frozen names in `eagle/assets/radiomics_features_v1.txt`).
4. **Clinical inputs** — twelve variables (cholelithiasis, age, A/G ratio, ALP, prealbumin, DBIL, TBIL, CEA, AFP, CA19-9, CA15-3, CA125). Continuous missing values use the development-split median; categorical missing values use the mode. Imputation is fit without the outcome.
5. **BiAMF** — frozen tabular encoder plus two DVSE (ResNet-18-3D + spatial prior modulation) streams, projected to 256-d, 4-head cross-modal attention, dual dynamic weighting, residual MLP. Five-fold ensemble mean; binary call at **T = 0.5**.

Random seed for fold use, initialization, augmentation, and bootstrap resampling is `42`.

## Layout

```text
eagle/                 Python package
configs/eagle_v1.yaml  public hyperparameters
examples/cases.csv     header template (synthetic rows only)
tests/                 unit tests, including a source leakage scan
```

Expected per-case image folder after `preprocess`:

```text
<cases>/case_0001/standard_image.nii.gz
<cases>/case_0001/standard_mask.nii.gz
<cases>/case_0001/enlarged_image.nii.gz
<cases>/case_0001/enlarged_mask.nii.gz
```

Case tables use a generic `case_id` column. Do not put identifiable source codes in that field if you share tables.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install -e ".[radiomics]"   # only if you need PyRadiomics extraction
```

## Commands

```bash
python -m eagle postprocess-mask --mask mask.nii.gz --output mask_clean.nii.gz
python -m eagle preprocess --image ct.nii.gz --mask mask.nii.gz --output-dir cases --case-id case_0001
python -m eagle fit-stats --table cases/table.csv --output-dir checkpoints
python -m eagle train-dvse --table cases/table.csv --image-root cases --output-dir outputs --stream standard
python -m eagle train-dvse --table cases/table.csv --image-root cases --output-dir outputs --stream enlarged
python -m eagle train-tabular --table cases/table.csv --image-root cases --output-dir outputs
python -m eagle train-fusion --table cases/table.csv --image-root cases --output-dir outputs --extractor-root outputs
python -m eagle infer --image ct.nii.gz --mask mask.nii.gz --clinical-table cases/table.csv --case-id case_0001 --weights checkpoints --output outputs/pred.json
```

`infer` expects `checkpoints/preprocess_stats.json` and `checkpoints/biamf/fold_{0-4}.pt`. Weights are not shipped here; place your own freeze package in that layout.

## Tests

```bash
pip install pytest
pytest
```

## Citation

Please cite the accompanying paper when using this code.
