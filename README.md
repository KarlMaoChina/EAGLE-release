# EAGLE

**EAGLE** (Enhanced AI for Gallbladder Lesion Evaluation) scores gallbladder malignancy from a preoperative non-contrast CT volume, a gallbladder mask, and twelve routine clinical variables. It writes a probability in \([0, 1]\) and a binary call at the operating point **T = 0.5**.

Installable as `eagle-gbc`. Geometry, feature names, and training hyperparameters live in [`eagle/spec.py`](eagle/spec.py); [`configs/eagle_v1.yaml`](configs/eagle_v1.yaml) repeats them in YAML. Seed `42` is used for folds, initialization, augmentation, and bootstrap resampling.

## Pipeline

```text
NCCT volume + gallbladder mask
        |                       12 clinical variables
        v                               |
 anatomy-aware post-process             |
        |                               v
 dual-view patches 96 x 112 x 80     median / mode imputation
   standard  0.72 x 0.72 x 1.0 mm    z-score (continuous)
   enlarged  0.9  x 0.9  x 2.0 mm         |
        |                                 |
   DVSE + SPM (204-d each)           32 IBSI features (5 mm ROI)
        |                                 |
        +-------- project to 256-d -------+
                        |
              4-head cross-modal attention
              dual dynamic weighting
              residual MLP
                        |
              5-fold mean probability
              binary flag at T = 0.5
```

### Mask cleanup

Connected components are scored with a three-axis Gaussian location prior \((\mu, \sigma, w) = ((0.34, 0.38, 0.60),\ (0.03, 0.05, 0.15),\ (0.40, 0.30, 0.30))\). Regions below \(0.05\) of the largest volume, farther than \(20\,\mathrm{mm}\) from the primary component, or below a prior score of \(0.3\) are removed. The retained mask is closed with a \(2\,\mathrm{mm}\) ellipsoid, hole-filled, and smoothed (\(\sigma = 0.5\)).

The CLI takes a mask as input. A typical upstream source is nnU-Net `3d_cascade_fullres`; this repository implements the anatomy-aware stage that follows it.

### Dual-view CT

Volumes are windowed at level \(40\,\mathrm{HU}\) / width \(300\,\mathrm{HU}\), resampled to \(0.72 \times 0.72 \times 1.0\,\mathrm{mm}^3\) and to \(0.9 \times 0.9 \times 2.0\,\mathrm{mm}^3\), cropped to a centroid-centred \(96 \times 112 \times 80\) patch, and z-scored. Spacing tuples follow the NIfTI data-array axes returned by nibabel.

Region-adaptive mask channels:

| Stream | Interior | Wall (\(\le 1.5\,\mathrm{mm}\)) | Exterior |
|---|---|---|---|
| Standard | \(0.7\) | \(0.9\) | \(1.0\) out to \(3\,\mathrm{mm}\) |
| Enlarged | \(0.6\) | \(1.0\) | \(0.8 / 0.4 / 0.2\) at \(5 / 10 / 15\,\mathrm{mm}\) |

`preprocess` writes:

```text
<cases>/<case_id>/standard_image.nii.gz
<cases>/<case_id>/standard_mask.nii.gz
<cases>/<case_id>/enlarged_image.nii.gz
<cases>/<case_id>/enlarged_mask.nii.gz
```

### Radiomics

PyRadiomics 3.1.0 extracts IBSI features from a \(5\,\mathrm{mm}\)-dilated gallbladder ROI. Images are resampled to \(1\,\mathrm{mm}\) isotropic voxels, z-scored, scaled by \(100\), and discretized with bin width \(5\). First-order and texture descriptors are computed on the original image and on LoG (\(\sigma = 1, 2, 3\,\mathrm{mm}\)), eight 3-D wavelet sub-bands, square, square-root, logarithm, and exponential filters (\(1{,}470\) candidates). The development set retained **32** names in [`eagle/assets/radiomics_features_v1.txt`](eagle/assets/radiomics_features_v1.txt).

### Clinical variables

`cholelithiasis`, `age`, `ag_ratio`, `alp`, `prealbumin`, `dbil`, `tbil`, `cea`, `afp`, `ca19_9`, `ca15_3`, `ca125`

Continuous fields use the development-split median; `cholelithiasis` uses the mode. Imputation is fit without the outcome. `exclude_high_missingness` drops rows with more than \(30\%\) missing clinical fields when you want that filter. English aliases (`CA19-9`, `A/G ratio`, `PA`, …) are accepted in `eagle/clinical.py`. Tables identify rows with `case_id`.

### DVSE and BiAMF

**DVSE** is a 3-D ResNet-18 with channel scale \(0.4\) (stage widths \(25, 51, 102, 204\)). Global average pooling gives a **204-d** vector per stream. Spatial prior modulation concatenates the mask after residual stages 1–3, reduces it with \(1\times1\times1\) convolutions, and applies a sigmoid gate. Each stream is trained with weighted BCE, AdamW (\(5\times10^{-4}\), weight decay \(0.01\)), batch size \(16\), `ReduceLROnPlateau` on validation AUC (factor \(0.5\), patience \(8\), floor \(10^{-6}\)), and early stopping after \(15\) stagnant epochs. Augmentation: coarse dropout (probability \(0.2\), 1–3 holes of size \(3\)) and intensity jitter. The two views are trained as separate jobs.

**Tabular encoder.** Concatenated clinical and radiomic values (\(12+32=44\)) map \(44 \to 128 \to 256\) with dropout \(0.5\). The 256-d hidden state is the fusion token.

**BiAMF.** The image and tabular encoders are held fixed while the fusion head trains. Image tokens are projected to \(256\)-d (Linear–LayerNorm–GELU–Dropout \(0.4\)). Four-head cross-modal attention lets each of the three tokens attend to the other two. Dual dynamic weighting multiplies a learned global vector by a per-case sigmoid gate and layer-normalizes. Concatenation (\(768\)-d) is reduced by \(4\) to \(192\)-d, then two residual MLP blocks (dropout \(0.4\)) and a final dropout \(0.3\) produce a logit. Fusion: AdamW (\(5\times10^{-4}\), weight decay \(0.15\)), batch size \(8\), scheduler patience \(4\), early stopping \(12\), at most \(100\) epochs. Selection metric \(0.7\,\mathrm{AUC} + 0.3\,\mathrm{AP}\).

**Inference.** Five fold checkpoints run independently. The reported score is the mean of the five sigmoid probabilities. The flag is \(p \ge 0.5\).

## Install

Python 3.10+ and PyTorch 2.1+ (CPU or CUDA).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
pip install -e ".[radiomics]"      # PyRadiomics 3.1.0
pip install -e ".[dev]"            # pytest
```

## Commands

```bash
python -m eagle postprocess-mask --mask mask.nii.gz --output mask_clean.nii.gz
python -m eagle preprocess --image ct.nii.gz --mask mask.nii.gz --output-dir cases --case-id case_0001
python -m eagle extract-radiomics --image ct.nii.gz --mask mask.nii.gz --output radiomics.json
python -m eagle fit-stats --table cases/table.csv --output-dir checkpoints
python -m eagle train-dvse --table cases/table.csv --image-root cases --output-dir outputs --stream standard
python -m eagle train-dvse --table cases/table.csv --image-root cases --output-dir outputs --stream enlarged
python -m eagle train-tabular --table cases/table.csv --image-root cases --output-dir outputs
python -m eagle train-fusion --table cases/table.csv --image-root cases --output-dir outputs --extractor-root outputs
python -m eagle infer \
  --image ct.nii.gz --mask mask.nii.gz \
  --clinical-table cases/table.csv --case-id case_0001 \
  --weights checkpoints --output outputs/pred.json
python -m eagle evaluate --table scores.csv --label-col label --score-col probability
```

Weight directory for `infer`:

```text
checkpoints/preprocess_stats.json
checkpoints/biamf/fold_0.pt
...
checkpoints/biamf/fold_4.pt
```

`preprocess_stats.json` holds imputation and z-score statistics. Checkpoints use the `eagle-v1` payload (`model_state_dict` plus `extra`). Keep weights and NIfTI files out of git; `.gitignore` already lists them.

[`examples/cases.csv`](examples/cases.csv) is a two-row English header template.

## Tests

```bash
pytest
python scripts/check_spec.py
python scripts/smoke_synthetic.py
```

`check_spec.py` asserts that `eagle/spec.py` and `configs/eagle_v1.yaml` agree on names, geometry, optimizer settings, and T = 0.5. `smoke_synthetic.py` runs cleanup, dual-view patches, `fit-stats`, and five-fold `infer` on a randomly initialized ensemble, which only shows that the pipe executes.

## Layout

```text
eagle/                      package: spec, preprocess, models, train, infer
configs/eagle_v1.yaml       YAML copy of the hyperparameters
examples/cases.csv          synthetic table header
scripts/check_spec.py
scripts/smoke_synthetic.py
tests/
```

## Acknowledgements

Late folds were trained with MyGO!!!!! and Ave Mujica in the headphones. They improved morale. They did not tune the learning rate.
