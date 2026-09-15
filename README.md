# InterFormer

This repository provides the implementation of **InterFormer** for selective logging mapping using Sentinel-1 image time series, associated with the study:

> Xinyao Huang, Raian Vargas Maretto, Leila Maria Garcia Fonseca, Alfred Stein, Claudio Persello, Mapping selective logging with Sentinel-1 image time series and a novel transformer-based model, International Journal of Applied Earth Observation and Geoinformation, Volume 153, 2026, 105567, https://doi.org/10.1016/j.jag.2026.105567.


## Installation

Run the following commands from the parent directory containing `interformer/`.

```bash
pip install -r interformer/requirements.txt
```

## Training

```bash
python interformer/test.py \
  --grd-root /path/to/GRD_s2_v1_flat \
  --glcm-root /path/to/GRD_s2_v1_flat_glcm \
  --output runs/experiment1
```

Organize GRD, GLCM, and sample-coordinate NPY files in matching `polygon_*` folders. The loader constructs `VH, VV, VH-VV, GLCM` features and partitions the data by polygon into training, validation, and test sets. Use a separate output directory for each experiment.

Optional arguments: `--epochs 500 --batch-size 64 --lr 0.001 --seed 42 --accelerator cpu`.

## Evaluation

```bash
python interformer/test.py \
  --grd-root /path/to/GRD_s2_v1_flat \
  --glcm-root /path/to/GRD_s2_v1_flat_glcm \
  --checkpoint path/to/best.ckpt \
  --split-file runs/experiment1/split.json \
  --output runs/evaluation1
```

Outputs include checkpoints, CSV logs, `split.json`, `results.json`, and `predictions.npz`.
