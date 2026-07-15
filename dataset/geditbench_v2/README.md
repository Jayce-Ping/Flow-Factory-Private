# GEditBench-v2 (Flow-Factory I2I layout)

Source: [`GEditBench-v2/GEditBench-v2`](https://huggingface.co/datasets/GEditBench-v2/GEditBench-v2) (~1,200 edit instructions, ~12 GB).

## Build

```bash
# Full build (~12 GB download on first run)
bash dataset/geditbench_v2/download.sh

# Or directly:
python dataset/geditbench_v2/download_and_build.py

# Smoke (streaming; does not download the full ~12 GB):
python dataset/geditbench_v2/download_and_build.py --max_samples 8 --force
# Note: smoke artifacts are partial; run without --max_samples (+ --force) for the
# official ~960 / ~240 split before training.
```

Idempotent by default. Pass `--force` to rebuild.

## Split

Deterministic **4:1** train/test with `random.Random(0)`:

```python
rng = random.Random(0)
idxs = list(range(N)); rng.shuffle(idxs)
n_train = int(round(0.8 * N))  # e.g. 960 / 240 for N=1200
train_idxs, test_idxs = idxs[:n_train], idxs[n_train:]
```

## Layout

```
geditbench_v2/
  images/{key}.png
  train.jsonl
  test.jsonl
  download_and_build.py
  download.sh
  README.md
```

## JSONL schema

```json
{"prompt": "<instruction>", "image": "<key>.png", "task": "<task>", "key": "<key>"}
```

- `instruction` → `prompt` (edit instruction)
- `source_image` → `images/{key}.png` (basename only in JSONL)
- `task` / `key` kept as sample metadata for eval aggregation

Compatible with `GeneralDataset` I2I loading (`image` + `images/`).
