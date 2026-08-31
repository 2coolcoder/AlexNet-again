# AlexNet-again

A PyTorch re-implementation of **AlexNet** (Krizhevsky, Sutskever & Hinton, 2012),
trained on ILSVRC-2012 with a modern training recipe, and distributed across two
GPUs two different ways — the paper's layer-wise model parallelism and standard
data parallelism — so the two can be measured against each other.

## What this project does

1. **Re-implements AlexNet** at the paper's layer geometry (96/256/384/384/256
   convolutions, 4096/4096/1000 fully-connected, 62.4M parameters, a 256×6×6
   conv5 feature map) and trains it on the full 1.28M-image ILSVRC-2012 set.
2. **Applies a modern training recipe**: He initialization, cosine learning-rate
   decay with linear warmup, bfloat16 mixed precision, label smoothing,
   RandomResizedCrop + horizontal flip augmentation, and no weight decay on
   BatchNorm parameters or biases.
3. **Distributes across 2 GPUs** two ways — the paper's two-column layer-wise
   split, and DistributedDataParallel — and benchmarks both honestly against a
   single-GPU baseline.
4. **Trains for extended schedules** (90 epochs by default) with per-iteration
   LR scheduling and resumable checkpointing.

## Results

> Fill in after running; see [Reproducing](#reproducing). Throughput numbers
> below are measured on this machine, accuracy numbers are pending the full run.

### Accuracy (held-out 50k split, single centre crop)

| Model | Top-1 err | Top-5 err |
|---|---|---|
| AlexNet paper, single net (ILSVRC-2012 val) | 40.7% | 18.2% |
| This implementation | _pending_ | _pending_ |

### Throughput

_Pending final measurement._

## Architecture notes

The network follows the paper's layer geometry, with one deliberate change and
one clarification:

- **BatchNorm replaces Local Response Normalization.** LRN is slow and
  contributes little; BatchNorm trains faster and more stably, and is what makes
  the accuracy target reachable within a 90-epoch budget. This is a real
  deviation from the 2012 architecture and is called out rather than hidden.
  Convolutions followed by BatchNorm carry no bias, since the BatchNorm shift
  supplies it.
- **conv1 uses `padding=2`** with an 11×11 stride-4 kernel on a 224×224 input,
  giving a 55×55 first feature map. The paper states a 224×224 input but its
  arithmetic implies 227×227; padding reconciles the two.

`AlexNetTwoColumn` reproduces the original two-GPU split: two half-width columns
(48/128/192/192/128) on separate devices, where conv3 and the fully-connected
layers see both columns while conv4 and conv5 stay within a column. Each
column's stem runs on its own CUDA stream so the two halves genuinely overlap.
A test asserts it computes the same function as an equivalent single-device
model, which is what catches silent cross-GPU corruption.

## Dataset

Only the ILSVRC-2012 **train** split (1,281,167 images, 1000 classes) is
available on this machine, so **validation is a held-out slice of train**: a
deterministic 50 images per class (50,000 total), matching the size of the
official validation set.

Because that held-out set is drawn from the same collection as the training
data, accuracy on it reads **slightly optimistic** compared with numbers
measured on the official ILSVRC-2012 validation set. It is a sound basis for
model selection and for comparing configurations against each other, but it is
not strictly comparable to published val-set results. To get a directly
comparable number, drop the official `val/` directory in wnid layout alongside
the training data and point `--split` at an index file built from it; no code
changes are needed.

Images are pre-resized once to a 256px shortest side. AlexNet is only ~0.7
GFLOPs, so training is bound by JPEG decode rather than compute, and decoding
full-resolution images every epoch would waste most of the CPU budget.

## Hardware note: peer-to-peer GPU transfer is broken on this machine

Direct `cuda:0 ↔ cuda:1` copies **silently return all-zero tensors** here, in
both directions and at every transfer size, and NCCL **hangs indefinitely** if
it tries to use the peer path. `cudaDeviceCanAccessPeer` reports `True`, so
nothing raises an error — the data is simply wrong.

The cause is the IOMMU being active (`/proc/cmdline` has no `iommu=pt` or
`iommu=off`) with the two GPUs under separate PCIe root complexes
(`0000:41:00.0` and `0000:c1:00.0`).

The code works around this automatically:

- `src/alexnet/p2p.py` probes each device pair at runtime with a **random**
  pattern and falls back to host-staged transfers when the peer path is broken.
  The probe pattern must be unpredictable: a deterministic one (`arange`) passes
  spuriously, because the destination can reuse a cached allocator block that
  already holds an identical pattern, so a copy that moved nothing still
  compares equal.
- `NCCL_P2P_DISABLE=1` is set automatically before the NCCL communicator is
  built, and exported by `scripts/launch.sh`. With it, all-reduce and broadcast
  are numerically correct.

**The permanent fix** is to add `iommu=pt` to the kernel command line and
reboot (needs root). That would remove the host-staging detour and should
improve both the DDP all-reduce and model-parallel throughput reported below.

## Setup

```bash
pip install -r requirements.txt
```

Prepare the data (one-off; the source directory is never modified):

```bash
# ~70 GB output, resumable, safe to re-run
python scripts/preprocess_imagenet.py \
    --src /data/users/cs24s008/Datasets/ilsvrc2012 \
    --dst /data/users/cs24s008/Datasets/ilsvrc2012_256

python scripts/make_splits.py \
    --root /data/users/cs24s008/Datasets/ilsvrc2012_256 --out splits
```

## Reproducing

```bash
# Two GPUs, data parallel (the recommended path)
./scripts/launch.sh --config configs/default.yaml

# Single GPU
python scripts/train.py --config configs/default.yaml \
    --set train.parallel_mode=single

# Two GPUs, the paper's layer-wise model parallelism
python scripts/train.py --config configs/default.yaml \
    --set train.parallel_mode=model_parallel

# Fast correctness check: 100 classes, 3 epochs
./scripts/launch.sh --config configs/smoke.yaml

# Evaluation and throughput
python scripts/evaluate.py --checkpoint runs/default/best.pth [--ten-crop]
python scripts/benchmark.py --modes single model_parallel
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 scripts/benchmark.py --modes ddp
```

Any config value can be overridden from the command line:

```bash
python scripts/train.py --config configs/default.yaml \
    --set train.epochs=150 data.batch_size=128 optim.lr=0.05
```

## Training recipe

| Setting | Value |
|---|---|
| Epochs | 90 (5 linear warmup) |
| Batch size | **256 per GPU** → 512 global on 2 GPUs |
| Optimizer | SGD, momentum 0.9, Nesterov |
| Learning rate | `0.01 × global_batch / 256` (= 0.02 on 2 GPUs), cosine decay to 0 |
| Weight decay | 5e-4, excluded on BatchNorm parameters and biases |
| Label smoothing | 0.1 |
| Precision | bfloat16 autocast, channels-last |
| Augmentation | RandomResizedCrop(224), horizontal flip, optional PCA lighting |

`data.batch_size` is **per GPU**, so each GPU does identical work whatever the
world size, and the learning rate follows the global batch automatically. This
keeps single-GPU and 2-GPU runs consistently tuned, and makes the throughput
comparison a clean weak-scaling measurement.

bfloat16 is used rather than float16 because it has the same exponent range as
fp32, so no gradient scaler is needed.

## Layout

```
configs/       default (90-epoch) and smoke (100-class) recipes
src/alexnet/   model, data, engine, optim, distributed, p2p, checkpoint, metrics
scripts/       preprocess, make_splits, train, evaluate, benchmark, launch.sh
tests/         pytest suite (model, data/splits, optim schedule, p2p)
splits/        generated index files (gitignored)
runs/          checkpoints, TensorBoard logs, metrics.csv (gitignored)
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers layer geometry against the paper, He-init scaling, the
BatchNorm-for-LRN substitution, split determinism/disjointness, LR warmup and
decay shape, the no-weight-decay grouping, and — importantly — that the
two-column model matches a single-device reference and that cross-GPU transfers
carry correct data and gradients.

## Reference

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification
with Deep Convolutional Neural Networks.* NeurIPS 25.
