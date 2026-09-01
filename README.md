# AlexNet-again

A PyTorch re-implementation of **AlexNet** (Krizhevsky, Sutskever & Hinton, 2012),
trained on ILSVRC-2012 with a modern training recipe, and distributed across two
GPUs two different ways — the paper's layer-wise model parallelism and standard
data parallelism — so the two can be measured against each other.

**Result:** 37.60% top-1 error (34.78% with 10-crop) after 90 epochs in
4h46m on 2 GPUs, against the paper's 40.7%. See more in [Results](#results).

## What this project does

1. **Re-implements AlexNet** at the paper's layer geometry (96/256/384/384/256
   convolutions, 4096/4096/1000 fully-connected, 62.4M parameters, a 256×6×6
   conv5 feature map) and trains it on the full 1.28M-image ILSVRC-2012 set.
2. **Applies a modern training recipe**: He initialization, cosine learning-rate
   decay with linear warmup, bfloat16 mixed precision, label smoothing,
   RandomResizedCrop + horizontal flip augmentation, and no weight decay on
   BatchNorm parameters or biases — reaching 3.1 points below the original's
   top-1 error.
3. **Distributes across 2 GPUs** two ways — the paper's two-column layer-wise
   split, and DistributedDataParallel — and benchmarks both against a
   single-GPU baseline.
4. **Trains for extended schedules** (90 epochs by default) with per-iteration
   LR scheduling and resumable checkpointing.

## Results

### Accuracy

Best checkpoint (epoch 88) evaluated on the held-out 50,000-image split.
The full 90-epoch run took **4h46m** on 2 GPUs.

| Model | Top-1 err | Top-5 err |
|---|---|---|
| AlexNet paper, single net (ILSVRC-2012 val) | 40.7% | 18.2% |
| **This implementation, centre crop** | **37.60%** | **15.98%** |
| **This implementation, 10-crop** | **34.78%** | **13.86%** |

Centre crop is **3.1 points** below the paper's top-1 error and 2.2 points below
its top-5, clearing the 2-point target. With the paper's 10-crop test-time
augmentation the margin widens to 5.9 and 4.3 points.

Read these with the caveat in [Dataset](#dataset): validation is a held-out
slice of train, not the official ILSVRC-2012 validation set, so the numbers run
slightly optimistic against published results. The comparison that is fully
sound is against the other configurations in this repo, which are measured the
same way.

Validation accuracy over the run:

| epoch | 0 | 4 | 9 | 19 | 39 | 59 | 79 | 89 |
|---|---|---|---|---|---|---|---|---|
| val top-1 | 2.86% | 24.68% | 37.60% | 44.29% | 49.57% | 55.01% | 61.28% | 62.29% |
| val top-5 | 9.09% | 47.74% | 62.86% | 69.90% | 74.36% | 78.87% | 83.35% | 84.01% |

The jump between epochs 59 and 79 is the cosine schedule annealing the learning
rate toward zero.

### Throughput (measured)

Synthetic data, batch **256 per GPU**, bfloat16, channels-last, 40 timed steps
after 10 warmup steps. Per-GPU batch is held constant, so the 2-GPU run does
twice the work per step and images/sec is the fair comparison.

| Mode | GPUs | Global batch | img/s | vs. single | Peak mem |
|---|---|---|---|---|---|
| Single GPU | 1 | 256 | 10,583 | 1.00× | 2.0 GiB |
| **DDP + bf16 gradient compression** | 2 | 512 | **14,734** | **1.39×** | 2.9 GiB |
| DDP, no gradient compression | 2 | 512 | 10,832 | 1.02× | 2.1 GiB |
| Model parallel (two columns) | 2 | 256 | 1,780 | 0.17× | 1.5 GiB |
| Model parallel (two columns) | 2 | 512 | 1,855 | 0.18× | 2.3 GiB |

**Layer-wise model parallelism is ~6× slower than a single GPU**, not faster.
This is the expected result and worth stating plainly: the paper's two-column
split existed because a GTX 580 had only 3 GB of memory, not because it was
fast. AlexNet fits in 2 GB, so on modern hardware the split buys nothing and
costs a great deal — it serializes the pipeline and adds cross-GPU transfers at
conv3 and the fully-connected layers. **Note** that on the machine used for these experiments, those transfers must be
**staged through host memory** (see the peer-transfer section), which is slowing down cross-GPU transfer speeds.

Real two-GPU speedup comes from data parallelism, and even there the gain is
modest: **1.39×**, not the ~2× that a compute-bound model would give. AlexNet is
62M parameters but only ~0.7 GFLOPs, so each step all-reduces ~250 MB of
gradients against very little compute, and the interconnect dominates.

**bfloat16 gradient compression is what makes 2 GPUs worthwhile at all.**
Without it, DDP runs at 10,832 img/s — statistically indistinguishable from a
single GPU (10,583), so the second GPU contributes nothing. Halving the
all-reduce payload turns that into a 1.39× speedup. It is enabled by default
(`train.bf16_grad_compress`).


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

## Storage note: stage the dataset into RAM before training

The `/data` volume is a shared spinning-disk array. Random reads of the 1.28M
small image files measure **~35 ms each (~28 img/s single-threaded, ~1 MB/s)**,
which caps training at roughly **510 img/s** regardless of worker count. For
comparison, JPEG decode costs 1.5 ms/image and augmentation 2.9 ms/image, so
32 workers would sustain ~11,000 img/s if the bytes were already in memory, and
the GPUs can consume ~14,700 img/s. **Storage latency, not compute, is the
binding constraint** — by a factor of roughly 20.

Thus we copy the 38 GB resized set into `/dev/shm`:

```bash
./scripts/stage_to_ram.sh
./scripts/launch.sh --config configs/default.yaml \
    --set data.root=/dev/shm/ilsvrc2012_256
```

This takes epoch time from **~40 minutes to ~3 minutes** (~6,300 img/s
end-to-end) and is the difference between a 60-hour and a 5-hour run.

## Hardware note: peer-to-peer GPU transfer is broken on the machine used to run these experiments

Direct `cuda:0 ↔ cuda:1` copies **silently return all-zero tensors** here, in
both directions and at every transfer size, and NCCL **hangs indefinitely** if
it tries to use the peer path. `cudaDeviceCanAccessPeer` reports `True`, so
nothing raises an error — the data is simply wrong.

This causes significant delay in cross-gpu data transfers, and may be the cause behind some of the observed results.

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

## Setup

```bash
pip install -r requirements.txt
```

Prepare the data (one-off; the source directory is never modified):

```bash
# ~38 GB output, resumable, safe to re-run
python scripts/preprocess_imagenet.py \
    --src /data/users/cs24s008/Datasets/ilsvrc2012 \
    --dst /data/users/cs24s008/Datasets/ilsvrc2012/256px

python scripts/make_splits.py \
    --root /data/users/cs24s008/Datasets/ilsvrc2012/256px --out splits

# Stage into RAM -- worth ~12x on this machine, see the storage note above.
# Volatile: lost on reboot, freed with `rm -rf /dev/shm/ilsvrc2012_256`.
./scripts/stage_to_ram.sh
```

## Reproducing

```bash
# Two GPUs, data parallel (the recommended path; run stage_to_ram.sh first)
./scripts/launch.sh --config configs/default.yaml \
    --set data.root=/dev/shm/ilsvrc2012_256

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
scripts/       preprocess, make_splits, stage_to_ram, train, evaluate,
               benchmark, launch.sh
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
