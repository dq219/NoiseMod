# CIFAR-10 Modular FC Pipeline

Test whether a weighted-activity regularizer drives modular connectivity in the
fully-connected head of a ResNet18 when trained on horizontally concatenated
CIFAR-10 images. One input contains `N` frames; each frame has its own label;
the network must classify all `N` simultaneously, and we want the FC head to
break into `N` independent sub-networks.

The theory is in `manuscript_revised.pdf` (this folder).
Sibling implementations live under `../modularisation_via_noise/`,
`../mod_bulk_competition/`, `../modular_generalisation_Gaussian/`,
`../modular_generalisation_Random/`. The Laplacian-spectrum convention used
here mirrors `../modularisation_via_noise/utils/graphs.py`.

## Files

- `data.py` — `ConcatCIFAR10` dataset (see "Data" below).
- `model.py` — `ModularResNet18` (see "Network" below).
- `run.py` — main training script. Trains, evaluates, computes the
  Laplacian spectrum of fc2 each epoch, saves everything into a per-run folder.
- `slurm_submit.sh` — wrapper that submits one slurm job per call:
  `./slurm_submit.sh <N> <reg_weight> [extra run.py args ...]`.
  Env-var overrides: `PARTITION` (default `mit_normal_gpu`), `TIME`
  (`04:00:00`), `MEM` (`8G`), `CPUS` (`2`), `GPUS` (`1`), `ACCOUNT`,
  `EPOCHS` (`100`), `HIDDEN_DIM` (`-1` → `512*N`), `RESULTS_ROOT`
  (`./results`).
- `plot_eigengap.py` — cross-run modularity analysis. Reads every
  `metrics.npz` under `--results-root` and writes a single 2×3 panel
  figure (see "Cross-run analysis" below).
- `manuscript_revised.pdf` — theory writeup.
- `results/` — per-run outputs, grouped by batch size
  (`results/batch_size_128/<run-name>/`, see "Outputs").
- `summary/` — slide deck + figure-generation scripts
  (see "Slide deck (`summary/`)" below).

## Data (`data.py`)

`ConcatCIFAR10(root, N, train, transform, download)` returns:
- `x`: tensor `(3, 32, 32 * N)` — `N` CIFAR-10 frames concatenated along width.
- `y`: tensor `(N,)` of integer labels in `[0, 10)`.

`N` frames are drawn uniformly at random inside `__getitem__`, so the same
index returns a different concatenation each call. `len(dataset)` is
`len(base_cifar10) // N` — an arbitrary "epoch" length, not a real index range.

## Network (`model.py`)

`ModularResNet18(N, num_classes=10, hidden_dim=None)`:

```
input  (B, 3, 32, 32*N)
  └─ stem (3x3 stride-1, no initial maxpool — CIFAR adaptation)
  └─ layer1 .. layer4 (standard ResNet18 blocks)
  └─ AdaptiveAvgPool2d((1, N))             → (B, 512, 1, N)
  └─ flatten                                → (B, 512*N)
  └─ fc1 (512*N → hidden_dim)              ← default hidden_dim = 512*N
  └─ LeakyReLU(0.1)                         ← prevents dead-unit collapse
  └─ fc2 (hidden_dim → 10*N)
  └─ view → (B, N, 10) logits
```

Flatten layout: the (B, 512, 1, N) tensor flattens channel-major,
position-minor — input col `i` of fc1 corresponds to channel `i // N` at
frame-position `i % N`. Output of fc2 is viewed as `(B, N, 10)`, so output
row `j` corresponds to frame `j // 10`, class `j % 10`.

### Weighted-activity regularizer

`model.weighted_act_regularizer(hidden, alpha=2.0)` (called only on the
**hidden activation**, using **fc2** weights):

```
E_batch[ sum_j ( |h|^alpha @ (W2.T)^2 )_j ]
```

where `h` is the post-LeakyReLU output of fc1 and `W2 = fc2.weight`. Penalises
each hidden unit by the squared magnitudes of its outgoing fc2 connections, so
the pressure is to make every hidden unit project to a small subset of fc2
outputs — pushing fc2 toward block-diagonal structure across the `N` frames.

The single-layer version (regularizing fc directly) was abandoned: with only
one FC layer, irrelevant inputs trivially get small outgoing weights and
modularity is meaningless.

### Modularity readout

`run.py:laplacian_spectrum(W)`:
1. `S = |W|.T @ |W| / ||cols||²` — cosine similarity between hidden units
   based on their fc2 outgoing patterns.
2. `L = I − D^(−1/2) S D^(−1/2)` (symmetric normalized Laplacian).
   Same spectrum as the random-walk Laplacian `I − D^(−1) S` used in
   `../modularisation_via_noise/utils/graphs.py`, but symmetric so
   `np.linalg.eigvalsh` returns exact real eigenvalues.
3. For `N` true modules, the `N`-th smallest eigenvalue should be ~0 and the
   `(N+1)`-th should jump — eigengap at position `N`.

The first 10 eigenvalues are tracked per epoch.

## Running

Local:
```
python run.py --N 2 --reg-weight 1.0 --epochs 100
```

Slurm (one (N, reg) per call):
```
./slurm_submit.sh 2 1.0
MEM=16G TIME=02:00:00 ./slurm_submit.sh 4 0.5
```

Default optimizer is AdamW (lr=1e-3, wd=1e-2) with cosine LR schedule.
`--num-workers 2` matches the cluster's typical CPU-per-task allocation.

## Outputs

Per-run outputs are grouped by batch size under
`results/batch_size_{bs}/<run-name>/` (currently only `batch_size_128/`).
`run-name = N{N}_reg{reg:g}_h{hidden_dim}` (e.g. `N2_reg1_h1024`):

- `config.json` — all CLI args + resolved `hidden_dim` + `run_name`.
- `train.log` — per-epoch line-buffered progress, with `t_epoch Xs eta Xh Xm
  (total ~Xh Xm)`. Mirrors stdout.
- `metrics.npz` — keys:
  - `epoch` (E+1,) — `0` is init, `1..E` are post-train.
  - `test_acc` (E+1,) — per-frame test accuracy. NaN at epoch 0 (no eval).
  - `train_loss`, `train_ce`, `train_reg` (E+1,) — epoch averages.
  - `eigvals` (E+1, 10) — first 10 Laplacian eigenvalues of fc2 (ascending).
  - `epoch_time_seconds` (E,) — wall time per epoch (training + eval).
  - Scalars: `N`, `hidden_dim`, `reg_weight`.
- `weights.pt` — final `state_dict` (`torch.load(...)`).
- `accuracy.png` — test-accuracy curve.
- `eigenvalues.png` — 2 rows × 5 cols panel of `λ_1 .. λ_10` over training.
- `slurm-<jobid>.out` (if submitted via slurm) — duplicates `train.log` plus
  the host/CUDA banner from the submit wrapper.

Easy reload:
```python
import numpy as np, torch, json
run = "results/batch_size_128/N2_reg1_h1024"
d = np.load(f"{run}/metrics.npz")
weights = torch.load(f"{run}/weights.pt", map_location="cpu")
config = json.load(open(f"{run}/config.json"))
```

## Cross-run analysis (`plot_eigengap.py`)

Aggregates every run under `--results-root` (default
`./results/batch_size_128`) and writes one figure
(`modularity_analysis.png` in the same folder) with 5 panels arranged 2×3:

1. `λ_N` (dashed) and `λ_{N+1}` (solid) over epochs, one color per N.
2. Eigengap `λ_{N+1} − λ_N` over epochs.
3. Same eigengap against scaled epoch `t / N`.
4. Same eigengap zoomed to early training (`xlim=(0, 10)`, `ylim` auto-fit
   to the max gap value across N within the window × 1.2). Tunable via
   `--zoom-xmax`.
5. **Half-rise epoch vs N**: for each run, `t_{1/2}` is the first epoch
   where the gap reaches `(g_init + g_final) / 2`, linearly interpolated
   between samples. Quantifies how training time to half-modularity scales
   with the number of modules. N=1 is excluded throughout (`λ_1` is always
   0 for a connected Laplacian, so `λ_2 − λ_1` is not the same quantity as
   the modularity gap for N ≥ 2).

Run with `python plot_eigengap.py [--results-root PATH] [--reg 1.0]
[--zoom-xmax 10]`. Stdout reports per-N `g_init`, `g_final`, `g_half`,
`t_half`.

## Steps already taken

1. **Data + network skeleton** provided in `data.py` and `model.py`.
2. **Initial single-FC modularity test** (sum `|W|` on-pathway vs off-pathway)
   discarded as trivial.
3. **Hidden MLP head added** (fc1 → activation → fc2) and the regularizer
   moved to act on the *hidden* activation with *fc2* weights. Modularity
   metric switched to the Laplacian spectrum (graphs.py convention).
4. **Optimizer / activation sweep** at N=2, 20 epochs:
   - SGD + ReLU + reg=0.1: test_acc 0.92, eigengap at N=2 of 0.19. Works but
     weak modularity.
   - AdamW + ReLU + reg≥1.0: dead-ReLU collapse — hidden activations driven
     to zero, test_acc stuck at chance (0.10), reg term → 0.
   - **AdamW + LeakyReLU(0.1) + reg=1.0**: best operating point so far.
     test_acc 0.89, eigengap 0.665, `λ_N=2 = 0.11` at 20 epochs — modularity
     trajectory still rising at end of run.
5. **Saving + plotting + slurm infrastructure**: per-epoch timing and ETA in
   `train.log`, full metrics in `.npz`, plots in `.png`, single-job slurm
   wrapper.
6. **N=1..8 sweep submitted** at reg=1.0, 100 epochs, 4h walltime,
   `--mem = max(8, 4*N) GB`, batch size 128. Output dirs
   `results/batch_size_128/N{N}_reg1.0_h{512*N}/`. QOS limit is 2 GPUs per
   user, so jobs roll out 2 at a time. All 8 runs completed.
7. **Cross-run modularity analysis** in `plot_eigengap.py` (see "Cross-run
   analysis" above). Headline finding from the N=1..8 sweep at batch 128:
   the half-rise epoch `t_{1/2}` of the eigengap grows roughly linearly
   with N (5.0 → 11.4 → 19.8 → 27.6 → 33.2 → 47.8 epochs for N=2..7), and
   the final eigengap shrinks with N (0.71 at N=2 → 0.25 at N=8 — N=8
   hasn't finished rising within 100 epochs, so its `t_{1/2}` is an
   unreliable underestimate). The `t / N` rescaling in panel 3 does **not**
   collapse the curves — modularity emergence is slower than a simple
   linear stretch in N.
8. **Slide deck assembled** in `summary/` for the batch=128, reg=1.0 sweep
   (see "Slide deck (`summary/`)" below). All figures regenerable via
   `summary/build_figures.sh`. Deck is `summary/slides.tex` (12 frames,
   Beamer 16:9). Not compiled in-tree — no `pdflatex` on the cluster login
   node; compile elsewhere with `pdflatex slides.tex`.

## Slide deck (`summary/`)

Beamer slide deck covering setup + results from the N=1..8, batch=128,
reg=1.0 sweep. Everything in `summary/` is anchored to that sweep —
re-pointing to another sweep is a one-line edit (see "What to change for a
different sweep" below).

### Files
- `summary/slides.tex` — 12-frame Beamer 16:9 deck. References every PNG in
  `summary/figs/` by `\graphicspath{{figs/}}`. Compile with
  `pdflatex slides.tex` (login node has no `pdflatex`, so this is done
  off-cluster).
- `summary/build_figures.sh` — runs every figure script in order; outputs
  go to `summary/figs/`.
- `summary/make_task_figure.py` — one `ConcatCIFAR10` sample each for
  N=2,4,8 with per-frame class labels. Uses `torch.manual_seed(7)` for
  reproducibility. Loads `data/` directly (no download needed since the
  CIFAR-10 archive is already cached there).
- `summary/make_arch_figure.py` — two matplotlib schematics:
  `arch_pipeline.png` (boxes-and-arrows for the ResNet18 + MLP head, MLP
  highlighted with the regularizer annotation) and
  `modularity_pipeline.png` (|W| → S → L → eigenspectrum, illustrated on a
  synthetic N=3 block matrix).
- `summary/make_training_figures.py` — loads every run under
  `results/batch_size_128/` and writes `training_accuracy.png` (overlay
  across N), `eigvals_representative.png` (2×5 panel for the N=4 run, with
  the λ_N panel red-tinted and λ_{N+1} blue-tinted), and `lamN_pair.png`
  (λ_N dashed, λ_{N+1} solid, one color per N).
- `summary/make_modularity_figures.py` — writes `eigengap_overview.png`
  (1×2: gap-vs-epoch with the half-rise point `(t_{1/2}, g_{1/2})`
  scattered on each curve, and t_{1/2}-vs-N panel) and
  `final_modularity_vs_N.png` (λ_N and λ_{N+1}, plus the gap, at init vs
  final epoch). `half_rise_epoch` returns `(t_half, g_half)` so the scatter
  marker sits exactly on the half-rise level.
- `summary/make_weight_heatmap.py` — clusters fc2 hidden units (columns)
  with spectral k-means at k=N, then matches clusters to frames by
  max-affinity assignment so cluster i is the predicted "frame-i module".
  Writes `fc2_heatmap_N{2,4,8}.png` and the combined `fc2_heatmaps_grid.png`
  (cyan vertical lines = cluster boundary, white horizontal lines = frame
  boundary). Note: the reference `DetectCommunity` in
  `../modularisation_via_noise/utils/graphs.py` hardcodes `range(4)` in its
  reorder loop, so it isn't directly reusable for arbitrary N — this script
  inlines the k-N version.

### Regenerating
```
cd summary
bash build_figures.sh   # writes figs/*.png
pdflatex slides.tex     # off-cluster; pdflatex not installed on login node
```

### What to change for a different sweep
- **Different batch size**: edit `RESULTS = REPO / "results" / "batch_size_128"`
  in `make_training_figures.py`, `make_modularity_figures.py`, and
  `make_weight_heatmap.py` (three identical lines). Then rerun
  `bash build_figures.sh`. `slides.tex` uses no batch-size strings except
  the title/subtitle and a couple of bullet points (search for
  "batch size 128" / "batch=128").
- **Different reg_weight or different runs**: `load_runs()` in
  `make_training_figures.py` and `make_modularity_figures.py` filters
  nothing — it picks up every subdir of the results root. If you mix reg
  values you'll want to filter on `cfg["reg_weight"]`. The heatmap script
  hardcodes `N{N}_reg1.0_h{512*N}` in `load_fc2`; change that path
  template if needed.
- **Different representative N for the 2×5 eigvals panel**: pass a
  different `N_rep` to `make_eigvals_panel(runs, N_rep=4)` in
  `make_training_figures.py`.
- **Different N set in the heatmap grid**: edit `DEFAULT_N_LIST` at the
  top of `make_weight_heatmap.py`.

### Practical pointers (kept for future agents)
- All per-run data is self-describing — `config.json` tells you `N`,
  `hidden_dim`, `reg_weight`; `metrics.npz` has the time series.
- `run.py:save_plots` is the source for the matplotlib styling reused in
  the summary scripts.
- `analyse_connectivity` in `../modularisation_via_noise/utils/graphs.py`
  is the reference workflow for multi-layer reordering; the single-layer
  case here only needs spectral clustering on `fc2.weight.T`, and the
  `make_weight_heatmap.py` implementation is self-contained because the
  reference's `range(4)` reorder loop is k=4-specific.
- ConcatCIFAR10 frames are drawn at random per `__getitem__`, so the task
  figure relies on a fixed seed.

## Future steps

### Other directions worth queuing
- **Batch-size sweep**: re-do the N=1..8 sweep at smaller batch sizes
  (e.g. 32, 64) and re-run `plot_eigengap.py --results-root
  ./results/batch_size_{bs}` for each. Open question: does the
  near-linear `t_{1/2}` vs N relationship hold across batch sizes, or
  does the slope (or the long-N saturation at N=7,8) change with batch
  size? The directory layout (`results/batch_size_{bs}/`) is already set
  up for this — just pass `--batch-size` to `run.py` and put outputs
  under the matching subfolder.
- Sweep `reg_weight` at a fixed N to find the cleanest-cut operating point
  (reg=10 was confirmed too strong even with LeakyReLU at N=2).
- Baseline runs at `reg=0` for each N to bound the "no regularization"
  modularity (should be at chance / single connected component).
- Try alternative regularizers from
  `../modularisation_via_noise/utils/network.py` (`regularizer_L12`,
  `regularizer_L21`).
- Verify on-pathway vs off-pathway weight ratio in fc2 once modules are
  clean, using the input-index → frame-position mapping documented above.
