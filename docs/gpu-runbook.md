# GPU runbook — NVIDIA DGX B200

How to move a run from a laptop CPU onto the college DGX B200. Read
[§ There is no API key](#there-is-no-api-key) first if you were expecting to plug
a key into a config file.

---

## There is no API key

The access note describes a **web portal on the campus network**:

```
http://172.7.188.204/   (or)   http://10.10.50.240/
```

Log in with your official email, set a password, and you get an interactive
environment. This is a **shared cluster you log into**, not a hosted inference
API. Concretely:

- Both addresses are **private RFC-1918 IPs**. They resolve only from inside the
  campus network — from anywhere else the connection simply times out. Off
  campus you need the college VPN.
- There is **no API key, no endpoint URL, and no token** in this scheme. Nothing
  in this repository should be written to consume one.
- It serves plain **HTTP, not HTTPS**. Your password crosses the network
  unencrypted, so use a password you do not use anywhere else. Treat anything
  you upload as visible to whoever administers the cluster.

The consequence for this codebase is a simplification: **the integration is
`git clone`, not a network client.** The project is ordinary Python and PyTorch,
so the same scripts that run on your laptop run unchanged on the DGX. What
changes is one flag.

---

## The one flag

```bash
python scripts/train.py --profile dgx_b200
```

`configs/profiles/dgx_b200.yaml` is an *overlay* — it holds only the keys that
differ from the CPU baseline (device, precision, batch size, worker count, patch
geometry) and inherits everything else. There is no second copy of the config to
keep in sync, and a profile is forbidden by test from touching `data`, `loss` or
`evaluation`: **changing hardware must never silently change what is measured.**

---

## Procedure

### 1. Reach the portal

On campus, or on the college VPN, open one of the two addresses and register.
Then find out what it actually gives you — the answer determines everything
below:

| What you see | What it is | How to run |
| --- | --- | --- |
| A notebook interface | JupyterHub | Terminal tab → the commands below |
| "Workloads" / "Submit job" | NVIDIA Base Command or Run:ai | Submit a container job |
| A plain shell | SSH gateway or Open OnDemand | The commands below |
| "Launch container / image" | Container platform | Choose an NGC PyTorch image |

Most give you a terminal somewhere. That is all this project needs.

### 2. Get the code onto it

```bash
git clone https://github.com/BhargavaKandala/Satellite-Super-Resolution-using-DL.git
cd Satellite-Super-Resolution-using-DL
```

Data is deliberately **not** in the repository. Either upload your Sentinel-2
scenes to `data/raw/`, or generate synthetic ones to check the plumbing:

```bash
python scripts/prepare_dataset.py --synthetic --profile dgx_b200
```

### 3. Install PyTorch — the part that actually bites

**B200 is Blackwell, compute capability `sm_100`.** A plain `pip install torch`
can fetch a build with no `sm_100` kernels. It will import fine, report
`cuda.is_available() == True`, print a healthy banner, and then die at the first
convolution with:

```
CUDA error: no kernel image is available for execution on the device
```

`src/compute.py::check_cuda_build` detects this at startup and prints the
mismatch instead of letting you find out mid-epoch.

**Preferred — use the cluster's NGC container.** It ships a PyTorch built for the
exact hardware and needs no CUDA setup:

```bash
# inside nvcr.io/nvidia/pytorch:25.01-py3 or newer
pip install -r requirements.txt --no-deps rasterio opencv-python-headless \
    scikit-image PyYAML streamlit pytest
```

Install the geospatial and imaging packages, but **let the container keep its own
torch** — that is the whole point of using it.

**Fallback — wheels.** Needs CUDA 12.8 or newer:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 4. Verify before committing to a long run

```bash
python -c "
import torch
from src.compute import describe_device, check_cuda_build
d = torch.device('cuda')
print(describe_device(d))
print(check_cuda_build(d) or 'build OK')
print((torch.randn(8,8,device=d) @ torch.randn(8,8,device=d)).sum().item())
"
```

You want `sm_100` in the description, `build OK`, and a finite number. If the
matmul throws, the torch build is wrong — go back to step 3.

### 5. Run

```bash
python scripts/prepare_dataset.py --profile dgx_b200
python scripts/train.py           --profile dgx_b200
python scripts/evaluate.py        --profile dgx_b200 --downstream
```

Then release the node. The access note asks for this explicitly, and it is a
shared academic resource.

---

## What the profile changes, and why

| Setting | CPU | DGX B200 | Reason |
| --- | --- | --- | --- |
| `compute.device` | `cpu` | `cuda` | — |
| `compute.amp_dtype` | `auto` | `bf16` | Blackwell has native bf16; same exponent range as fp32, so no loss scaling and no overflow risk |
| `training.batch_size` | 8 | 64 | — |
| `training.num_workers` | 2 | 16 | The real bottleneck — see below |
| `training.compile` | off | on | Warm-up cost amortises over 200 epochs |
| `patches.hr_patch_size` | 128 | 192 | More spatial context per sample |
| `training.epochs` | 20 | 200 | — |

Note `num_workers` rises 8× while `batch_size` rises only 8×. That is
intentional.

---

## Read this before you scale up

**EDSR-Lite is 1.2 M parameters.** One B200 carries 180 GB of HBM3e. This model
does not remotely stress that hardware — the GPU will spend most of its time
waiting on the data loader, which is why the profile raises worker count
aggressively. Simply moving the current model to a B200 buys you a faster version
of the same result.

The hardware is worth having for the things it *unlocks*, in order of value:

1. **Real data at volume.** The single highest-value change, and the only one
   that turns the metrics from plumbing checks into science.
2. **A larger architecture.** SwinIR or a Transformer SR model registered via
   `@register_model` — a drop-in swap the pipeline already supports.
3. **Larger patches and longer schedules.** Better context, better convergence.
4. **Multi-GPU.** There are 8 B200s in a DGX. Not implemented; single-GPU is far
   from saturated, so DDP would be premature.

A B200 will train the wrong model on synthetic data very quickly. Fix the data
first.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Portal times out | Off campus — private IP | Connect to the college VPN |
| `CUDA was requested but torch.cuda.is_available() is False` | No GPU visible, or driver/torch mismatch | Check `nvidia-smi`; confirm the job requested a GPU |
| `no kernel image is available` | torch has no `sm_100` kernels | Step 3 — use the NGC container |
| `CUDA out of memory` | Batch too large | `--batch-size 32`; unlikely on 180 GB |
| Training slower than expected | Data loader starved | Raise `num_workers`; keep data on local disk, not a network mount |
| First epoch very slow, rest fast | `compile: true` warming up | Expected — it pays back over a long run |
