# Recomputing the RbA score maps on Google Colab

The RbA method (Nayal et al., ICCV 2023, https://github.com/NazirNayal8/RbA) requires
Detectron2 plus the compiled **MSDeformAttn** CUDA kernel from Mask2Former. This does
not build reliably on Windows, so all RbA score maps in this thesis were computed on
**Google Colab (Tesla T4, free tier)** and merged into the local caches afterwards.

You do **not** need to do this yourself — the finished maps are part of the Zenodo
archive (`python scripts/download_score_maps.py`). The recipe below documents exactly
how they were created, for full reproducibility.

## 1. Environment (Colab, T4 GPU)

The dependency pinning matters — newer torch/numpy combinations break Detectron2 and
the kernel build:

```bash
# torch matching CUDA 12.1, numpy must stay < 2
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"

# Detectron2 from source
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# fairscale would otherwise pull numpy back to 2.x — pin it
pip install fairscale==0.4.13

# RbA repo
git clone https://github.com/NazirNayal8/RbA.git
cd RbA
pip install -r requirements.txt   # check that numpy stays < 2 afterwards!
```

## 2. Build the MSDeformAttn CUDA kernel

The T4 has compute capability 7.5 — set it explicitly, otherwise the build targets the
wrong architecture:

```bash
cd mask2former/modeling/pixel_decoder/ops
TORCH_CUDA_ARCH_LIST=7.5 sh make.sh
```

## 3. Checkpoint

Download the **`swin_b_1dl`** checkpoint (Swin-B backbone, 1 decoder layer) from the
links in the RbA README and place it in `RbA/ckpts/swin_b_1dl/`.

## 4. Compute the score maps

Upload the images (L&F test split / SMIYC validation images) to Colab (e.g. via Google
Drive). For every image the RbA score map is computed as

```
RbA(x) = -Σ_k σ(L_k(x))
```

(sum of sigmoid-activated per-class mask logits over all K classes; high values = OoD)
and saved per image as `<image_stem>_rba.npz` containing a float16 array `rba_map`
of shape [H, W] at the original image resolution.

<!-- TODO(Gerd): das eigentliche Colab-Notebook (rba_score_maps.ipynb) hier ablegen —
     dein bestehendes Colab-Notebook exportieren via File → Download → .ipynb -->

## 5. Merge into the local caches (back on the local machine)

```powershell
cd evaluation
python merge_rba_into_score_maps.py        # Lost & Found
python merge_rba_into_score_maps_smiyc.py  # SMIYC (both tracks)
```

Filename matching: L&F RbA files are matched to cache files via the image stem with the
`_leftImg8bit` suffix handled automatically; SMIYC filenames match directly. Both merge
scripts support `--dry_run` to verify the matching before writing.
