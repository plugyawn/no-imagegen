# srno_recursive_circle_generator.py
"""
CTF-NO
=======================================

Train a fourier neural operator recursively 4→8→16→32→64 to grow an image from a tiny seed.

Run
----
```bash
pip install torch torchvision einops tqdm matplotlib
python srno_recursive_circle_generator.py --epochs 30 --batch 64
```
This should finish on a single consumer GPU in <15 min and emit samples to
`outputs/cf_samples.png`.
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights  # type: ignore
from pytorch_msssim import ssim  # type: ignore

# ---------------------------------------------------------------------------------
# Synthetic data — coloured discs --------------------------------------------------
# ---------------------------------------------------------------------------------

class ImagenetteSubset(Dataset):
    """A small ImageNet-derived dataset (Imagenette) automatically downloaded.

    Returns RGB tensors scaled to the *hr* resolution (default 64×64) and
    normalised to [0, 1].  The Imagenette corpus contains 10 varied classes and
    is ~1 GB on disk ─ small enough for quick experiments yet sufficiently
    diverse to stress-test a model on natural images.
    """

    _URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

    def __init__(self, root: str | Path = "data/imagenette2-320", *, split: str = "train", hr: int = 64,
                 subset: int | None = None, download: bool = True):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.hr = hr

        if download:
            self._ensure_data()

        # Imagenette directory has sub-folders *train* and *val* laid out as
        # class-named directories, perfectly suited for ``ImageFolder``.
        tfm = transforms.Compose([
            transforms.Resize(hr, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(hr),
            transforms.ToTensor(),  # converts to [0,1] range
        ])
        ds = datasets.ImageFolder(str(self.root / split), transform=tfm)

        # Optionally subsample for even faster experiments ─ random but
        # deterministic thanks to torch's RNG.
        if subset is not None and subset < len(ds):
            idx = torch.randperm(len(ds))[:subset]
            self.ds = torch.utils.data.Subset(ds, idx)
        else:
            self.ds = ds

    # ------------------------------------------------------------------ utils -----

    def _ensure_data(self):
        if self.root.exists():
            return  # already downloaded
        self.root.parent.mkdir(parents=True, exist_ok=True)
        tgz_path = self.root.parent / "imagenette2-320.tgz"
        if not tgz_path.exists():
            print("Downloading Imagenette… (~1 GB)")
            import urllib.request, shutil
            with urllib.request.urlopen(self._URL) as resp, open(tgz_path, "wb") as f:
                shutil.copyfileobj(resp, f)
        print("Extracting Imagenette…")
        import tarfile
        with tarfile.open(tgz_path) as tar:
            tar.extractall(path=self.root.parent)

    # ---------------------------------------------------------------- Dataset API --

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]  # discard label; generative task doesn't need it
        return img

# ---------------------------------------------------------------------------------
# CIFAR-10 dataset wrapper (32×32 → resized to 64×64) ------------------------------
# ---------------------------------------------------------------------------------

class CIFAR10Dataset(Dataset):
    """Convenience wrapper around ``torchvision.datasets.CIFAR10``.

    Images are up-scaled to *hr* (default 64) so they fit the current coarse-to-fine
    pyramid.  Returns tensors in [0,1].
    """

    def __init__(self, root: str | Path = "data/cifar10", *, split: str = "train", hr: int = 64,
                 download: bool = True):
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize(hr, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.ds = datasets.CIFAR10(str(root), train=(split == "train"), download=download, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        return img

# ---------------------------------------------------------------------------------
# Utility: average-pool downsample by factor 2 -------------------------------------
# ---------------------------------------------------------------------------------

def downsample(x: torch.Tensor) -> torch.Tensor:
    """Average-pool downsample by factor 2.

    Works with tensors shaped (B, C, H, W) or (C, H, W) by delegating to
    ``torch.nn.functional.avg_pool2d``.
    """
    # If the input is unbatched (C, H, W), add a dummy batch dim to satisfy
    # avg_pool2d, then squeeze it back afterwards.
    added_batch_dim = x.ndim == 3  # (C, H, W)
    if added_batch_dim:
        x = x.unsqueeze(0)

    x = F.avg_pool2d(x, kernel_size=2, stride=2)

    if added_batch_dim:
        x = x.squeeze(0)
    return x


def make_pyramid(hr: torch.Tensor, levels: int = 4) -> List[torch.Tensor]:
    pyr = [hr]
    for _ in range(levels):
        pyr.insert(0, downsample(pyr[0]))
    return pyr  # [4²,8²,16²,32²,64²]

# ---------------------------------------------------------------------------------
# Fourier Neural Operator layer ----------------------------------------------
# ---------------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        self.scale = 1 / (in_ch * out_ch)
        self.weight = nn.Parameter(self.scale * torch.randn(in_ch, out_ch, modes, modes, 2))

    def compl_mul(self, a, w):
        ar, ai = a[..., 0], a[..., 1]
        wr, wi = w[..., 0], w[..., 1]
        real = torch.einsum("bihw,iohw->bohw", ar, wr) - torch.einsum("bihw,iohw->bohw", ai, wi)
        imag = torch.einsum("bihw,iohw->bohw", ar, wi) + torch.einsum("bihw,iohw->bohw", ai, wr)
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):  # x: (b,c,h,w)
        B, _, H, W = x.shape

        # Forward FFT (real-to-complex) and convert to real-imag pair
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        x_ft = torch.view_as_real(x_ft)  # (B, C, H, W//2+1, 2)

        # Limit the number of Fourier modes to those available in the input to
        # avoid size mismatches when H or W is smaller than self.modes.
        Hm = min(self.modes, H)
        Wm = min(self.modes, W // 2 + 1)

        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, 2, device=x.device, dtype=x.dtype)

        # Multiply only the retained modes.
        out_ft[:, :, :Hm, :Wm] = self.compl_mul(
            x_ft[:, :, :Hm, :Wm],
            self.weight[:, :, :Hm, :Wm],
        )

        # Inverse FFT back to spatial domain
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfftn(out_ft, s=(H, W), dim=[-2, -1])
        return x

class FNO_Double(nn.Module):
    """Recursive FNO block for *next-scale prediction*.

    Given an up-sampled (bilinear) low-resolution frame of size (2r×2r) this
    module predicts the high-frequency *residual* that, when added back to the
    up-sample, forms the next-scale RGB image.  Random Fourier features in the
    input (via a `noise_ch` channel) enable stochastic generation at inference
    time while grid coordinates supply positional information – both are
    essential for an *autoregressive* coarse-to-fine pipeline.
    """

    def __init__(
        self,
        *,
        width: int = 96,
        modes: int = 42,
        layers: int = 4,
        noise_ch: int = 1,
        concat_coords: bool = True,
    ):
        super().__init__()
        self.noise_ch = noise_ch
        self.concat_coords = concat_coords

        in_ch = 3 + noise_ch + (2 if concat_coords else 0)
        self.in_proj = nn.Conv2d(in_ch, width, kernel_size=1)

        # A *single* spectral convolution shared across layers (true weight-tying).
        self.spec = SpectralConv2d(width, width, modes)

        # Decide number of groups for GroupNorm.
        ng = max(1, min(8, width // 16))

        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                    "dense": nn.Conv2d(width, width, 1),
                        # depth-wise 3×3 conv captures local edges; followed by pointwise fuse
                        "local": nn.Conv2d(width, width, 3, padding=1, groups=width),
                        "norm": nn.GroupNorm(ng, width),
                    }
                )
            )

        self.out_proj = nn.Conv2d(width, 3, kernel_size=1)

    @staticmethod
    def _coord_grid(B: int, H: int, W: int, device, dtype):
        """Return a normalised ([-1,1]) 2-D coordinate grid with shape (B,2,H,W)."""
        y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), dim=0)  # (2,H,W)
        return grid.unsqueeze(0).repeat(B, 1, 1, 1)       # (B,2,H,W)

    def forward(self, upsampled: torch.Tensor, *, noise: Optional[torch.Tensor] = None, sigma: float = 0.1):
        """Return high-resolution prediction.

        Args:
            upsampled: Bilinear-upsampled LR tensor of shape (B,3,2r,2r).
            noise:     Optional stochastic input of shape (B,noise_ch,2r,2r).
                         If *None* random N(0,sigma) noise is used (sampling).
                         Pass zeros at training time for deterministic targets.
        """
        B, _, H, W = upsampled.shape
        if noise is None:
            noise = sigma * torch.randn(B, self.noise_ch, H, W, device=upsampled.device, dtype=upsampled.dtype)

        feats = [upsampled, noise]
        if self.concat_coords:
            feats.append(self._coord_grid(B, H, W, upsampled.device, upsampled.dtype))
        x = torch.cat(feats, dim=1)                          # (B,C,H,W)

        x = F.gelu(self.in_proj(x))
        residual_scale = 0.5
        for block in self.layers:
            delta = self.spec(x) + block["dense"](x) + block["local"](x)
            x = x + residual_scale * delta
            x = block["norm"](x)
            x = F.gelu(x)

        residual = self.out_proj(x)
        return (upsampled + residual).clamp(0.0, 1.0)

# ---------------------------------------------------------------------------------
# Training & validation -----------------------------------------------------------
# ---------------------------------------------------------------------------------

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return -10 * torch.log10(mse + 1e-8)


# ------------------------------ perceptual backbone --------------------------------

# We cache a trimmed VGG-16 (layers up to conv3_3) for perceptual loss.

_VGG_FEAT: Optional[nn.Module] = None


def _get_vgg_feat(device) -> nn.Module:
    global _VGG_FEAT
    if _VGG_FEAT is None:
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[:16]  # up to relu3_3
        for p in vgg.parameters():
            p.requires_grad_(False)
        _VGG_FEAT = vgg.to(device).eval()
    return cast(nn.Module, _VGG_FEAT)

# ---------------------------------------------------------------------------------
# Training epoch -----------------------------------------------------------
# ---------------------------------------------------------------------------------

def _gradients(x: torch.Tensor):
    """Return simple finite-difference gradients along x and y (same shape as x)."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    gy[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return gx, gy

def _laplacian_pyramid(x: torch.Tensor, levels: int = 3):
    pyr = []
    current = x
    for _ in range(levels):
        down = F.avg_pool2d(current, kernel_size=2, stride=2)
        up = F.interpolate(down, size=current.shape[-2:], mode="nearest")
        lap = current - up
        pyr.append(lap)
        current = down
    pyr.append(current)
    return pyr

# ------------------------------ PatchGAN discriminator ----------------------------


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        # 64×64 → 4×4 receptive field ~ 16×16 patches
        chs = [64, 128, 256, 512]
        layers = []
        prev = in_ch
        for c in chs:
            layers.append(nn.Conv2d(prev, c, 4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = c
        layers.append(nn.Conv2d(prev, 1, 1))  # output patch map
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_epoch(model, loader, opt, device, epoch: int | None = None,
               *, disc: Optional[nn.Module] = None, disc_opt: Optional[torch.optim.Optimizer] = None,
               lambda_vgg: float = 0.1, lambda_ssim: float = 0.8, lambda_grad: float = 0.2,
               lambda_lap: float = 0.1, lambda_adv: float = 1e-3):
    """Run one training epoch and return average L1 loss.

    Progress is displayed with a tqdm bar. If *epoch* is provided, it will be
    shown in the bar description.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0  # for accurate running average

    pbar_desc = f"Train e{epoch:02d}" if epoch is not None else "Train"
    pbar = tqdm(loader, desc=pbar_desc, leave=False)

    for hr in pbar:
        hr = hr.to(device)
        pyr = make_pyramid(hr)  # [4,8,16,32,64]
        loss = 0
        wt = 1
        max_lvl = len(pyr) - 2  # index of 64×64 prediction step
        for lvl in range(len(pyr) - 1):
            lr = pyr[lvl]
            gt = pyr[lvl + 1]
            up = F.interpolate(lr, size=gt.shape[-2:], mode="bilinear")
            noise = torch.zeros(hr.size(0), model.noise_ch, gt.shape[-2], gt.shape[-1], device=hr.device, dtype=hr.dtype)
            pred = model(up, noise=noise)
            l1 = F.l1_loss(pred, gt)
            loss += wt * l1

            if lvl == max_lvl and (lambda_vgg > 0 or lambda_ssim > 0):
                # High-res perceptual & MS-SSIM losses
                if lambda_vgg > 0:
                    vgg = _get_vgg_feat(device)
                    feat_pred = vgg(pred)
                    feat_gt = vgg(gt)
                    # print("vgg loss", F.l1_loss(feat_pred, feat_gt.detach()))
                    loss += lambda_vgg * F.l1_loss(feat_pred, feat_gt.detach())

                if lambda_ssim > 0:
                    ssim_loss = 1 - ssim(pred, gt, data_range=1.0, size_average=True)
                    loss += lambda_ssim * ssim_loss

                if lambda_grad > 0:
                    gx_p, gy_p = _gradients(pred)
                    gx_g, gy_g = _gradients(gt)
                    grad_loss = F.l1_loss(gx_p, gx_g) + F.l1_loss(gy_p, gy_g)
                    loss += lambda_grad * grad_loss

            if lambda_lap > 0 and lvl == max_lvl:
                lap_pred = _laplacian_pyramid(pred, 3)
                lap_gt = _laplacian_pyramid(gt, 3)
                lap_loss = sum(F.l1_loss(a, b) for a, b in zip(lap_pred, lap_gt))
                loss += lambda_lap * lap_loss

            wt *= 2

        # ---------------- GAN discriminator update (only highest-res pred) --------
        if disc is not None and disc_opt is not None and lambda_adv > 0:
            disc.requires_grad_(True)
            disc_opt.zero_grad()
            real_logits = disc(gt.detach())
            fake_logits = disc(pred.detach())
            d_loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
            d_loss.backward()
            disc_opt.step()

            # generator adversarial term
            disc.requires_grad_(False)
            g_adv = -disc(pred).mean()
            loss += lambda_adv * g_adv

        opt.zero_grad()
        loss.backward()
        opt.step()
        batch_loss = loss.item()
        total_loss += batch_loss * hr.size(0)
        num_samples += hr.size(0)

        # Update bar postfix with current & running avg
        pbar.set_postfix({"batch_L1": f"{batch_loss:.4f}",
                          "avg_L1": f"{total_loss / num_samples:.4f}"})

    return total_loss / num_samples


# ------------------------------ validation ----------------------------------------

def validate(model, loader, device, epoch: int | None = None):
    """Validate model and return (avg L1, avg PSNR).

    Shows a tqdm bar over the validation loader.
    """
    model.eval()
    l1s, psnrs = [], []
    pbar_desc = f"Val  e{epoch:02d}" if epoch is not None else "Val"
    pbar = tqdm(loader, desc=pbar_desc, leave=False)

    with torch.no_grad():
        for hr in pbar:
            hr = hr.to(device)
            pyr = make_pyramid(hr)
            lvl_loss, lvl_psnr = 0.0, 0.0
            for lvl in range(len(pyr) - 1):
                lr = pyr[lvl]
                gt = pyr[lvl + 1]
                up = F.interpolate(lr, size=gt.shape[-2:], mode="bilinear")
                noise = torch.zeros(hr.size(0), model.noise_ch, gt.shape[-2], gt.shape[-1], device=hr.device, dtype=hr.dtype)
                pred = model(up, noise=noise)
                lvl_loss += F.l1_loss(pred, gt).item()
                lvl_psnr += psnr(pred, gt).item()
            l1 = lvl_loss / 4
            p = lvl_psnr / 4
            l1s.append(l1)
            psnrs.append(p)
            pbar.set_postfix({"L1": f"{l1:.4f}", "PSNR": f"{p:.2f}"})
  
    return float(np.mean(l1s)), float(np.mean(psnrs))

# ---------------------------------------------------------------------------------
# Inference: generate from noise --------------------------------------------------
# ---------------------------------------------------------------------------------

def generate_samples(
    model,
    device,
    outdir: Path,
    n: int = 6,
    val_ds: Optional[Dataset] = None,
):
    """Write *n* coarse-to-fine image chains to *outdir*.

    If *val_ds* is provided, we take HR samples from it, downsample them to 4 × 4
    and let the model autoregressively upscale back to 64 × 64.  Otherwise we
    fall back to pure noise seeds (as before).
    """
    model.eval()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n, 6 if val_ds else 5, figsize=(10 if val_ds else 8, n * 1.6))

    rng = np.random.default_rng()

    with torch.no_grad():
        for i in tqdm(range(n), desc="Sampling"):
            if val_ds is not None:
                # --- seed from validation HR image ---
                idx = int(rng.integers(0, len(val_ds)))
                hr_img = val_ds[idx]  # returns (3,64,64) tensor normalised 0-1
                hr_img = hr_img.to(device)
                x = F.interpolate(hr_img.unsqueeze(0), size=(4, 4), mode="area")
                imgs = [hr_img.cpu()]  # ground truth reference
                col_titles = ["GT 64×64", "bilinear↑4", "8×8", "16×16", "32×32", "64×64"]
            else:
                x = torch.rand(1, 3, 4, 4, device=device)
                imgs = [F.interpolate(x, 64, mode="bilinear").cpu().squeeze(0)]
                col_titles = ["bilinear↑4", "8×8", "16×16", "32×32", "64×64"]

                # For the noise-seed case we already have the bilinear-up image
                # as the first panel, so we *do not* append another copy.
                pass

            # For the val-seed case we still need to add the bilinear-up baseline
            if val_ds is not None:
                imgs.append(F.interpolate(x, 64, mode="bilinear").cpu().squeeze(0))

            # Autoregressive upscales (4 steps: 4→8→16→32→64)
            sigmas = [0.4, 0.2, 0.1, 0.05]  # for the 4 upscales (4→8→…→64)
            for s in sigmas:
                up = F.interpolate(x, scale_factor=2, mode="bilinear")
                x = model(up, sigma=s)
                imgs.append(x.cpu().squeeze(0))

            for j, img in enumerate(imgs):
                axes[i, j].imshow(rearrange(img, "c h w -> h w c"))
                axes[i, j].axis("off")
                if i == 0:
                    axes[0, j].set_title(col_titles[j], fontsize=9)

    plt.tight_layout()
    plt.savefig(outdir / "cf_samples.png")
    plt.close(fig)

# ---------------------------------------------------------------------------------
# Main ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

def main():

    device = torch.device("cuda")

    # Switch to CIFAR-10 (50 000 train / 10 000 test) upscaled to 64×64.
    train_ds = CIFAR10Dataset(split="train", hr=64, download=True)
    val_ds = CIFAR10Dataset(split="val", hr=64, download=True)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    model = FNO_Double(width=192, modes=42, layers=6).to(device)

    disc = PatchDiscriminator().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

    best = -math.inf
    train_hist: list[float] = []
    val_hist: list[float] = []
    psnr_hist: list[float] = []

    epoch_iter = tqdm(range(1, 25 + 1), desc="Epochs")
    for epoch in epoch_iter:
        tr = train_epoch(model, train_loader, opt, device, epoch,
                          disc=disc, disc_opt=disc_opt)
        vl, ps = validate(model, val_loader, device, epoch)

        train_hist.append(tr)
        val_hist.append(vl)
        psnr_hist.append(ps)

        out_root = Path("outputs")
        # Update curves
        # plot_metrics(train_hist, val_hist, psnr_hist, out_root)

        # Generate sample images for this epoch
        generate_samples(model, device, out_root / f"epoch_{epoch:02d}", n=6, val_ds=val_ds)

        log_str = f"Epoch {epoch:02d} | train L1 {tr:.4f} | val L1 {vl:.4f} | val PSNR {ps:.2f}"
        epoch_iter.set_postfix_str(log_str)
        print(log_str)

        if ps > best:
            best = ps
            torch.save(model.state_dict(), "best_cf_srno.pth")

    print("training done | best PSNR so far=", best)
    model.load_state_dict(torch.load("best_cf_srno.pth", map_location=device))
    generate_samples(model, device, Path("outputs"), val_ds=val_ds)
    print("written to outputs/cf_samples.png")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------
# Plotting utility ----------------------------------------------------------------
# ---------------------------------------------------------------------------------

def plot_metrics(train_L1s: list[float], val_L1s: list[float], val_PSNRs: list[float], outdir: Path):
    """Plot training/validation curves up to the current epoch."""
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_L1s) + 1)

    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(epochs, train_L1s, label="train L1", color="blue")
    ax1.plot(epochs, val_L1s, label="val L1", color="green")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("L1 loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_PSNRs, label="val PSNR", color="red")
    ax2.set_ylabel("PSNR (dB)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "training_curves.png", dpi=150)
    plt.close(fig)
