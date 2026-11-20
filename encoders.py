"""Registry of patch encoders for the PLISM-sm study.

Every encoder exposes the same minimal interface::

    enc = get_encoder("uni")          # cheap: nothing is downloaded/loaded yet
    enc.load(device="cuda")           # downloads weights (needs HF_TOKEN for gated models), builds the model
    x = enc.preprocess_one(rgb_uint8) # (3, H, W) float32, model-specific resize/crop/normalise (CPU, numpy+PIL)
    z = enc.encode(np.stack([x...]))  # (B, D) float32 numpy, fp16 autocast on GPU; analysis L2-normalises later

Preprocessing is deliberately done with PIL + numpy so it can run in worker threads and so that the
registry/preprocessing can be tested without torch installed.

Provenance
----------
Model-loading code and normalisation constants were cross-checked against the wrappers in
https://github.com/owkin/plism-benchmark (README: CC BY 4.0; code in ``plismbench/models/*.py``), the CONCH
and Prov-GigaPath GitHub READMEs / source (fetched), and model-card knowledge.  Hugging Face was NOT
reachable when this file was written, so model cards could not be re-read; every detail that could not be
confirmed against a fetched source is marked ``# UNVERIFIED``.  Snippets adapted from plism-benchmark are
marked ``(plism-benchmark)``.  Nothing was executed against real weights.

ALL ``license``/``gated`` fields in the registry are from memory and UNVERIFIED (HF unreachable): re-check each
model card before publishing.

NOTE on input size: PLISM-sm tiles are 512x512.  plism-benchmark feeds 224x224 tiles; here every tile is
*resized* (area/bicubic, PIL) to the model's native resolution (224, CONCH: 448) -- see README.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)  # CONCH (conch/open_clip_custom/constants.py, fetched)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)  # (plism-benchmark) / H-optimus model cards
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    pretty: str
    dim: int
    resize: int  # shortest-side resize target (bicubic)
    crop: int  # centre-crop size (== resize for a pure resize)
    mean: tuple
    std: tuple
    source: str  # HF repo / package
    license: str
    gated: bool
    max_batch: int = 256  # upper bound on batch size (memory), e.g. CONCH at 448 px
    notes: str = ""


class Encoder:
    """Base class: handles preprocessing; subclasses provide ``_build`` (torch module -> (B, D))."""

    def __init__(self, spec: EncoderSpec, builder: Callable | None):
        self.spec = spec
        self._builder = builder
        self.model = None
        self.device = "cpu"
        self.fp16 = False

    # ---- preprocessing (numpy / PIL only) -------------------------------------------------
    def preprocess_one(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 (H, W, 3) -> float32 (3, crop, crop), resized/cropped/normalised."""
        s = self.spec
        im = Image.fromarray(rgb)
        w, h = im.size
        scale = s.resize / min(w, h)
        nw, nh = max(s.resize, round(w * scale)), max(s.resize, round(h * scale))
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.BICUBIC)
        if nw != s.crop or nh != s.crop:
            l, t = (nw - s.crop) // 2, (nh - s.crop) // 2
            im = im.crop((l, t, l + s.crop, t + s.crop))
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - np.asarray(s.mean, np.float32)) / np.asarray(s.std, np.float32)
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    # ---- model ------------------------------------------------------------------------------
    def load(self, device: str | None = None, fp16: bool = True) -> "Encoder":
        import torch  # lazy: not needed for preprocessing / tests

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.fp16 = device, bool(fp16 and device.startswith("cuda"))
        module = self._builder(self, os.environ.get("HF_TOKEN"))
        module = module.eval().to(device)
        if device.startswith("cuda") and torch.cuda.device_count() > 1 and self.spec.key not in NO_DATAPARALLEL:
            module = torch.nn.DataParallel(module)  # 2x T4: split each batch over both GPUs
        self.model = module
        return self

    def encode(self, batch: np.ndarray) -> np.ndarray:
        """float32 (B, 3, H, W) preprocessed batch -> float32 (B, D) numpy."""
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):  # T4: fp16, no bf16
                z = self.model(x)
        return z.float().cpu().numpy()


NO_DATAPARALLEL = {"conch", "conchv15"}  # custom forward wrappers break under DataParallel (seen on Kaggle 2x T4)


class DebugEncoder(Encoder):
    """Torch-free fake encoder for tests: fixed random projection of a 16x16 downsampled image."""

    def __init__(self, spec: EncoderSpec):
        super().__init__(spec, None)
        rng = np.random.default_rng(0)
        self._W = rng.standard_normal((3 * 16 * 16, spec.dim)).astype(np.float32) / 16.0

    def load(self, device=None, fp16=True):
        return self

    def encode(self, batch: np.ndarray) -> np.ndarray:
        b = batch.shape[0]
        small = batch.reshape(b, 3, 16, 2, 16, 2).mean((3, 5)).reshape(b, -1)  # 32 -> 16 px block-average
        return (small @ self._W).astype(np.float32)


# ======================================================================================
# builders: (encoder, hf_token) -> torch.nn.Module returning (B, D)
# ======================================================================================


def _timm_cfg_check(enc: Encoder, model) -> None:
    """Log (never override) any mismatch between our hard-coded mean/std and the timm pretrained_cfg."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    for k, ours in (("mean", enc.spec.mean), ("std", enc.spec.std)):
        theirs = cfg.get(k)
        if theirs is not None and not np.allclose(theirs, ours, atol=1e-3):
            print(f"[encoders] WARNING {enc.spec.key}: timm pretrained_cfg {k}={tuple(theirs)} != spec {ours}", file=sys.stderr)


def _use_timm_data_config(enc: Encoder, model) -> None:
    """Baselines only: take mean/std (and input size) from timm's resolved config instead of hard-coding."""
    import timm

    cfg = timm.data.resolve_data_config({}, model=model)
    enc.spec = replace(enc.spec, mean=tuple(cfg["mean"]), std=tuple(cfg["std"]))  # keep spec resize/crop
    print(f"[encoders] {enc.spec.key}: data config from timm -> {cfg}")


def _timm_plain(hub: str, **kw):
    """timm hub model whose forward() already returns the pooled embedding (UNI, UNI2-h, H-optimus, GigaPath)."""

    def build(enc, token):
        import timm

        m = timm.create_model(hub, pretrained=True, **kw)
        _timm_cfg_check(enc, m)
        return m

    return build


def _build_virchow2(enc, token):
    """Virchow2: tokens (B, 1+4+256, 1280); embedding = [CLS, mean(patch tokens)] = 2560 (model card)."""
    import timm, torch

    base = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                             mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)  # (plism-benchmark)
    _timm_cfg_check(enc, base)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            # tokens 1..4 are register tokens, patch tokens start at index 5 (model card / plism-benchmark)
            return torch.cat([out[:, 0], out[:, 5:].mean(1)], dim=-1)

    return Wrap(base)


def _build_h0mini(cls_and_mean: bool):
    def build(enc, token):
        import timm, torch

        base = timm.create_model("hf-hub:bioptimus/H0-mini", pretrained=True,
                                 mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)  # (plism-benchmark)
        _timm_cfg_check(enc, base)

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x)  # tokens (B, 1+4?+N, 768)  # UNVERIFIED: number of register tokens (5 assumed below)
                if cls_and_mean:
                    return torch.cat([out[:, 0], out[:, 5:].mean(1)], dim=-1)  # UNVERIFIED patch start index
                return out[:, 0]  # CLS only, as in plism-benchmark

        return Wrap(base)

    return build


def _build_uni(enc, token):
    import timm

    m = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)  # (plism-benchmark)
    _timm_cfg_check(enc, m)
    return m


def _build_uni2h(enc, token):
    import timm, torch

    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)  # (plism-benchmark) == model card
    m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw)
    _timm_cfg_check(enc, m)
    return m


def _build_conch(enc, token):
    """CONCH v1 image encoder only: pre-projection ViT-B/16 output (512-d), as in the CONCH README."""
    import torch
    from conch.open_clip_custom import create_model_from_pretrained  # pip install git+https://github.com/mahmoodlab/CONCH.git

    model, _ = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=token)
    vm = getattr(model.visual, "image_mean", None)
    if vm is not None and not np.allclose(vm, enc.spec.mean, atol=1e-4):
        print(f"[encoders] WARNING conch mean {vm} != spec {enc.spec.mean}", file=sys.stderr)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # README: model.encode_image(image, proj_contrast=False, normalize=False) -> 512-d image embedding
            return self.m.encode_image(x, proj_contrast=False, normalize=False)

    return Wrap(model)


def _build_conchv15(enc, token):
    """CONCH v1.5 is distributed inside TITAN (remote code); ``return_conch()`` gives (patch encoder, transform)."""
    import torch
    from transformers import AutoModel

    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True, token=token)  # (plism-benchmark)
    conch, eval_transform = titan.return_conch()
    print(f"[encoders] conchv1_5 eval_transform (compare with spec!): {eval_transform}")
    return conch  # forward(images) -> (B, 768)  (plism-benchmark)


def _build_phikon2(enc, token):
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained("owkin/phikon-v2", token=token)  # (plism-benchmark)
    try:  # take the processor's own normalisation if the hub is reachable
        from transformers import AutoImageProcessor

        p = AutoImageProcessor.from_pretrained("owkin/phikon-v2", token=token)
        print(f"[encoders] phikon-v2 processor mean={p.image_mean} std={p.image_std} size={getattr(p, 'size', None)}")
        enc.spec = replace(enc.spec, mean=tuple(p.image_mean), std=tuple(p.image_std))
    except Exception as e:  # pragma: no cover
        print(f"[encoders] could not read phikon-v2 processor ({e}); using spec values", file=sys.stderr)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).last_hidden_state[:, 0]  # CLS (plism-benchmark)

    return Wrap(m)


def _build_baseline(hub_name: str, **kw):
    def build(enc, token):
        import timm

        m = timm.create_model(hub_name, pretrained=True, num_classes=0, **kw)
        _use_timm_data_config(enc, m)
        return m

    return build


# ======================================================================================
# registry
# ======================================================================================
_S = EncoderSpec
_SPECS: list[tuple[EncoderSpec, Callable | None]] = [
    (_S("uni", "UNI", 1024, 224, 224, IMAGENET_MEAN, IMAGENET_STD, "hf-hub:MahmoodLab/uni", "CC-BY-NC-ND 4.0", True,
        notes="Resize 224 per model card; init_values=1e-5, dynamic_img_size (plism-benchmark)"), _build_uni),
    (_S("uni2h", "UNI2-h", 1536, 224, 224, IMAGENET_MEAN, IMAGENET_STD, "hf-hub:MahmoodLab/UNI2-h", "CC-BY-NC-ND 4.0", True,
        notes="ViT-H/14 w/ 8 register tokens, SwiGLU; kwargs from model card"), _build_uni2h),
    (_S("conch", "CONCH", 512, 448, 448, OPENAI_MEAN, OPENAI_STD, "MahmoodLab/CONCH (pip: conch)", "CC-BY-NC-ND 4.0", True,
        max_batch=32, notes="image encoder only, 448 px (native config image_size), OpenAI mean/std"), _build_conch),
    (_S("conchv15", "CONCH v1.5", 768, 448, 448, IMAGENET_MEAN, IMAGENET_STD, "MahmoodLab/TITAN (return_conch)", "CC-BY-NC-ND 4.0", True,
        max_batch=32, notes="448 px; ImageNet mean/std  # UNVERIFIED (read from TITAN eval_transform printed at load)"), _build_conchv15),
    (_S("virchow2", "Virchow2", 2560, 224, 224, IMAGENET_MEAN, IMAGENET_STD, "hf-hub:paige-ai/Virchow2", "CC-BY-NC-ND 4.0", True,
        notes="CLS + mean(patch tokens[5:]); resize 224 bicubic"), _build_virchow2),
    (_S("hoptimus0", "H-optimus-0", 1536, 224, 224, HOPTIMUS_MEAN, HOPTIMUS_STD, "hf-hub:bioptimus/H-optimus-0", "Apache-2.0", False,
        notes="init_values=1e-5, dynamic_img_size=False"), _timm_plain("hf-hub:bioptimus/H-optimus-0", init_values=1e-5, dynamic_img_size=False)),
    (_S("hoptimus1", "H-optimus-1", 1536, 224, 224, HOPTIMUS_MEAN, HOPTIMUS_STD, "hf-hub:bioptimus/H-optimus-1", "CC-BY-NC-ND 4.0", True,
        notes="same recipe as H-optimus-0  # UNVERIFIED licence/gating"), _timm_plain("hf-hub:bioptimus/H-optimus-1", init_values=1e-5, dynamic_img_size=False)),
    (_S("h0mini", "H0-mini", 768, 224, 224, HOPTIMUS_MEAN, HOPTIMUS_STD, "hf-hub:bioptimus/H0-mini", "CC-BY-NC-ND 4.0", True,
        notes="CLS token only (plism-benchmark)  # UNVERIFIED vs model card; see h0mini_cm"), _build_h0mini(False)),
    (_S("h0mini_cm", "H0-mini (CLS+mean)", 1536, 224, 224, HOPTIMUS_MEAN, HOPTIMUS_STD, "hf-hub:bioptimus/H0-mini", "CC-BY-NC-ND 4.0", True,
        notes="optional variant: concat CLS + mean patch tokens  # UNVERIFIED register-token count"), _build_h0mini(True)),
    (_S("phikon2", "Phikon-v2", 1024, 224, 224, IMAGENET_MEAN, IMAGENET_STD, "owkin/phikon-v2 (transformers)", "Owkin non-commercial (gated)", True,
        notes="CLS of last_hidden_state; mean/std overwritten from AutoImageProcessor at load  # UNVERIFIED defaults"), _build_phikon2),
    (_S("gigapath", "Prov-GigaPath (tile)", 1536, 256, 224, IMAGENET_MEAN, IMAGENET_STD, "hf_hub:prov-gigapath/prov-gigapath", "Apache-2.0 (gated)", True,
        notes="Resize 256 bicubic + CenterCrop 224 (GitHub README, fetched)"), _timm_plain("hf_hub:prov-gigapath/prov-gigapath")),
    (_S("vitb16_in", "ViT-B/16 (ImageNet-21k->1k)", 768, 224, 224, (0.5,) * 3, (0.5,) * 3, "timm vit_base_patch16_224.augreg_in21k_ft_in1k", "Apache-2.0", False,
        notes="baseline; mean/std re-resolved from timm at load"), _build_baseline("vit_base_patch16_224.augreg_in21k_ft_in1k")),
    (_S("dinov2_l", "DINOv2 ViT-L/14", 1024, 224, 224, IMAGENET_MEAN, IMAGENET_STD, "timm vit_large_patch14_dinov2.lvd142m", "Apache-2.0 (code) / CC-BY-NC 4.0 (weights)  # UNVERIFIED", False,
        notes="baseline; run at 224 px (img_size=224 pos-embed interpolation)"), _build_baseline("vit_large_patch14_dinov2.lvd142m", img_size=224)),
    (_S("debug", "debug random projection", 64, 32, 32, (0.5,) * 3, (0.25,) * 3, "none", "n/a", False, notes="tests only"), None),
]
REGISTRY: dict[str, tuple[EncoderSpec, Callable | None]] = {s.key: (s, b) for s, b in _SPECS}


def list_models() -> list[str]:
    return list(REGISTRY)


def get_encoder(key: str) -> Encoder:
    """Instantiate an (unloaded) encoder.  Call ``.load()`` before ``.encode()``."""
    if key not in REGISTRY:
        raise KeyError(f"unknown model {key!r}; choose from {list_models()}")
    spec, builder = REGISTRY[key]
    return DebugEncoder(spec) if key == "debug" else Encoder(spec, builder)


if __name__ == "__main__":
    print(f"{'key':11s} {'dim':>5s} {'resize/crop':>11s}  gated  licence / source")
    for k, (s, _) in REGISTRY.items():
        print(f"{k:11s} {s.dim:5d} {s.resize:>5d}/{s.crop:<5d}  {str(s.gated):5s}  {s.license} | {s.source}")
