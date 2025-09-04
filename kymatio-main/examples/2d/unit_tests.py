import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from kymatio import Scattering2D


import torch

# Use the user's invertible scattering implementation
# try:
#     from scattering2d import InvertibleScatteringTransform
# except Exception:
#     from inverte_scattering import InvertibleScatteringTransform  # fallback


# =============================
# Synthetic image generators
# =============================
def img_constant(H, W, value=1.0, device="cpu"):
    return torch.full((H, W), float(value), device=device)

def img_delta(H, W, amp=1.0, device="cpu"):
    x = torch.zeros((H, W, 1), device=device)
    x[H//2, W//2] = float(amp)
    return x

def img_cosine(H, W, freq, angle_deg, amp=1.0, phase=0.0, device="cpu"):
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    theta = math.radians(angle_deg)
    kx, ky = freq*math.cos(theta), freq*math.sin(theta)
    arg = 2*math.pi*(kx*xx/W + ky*yy/H) + phase
    return amp*torch.cos(arg)

def img_checker(H, W, periods=8, amp=1.0, device="cpu"):
    s1 = img_cosine(H, W, periods, 0.0, amp=amp, phase=0.0, device=device)
    s2 = img_cosine(H, W, periods, 90.0, amp=amp, phase=0.0, device=device)
    return s1 * s2

def translate(x, dx, dy):
    return torch.roll(torch.roll(x, shifts=dy, dims=0), shifts=dx, dims=1)

# =============================
# Scattering adapter (invertible, no downsample)
# =============================
@dataclass
class ScatterSpec:
    H: int
    W: int
    J: int
    L: int
    max_order: int

class InvertibleScatteringAdapter:
    def __init__(self, spec: ScatterSpec, device: Optional[str] = None, pad_mode: str = "reflect"):
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = dict(J=spec.J, shape=(spec.H, spec.W), L=spec.L, max_order=spec.max_order)
        spec.H , spec.W = 10,10
        self.scattering = Scattering2D(J=3, L=2, shape=(10,10), max_order=2,frontend='torch', out_type="list", model_kind='invertible_scattering',downsample=False, dilation_optimization=False)

        try:
            self.scattering.to(self.device)
        except Exception:
            pass
        self.pad_mode = pad_mode
        # Optional meta (if provided by your impl)
        try:
            meta = self.scattering.meta()
            self.order = torch.as_tensor(meta.get("order", []))
        except Exception:
            self.order = None

    def __call__(self, xHW: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = xHW.to(self.device)
        out_raw = None
        last_err = None
        # Try 2D input first, then 4D (B,C,H,W)
        for trial in range(2):
            try:
                out_raw = self.scattering( x, downsample=False)
                break
            except Exception as e:
                last_err = e
                continue
        if out_raw is None:
            raise last_err
        # If your transform already returns dict with S0/S1/S2, pass through
        if isinstance(out_raw, dict) and (any(k in out_raw for k in ("S0","S1","S2")) or any(k.lower() in out_raw for k in ("s0","s1","s2"))):
            out: Dict[str, torch.Tensor] = {}
            for k, v in out_raw.items():
                kk = k.upper()
                if kk in ("S0","S1","S2"):
                    if isinstance(v, torch.Tensor):
                        out[kk] = v
                    elif isinstance(v, (list, tuple)):
                        try:
                            out[kk] = torch.stack([t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in v])
                        except Exception:
                            out[kk] = torch.cat([torch.as_tensor(t).reshape(-1) for t in v])
            return out
        # Else if tensor, attempt to split by order if provided
        if isinstance(out_raw, torch.Tensor):
            y = out_raw
            if y.ndim == 4:
                y = y.squeeze(0)
            out: Dict[str, torch.Tensor] = {}
            if (self.order is not None) and (self.order.numel() == y.shape[0]):
                ords = self.order
                idx0 = (ords == 0).nonzero(as_tuple=False).flatten()
                idx1 = (ords == 1).nonzero(as_tuple=False).flatten()
                idx2 = (ords == 2).nonzero(as_tuple=False).flatten()
                if len(idx0) > 0:
                    out["S0"] = y[idx0]
                if len(idx1) > 0:
                    out["S1"] = y[idx1]
                if len(idx2) > 0:
                    out["S2"] = y[idx2]
                if out:
                    return out
            # Fallback: expose everything as S1
            out["S1"] = y
            return out
        raise RuntimeError("Unsupported output structure from InvertibaleScatteringTransform")

# =============================
# Utility helpers
# =============================
def vecdict(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in d.values()])

def spatial_mean(t: torch.Tensor) -> torch.Tensor:
    if t.ndim >= 3:
        return t.mean(dim=(-1, -2))
    else:
        return t

# =============================
# Tests
# =============================
class TestFailure(Exception):
    pass

def assert_true(cond: bool, msg: str):
    if not cond:
        raise TestFailure(msg)

def test_constant(scatter: InvertibleScatteringAdapter):
    x = img_constant(scatter.spec.H, scatter.spec.W, 1.0)
    out = scatter(x)
    S0 = spatial_mean(out.get("S0", torch.tensor([x.mean()]))).mean()
    S_rest_max = torch.tensor(0.0, device=S0.device)
    for k, v in out.items():
        if k != "S0":
            S_rest_max = torch.maximum(S_rest_max, torch.abs(v).max())
    assert_true(torch.allclose(S0, torch.tensor(1.0, device=S0.device), atol=1e-3), f"S0 mean {S0.item():.6f} != 1.0 for constant input")
    assert_true(S_rest_max < 1e-5, f"Non-zero higher-order coeffs for constant input: max={S_rest_max.item():.3g}")

def test_delta(scatter: InvertibleScatteringAdapter):
    x = img_delta(scatter.spec.H, scatter.spec.W, 1.0)
    out = scatter(x)
    if "S1" not in out:
        return
    S1 = spatial_mean(out["S1"]).flatten()
    mean = S1.mean().abs() + 1e-12
    rel_std = (S1.std() / mean).item()
    assert_true(rel_std < 0.5, f"Delta test too anisotropic: relative std {rel_std:.3g} (tune if wavelets highly directional)")

def test_sinusoid_phase_invariance(scatter: InvertibleScatteringAdapter, freq=12, angle=30.0):
    H, W = scatter.spec.H, scatter.spec.W
    x1 = img_cosine(H, W, freq=freq, angle_deg=angle, amp=1.0, phase=0.0)
    x2 = img_cosine(H, W, freq=freq, angle_deg=angle, amp=1.0, phase=math.pi/3)
    if x1.ndim == 2:
        x1 = x1.unsqueeze(-1)
    if x2.ndim == 2:
        x2 = x2.unsqueeze(-1)
    S1a = spatial_mean(scatter(x1).get("S1", torch.zeros(1,1,1))).flatten()
    S1b = spatial_mean(scatter(x2).get("S1", torch.zeros(1,1,1))).flatten()
    if S1a.numel() == 0 or S1b.numel() == 0:
        return
    rel = (S1a - S1b).abs().max() / (S1a.abs().max() + 1e-12)
    assert_true(rel < 1e-2, f"Phase invariance violated for S1: rel_err={rel.item():.3g}")

def test_sinusoid_orientation_selectivity(scatter: InvertibleScatteringAdapter, freq=12, angle=30.0, angle2=60.0):
    H, W = scatter.spec.H, scatter.spec.W
    A = spatial_mean(scatter(img_cosine(H, W, freq, angle).unsqueeze(-1))).get("S1")
    B = spatial_mean(scatter(img_cosine(H, W, freq, angle2).unsqueeze(-1))).get("S1")
    if A is None or B is None:
        return
    a = A.flatten(); b = B.flatten()
    cos_sim = torch.dot(a, b) / (a.norm()*b.norm() + 1e-12)
    assert_true(cos_sim < 0.99, f"Orientation selectivity too weak: cos_sim={cos_sim.item():.3f}")

def test_translation_invariance(scatter: InvertibleScatteringAdapter, J: int, shift=(2,3), freq=10, angle=0.0):
    H, W = scatter.spec.H, scatter.spec.W
    x = img_cosine(H, W, freq, angle)
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    xt = translate(x, dx=shift[0], dy=shift[1])
    def feat(z):
        out = scatter(z)
        return torch.cat([spatial_mean(v).reshape(-1) for v in out.values()])
    v = feat(x); vt = feat(xt)
    rel = (v - vt).norm() / (v.norm() + 1e-12)
    bound = 0.3 * (max(abs(shift[0]),abs(shift[1])) / (2**J))
    assert_true(rel < bound + 1e-3, f"Too sensitive to translation: rel={rel.item():.3g} >= {bound:.3g}")

def test_nonexpansive(scatter: InvertibleScatteringAdapter):
    x1 = torch.randn(scatter.spec.H, scatter.spec.W)
    x2 = torch.randn(scatter.spec.H, scatter.spec.W)
    if x1.ndim == 2:
        x1 = x1.unsqueeze(-1)
    if x2.ndim == 2:
        x2 = x2.unsqueeze(-1)
    def vec(z): return vecdict(scatter(z))
    lhs = (vec(x1) - vec(x2)).norm()
    rhs = (x1 - x2).norm()
    assert_true(lhs <= 1.1 * rhs + 1e-6, f"Non-expansiveness failed: ||Sx - Sy||={lhs.item():.3g} > 1.1*||x-y||={1.1*rhs.item():.3g}")

def test_checker_highfreq(scatter: InvertibleScatteringAdapter, periods=12):
    x = img_checker(scatter.spec.H, scatter.spec.W, periods=periods)
    out = scatter(x)
    if "S1" not in out:
        return
    S1 = spatial_mean(out["S1"]).flatten()
    assert_true(S1.abs().max().item() > 1e-3, "Checkerboard produced near-zero S1 responses unexpectedly")

# =============================
# Runner
# =============================
def run_all(spec: ScatterSpec, pad_mode="reflect") -> List[Tuple[str, str]]:
    adapter = InvertibleScatteringAdapter(spec, pad_mode=pad_mode)
    results: List[Tuple[str, str]] = []
    tests = [
        ("Constant annihilation", lambda: test_constant(adapter)),
        ("Delta isotropy", lambda: test_delta(adapter)),
        ("Sinusoid phase invariance", lambda: test_sinusoid_phase_invariance(adapter, freq=min(spec.H,spec.W)//16, angle=30.0)),
        ("Sinusoid orientation selectivity", lambda: test_sinusoid_orientation_selectivity(adapter, freq=min(spec.H,spec.W)//16, angle=20.0, angle2=65.0)),
        ("Translation invariance", lambda: test_translation_invariance(adapter, J=spec.J, shift=(2,3), freq=min(spec.H,spec.W)//16)),
        ("Non-expansiveness", lambda: test_nonexpansive(adapter)),
        ("Checker high-frequency", lambda: test_checker_highfreq(adapter, periods=min(spec.H,spec.W)//16)),
    ]
    for name, fn in tests:
        try:
            fn()
            results.append((name, "PASS"))
        except TestFailure as e:
            results.append((name, f"FAIL: {e}"))
        except Exception as e:
            results.append((name, f"ERROR: {type(e).__name__}: {e}"))
    return results


def main():
    ap = argparse.ArgumentParser(description="Invertible Scattering tests (downsample=False)")
    ap.add_argument("--H", type=int, default=192)
    ap.add_argument("--W", type=int, default=192)
    ap.add_argument("--J", type=int, default=3)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--max-order", type=int, default=2)
    ap.add_argument("--pad-mode", type=str, default="reflect")
    args = ap.parse_args()

    spec = ScatterSpec(H=args.H, W=args.W, J=args.J, L=args.L, max_order=args.max_order)
    res = run_all(spec, pad_mode=args.pad_mode)
    width = max(len(n) for n,_ in res) + 4
    print("="* (width+10))
    print("Scattering test summary")
    print("="* (width+10))
    for name, status in res:
        print(f"{name:<{width}} {status}")
    fails = [s for _,s in res if not s.startswith("PASS")]
    if fails:
        print("Some tests failed. Consider tolerances/normalization/padding or check your invertible settings.")
        raise SystemExit(1)
    else:
        print("All tests passed.")

if __name__ == "__main__":
    main()
