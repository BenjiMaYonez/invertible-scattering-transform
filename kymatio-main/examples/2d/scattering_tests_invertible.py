
# import torch
# import numpy as np
# import argparse

# from kymatio import Scattering2D


# def run_tests(H, W, J, L, max_order):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     scatter = Scattering2D(
#         J=J,
#         shape=(H, W),
#         L=L,
#         max_order=max_order,
#         downsample=False,
#         frontend='torch',
#         model_kind='invertible_scattering',
#         dilation_optimization=False,
#         out_type="list",
#     ).to(device)

#     def _stack_list_output(items):
#         """Convert list of dicts [{'coef':..., 'depth':d, ...}] into
#         a dict with keys 'S0','S1','S2' and tensors stacked along a channel dim.
#         Uses |coef| (magnitude) to be robust to complex outputs.
#         """
#         by_order = {0: [], 1: [], 2: []}
#         for it in items:
#             coef = it.get('coef', it.get('coeff', it.get('value')))
#             if not isinstance(coef, torch.Tensor):
#                 coef = torch.as_tensor(coef)
#             if torch.is_complex(coef):
#                 coef = coef.abs()
#             order = it.get('depth', None)
#             if order is None:
#                 j_tuple = it.get('j', ())
#                 try:
#                     order = len(j_tuple)
#                 except Exception:
#                     order = 1
#             order = int(order)
#             if order > 2:
#                 order = 2
#             by_order[order].append(coef)
#         out = {}
#         if by_order[0]:
#             out['S0'] = torch.stack([c if c.ndim >= 2 else c.reshape(1, 1) for c in by_order[0]], dim=0)
#         if by_order[1]:
#             out['S1'] = torch.stack([c if c.ndim >= 2 else c.reshape(1, 1) for c in by_order[1]], dim=0)
#         if by_order[2]:
#             out['S2'] = torch.stack([c if c.ndim >= 2 else c.reshape(1, 1) for c in by_order[2]], dim=0)
#         return out

#     def S(x):
#         out = scatter(x.to(device))
#         if isinstance(out, list):
#             return _stack_list_output(out)
#         if isinstance(out, dict):
#             norm = {}
#             for k, v in out.items():
#                 t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
#                 if torch.is_complex(t):
#                     t = t.abs()
#                 norm[k.upper()] = t
#             return norm
#         if isinstance(out, torch.Tensor):
#             t = out.abs() if torch.is_complex(out) else out
#             return {"S1": t}
#         raise RuntimeError("Unknown scattering output type")

#     def order_norms(coeffs):
#         """Return per-order (S0,S1,S2) norms if present, else 0."""
#         n0 = coeffs.get('S0')
#         n1 = coeffs.get('S1')
#         n2 = coeffs.get('S2')
#         n0 = n0.norm().item() if isinstance(n0, torch.Tensor) else 0.0
#         n1 = n1.norm().item() if isinstance(n1, torch.Tensor) else 0.0
#         n2 = n2.norm().item() if isinstance(n2, torch.Tensor) else 0.0
#         return n0, n1, n2
    
#     def _unsplit_get_Re_Im(t: torch.Tensor):
#         """
#         Given a tensor that may carry the 4 ReLU splits, return signed (Re, Im).
#         If a dim of size 4 exists, we treat it as [r+, r-, i+, i-] along that dim.
#         Otherwise: complex -> (real, imag), real -> (t, 0).
#         """
#         dims_of_4 = [d for d, sz in enumerate(t.shape) if sz == 4]
#         if dims_of_4:
#             ax = dims_of_4[0]
#             tt = t.movedim(ax, 0)  # [4, ...]
#             r_pos, r_neg, i_pos, i_neg = tt[0], tt[1], tt[2], tt[3]
#             Re = r_pos - r_neg
#             Im = i_pos - i_neg
#             return Re, Im
#         if torch.is_complex(t):
#             return t.real, t.imag
#         else:
#             return t, torch.zeros_like(t)


#     def magnitude_L2_from_splits(t: torch.Tensor) -> torch.Tensor:
#         """Return phase-invariant |z| from the four splits (or from complex t)."""
#         Re, Im = _unsplit_get_Re_Im(t)
#         return torch.sqrt(Re**2 + Im**2 + 1e-20)

#     def magnitude_L1_from_splits(t: torch.Tensor) -> torch.Tensor:
#         """Return |Re z| + |Im z| from the four splits (or from complex t)."""
#         Re, Im = _unsplit_get_Re_Im(t)
#         return Re.abs() + Im.abs()

#     def split_energy_profile(t: torch.Tensor) -> torch.Tensor:
#         """
#         Return a 4-vector with total energy per split: [sum r+, sum r-, sum i+, sum i-].
#         If no 4-way split is present, returns a length-4 zero vector.
#         """
#         dims_of_4 = [d for d, sz in enumerate(t.shape) if sz == 4]
#         if not dims_of_4:
#             return torch.zeros(4, dtype=t.dtype, device=t.device)
#         ax = dims_of_4[0]
#         tt = t.movedim(ax, 0)  # [4, ...]
#         # sum over all remaining dims
#         return torch.stack([tt[i].abs().sum() for i in range(4)], dim=0)


#     tol_small = 1e-6
#     tol_mean = 1e-3
#     results = []

#     # 1) Constant image (DC) — S0 ≈ 1, S1/S2 ≈ 0
#     const = torch.ones(H, W)
#     C = S(const)
#     s0_mean = C['S0'].mean().item() if 'S0' in C else 0.0
#     n0, n1, n2 = order_norms(C)
#     const_checks = (
#         abs(s0_mean - 1.0) < tol_mean,     # S0 ~ mean(const) = 1
#         n1 < 1e-5,                         # S1 ~ 0
#         n2 < 1e-5,                         # S2 ~ 0 (if present)
#     )
#     results.append(("Constant: S0≈1", const_checks[0]))
#     results.append(("Constant: S1≈0", const_checks[1]))
#     results.append(("Constant: S2≈0", const_checks[2]))

#     # 2) Impulse (delta) — orientation near-isotropy across channels (loose)
#     delta = torch.zeros(H, W)
#     delta[H // 2, W // 2] = 1.0
#     D = S(delta)
#     if 'S1' in D:
#         S1 = D['S1'].view(D['S1'].size(0), -1)  # [C, M]
#         means = S1.mean(dim=1)                  # per-channel mean magnitude
#         rel_std = (means.std() / (means.abs().mean() + 1e-12)).item()
#         results.append(("Delta: S1 orientation isotropy (rel std < 0.5)", rel_std < 0.5))
#     else:
#         results.append(("Delta: S1 present", False))

#     # 3) Sinusoid (phase invariance) — phase shift should not change magnitudes
#     f = max(2, W // 16)
#     x = torch.arange(W).float()
#     sinusoid  = torch.cos(2 * np.pi * f * x / W).repeat(H, 1)
#     sin_shift = torch.cos(2 * np.pi * f * x / W + np.pi / 2).repeat(H, 1)

#     S_sin   = S(sinusoid)
#     S_shift = S(sin_shift)

#     if 'S1' in S_sin and 'S1' in S_shift:
#         # Recombine the 4 splits to a phase-invariant magnitude before comparing
#         A = magnitude_L2_from_splits(S_sin['S1'])
#         B = magnitude_L2_from_splits(S_shift['S1'])
#         # (Optional) you can spatially average first; here we compare full fields:
#         diff_max = (A - B).abs().max().item()
#         ref_max  = A.abs().max().item() + 1e-12
#         rel_phase = diff_max / ref_max
#         results.append(("Sinusoid: phase invariance (rel max diff < 1e-2)", rel_phase < 1e-2))
#         results.append((f"rel_phase: {rel_phase:.3f}", True))
#     else:
#         results.append(("Sinusoid: S1 present", False))


#    # 4) Sinusoid (orientation selectivity) — horizontal vs vertical differ
#     # Horizontal grating: varies along x (k = (f, 0))
#     # Vertical   grating: varies along y (k = (0, f))
#     sinusoid_h = sinusoid              # from earlier (cos(2π f x / W) repeated across rows)
#     sinusoid_v = torch.cos(2 * np.pi * f * x[:, None] / H).repeat(1, W)

#     S_h = S(sinusoid_h)
#     S_v = S(sinusoid_v)

#     ok_orient = False
#     ok_splits = True  # default to True if no split axis is found

#     if 'S1' in S_h and 'S1' in S_v:
#         # (A) Phase-invariant orientation selectivity:
#         # Recombine the four splits back into |z| and compare horizontal vs vertical.
#         A = magnitude_L2_from_splits(S_h['S1']).flatten()
#         B = magnitude_L2_from_splits(S_v['S1']).flatten()
#         cos_sim = float((A @ B) / (A.norm() * B.norm() + 1e-12))
#         ok_orient = (cos_sim < 0.99)

#         # (B) Split-profile difference (only if a split axis of size 4 exists):
#         prof_h = split_energy_profile(S_h['S1'])
#         prof_v = split_energy_profile(S_v['S1'])
#         if prof_h.sum() > 0 and prof_v.sum() > 0:
#             cos_sim_splits = float((prof_h @ prof_v) / (prof_h.norm() * prof_v.norm() + 1e-12))
#             ok_splits = (cos_sim_splits < 0.99)

#         results.append(("Orientation selectivity (pooled |z|, cos sim < 0.99)", ok_orient))
#         results.append(("Orientation selectivity (split-energy profiles differ)", ok_splits))
#     else:
#         results.append(("Orientation selectivity: S1 present", False))


#     # 5) Translation invariance — shift ≤ 2^J -> small change after averaging
#     shift_px = max(1, 2 ** J // 2)
#     shifted = torch.roll(sinusoid, shifts=shift_px, dims=1)
#     S_orig = S(sinusoid)
#     S_shft = S(shifted)
#     if 'S1' in S_orig and 'S1' in S_shft:
#         v = S_orig['S1'].mean(dim=(-1, -2)).flatten()  # average spatially
#         vt = S_shft['S1'].mean(dim=(-1, -2)).flatten()
#         rel = (v - vt).norm() / (v.norm() + 1e-12)
#         bound = 0.3 * (shift_px / max(1, 2 ** J))
#         results.append((f"Translation invariance (rel diff < {bound:.3f})", rel.item() < bound + 1e-3))
#     else:
#         results.append(("Translation invariance: S1 present", False))

#     # 6) Non-expansiveness — ||S(x)-S(y)|| ≤ c * ||x-y||
#     x1 = torch.randn(H, W)
#     x2 = x1 + 1e-3 * torch.randn_like(x1)
#     Sx1 = S(x1)
#     Sx2 = S(x2)
#     # vectorize dicts (S0,S1,S2) -> one long vector
#     def vec(coeffs):
#         parts = []
#         for k in ('S0', 'S1', 'S2'):
#             if k in coeffs and isinstance(coeffs[k], torch.Tensor):
#                 parts.append(coeffs[k].reshape(-1))
#         return torch.cat(parts) if parts else torch.zeros(1)
#     lhs = (vec(Sx1) - vec(Sx2)).norm().item()
#     rhs = (x1 - x2).norm().item()
#     results.append(("Non-expansiveness (||Sx-Sy|| ≤ 1.1||x-y||)", lhs <= 1.1 * rhs + 1e-9))

#     # 7) Checkerboard — should elicit non-trivial high-frequency S1
#     checker = torch.tensor([[(i + j) % 2 for j in range(W)] for i in range(H)], dtype=torch.float32)
#     S_checker = S(checker)
#     if 'S1' in S_checker:
#         s1max = S_checker['S1'].abs().max().item()
#         results.append(("Checkerboard: S1 non-trivial (max > 1e-3)", s1max > 1e-3))
#     else:
#         results.append(("Checkerboard: S1 present", False))

#     # Print summary
#     print("\n==== Scattering Transform Unit Tests ====")
#     all_passed = True
#     for name, ok in results:
#         print(f"{name:45s}: {'OK' if ok else 'FAIL'}")
#         all_passed &= ok
#     return all_passed


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--H", type=int, default=64)
#     parser.add_argument("--W", type=int, default=64)
#     parser.add_argument("--J", type=int, default=2)
#     parser.add_argument("--L", type=int, default=4)
#     parser.add_argument("--max-order", type=int, default=2)
#     args = parser.parse_args()

#     success = run_tests(args.H, args.W, args.J, args.L, args.max_order)
#     exit(0 if success else 1)
import argparse
import numpy as np
import torch
from collections import defaultdict
from typing import Any, Tuple

from kymatio import Scattering2D


# ---------------------------
# Utilities for output parsing
# ---------------------------

def _canon_tuple(x: Any) -> Tuple:
    """
    Convert possibly None / scalar / iterable to a sortable tuple.
    Ensures stable ordering across runs.
    """
    if x is None:
        return ()
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)

def _key_for_group(d):
    """
    Identify a wavelet path WITHOUT the split tag, so we can collect the 4 splits
    that belong together. We use a robust key from available metadata.
    Expected fields per item: 'depth', 'j', 'n', 'theta', 'path', 'split'.
    If 'path' encodes split as a trailing element, we drop it.
    """
    depth = d.get('depth', None)
    j = d.get('j', None)
    n = d.get('n', None)
    theta = d.get('theta', None)
    p = d.get('path', ())
    if isinstance(p, tuple) and len(p) >= 1:
        path_wo_split = p[:-1]
    else:
        path_wo_split = p
    return (depth, _canon_tuple(j), n, _canon_tuple(theta), _canon_tuple(path_wo_split))

def _split_name(d):
    # e.g., 're_pos', 're_neg', 'im_pos', 'im_neg', or 'no_split'
    return d.get('split', 'no_split')

def _sort_items_stably(items):
    """
    Return items sorted by a canonical per-item key so channel ordering is stable
    across different inputs (e.g., across a phase shift).
    """
    def item_key(d):
        return _key_for_group(d) + (_split_name(d),)
    return sorted(items, key=item_key)


def _stack_list_output_with_meta(items):
    """
    Convert list of dicts [{'coef':..., 'depth':d, 'split':..., 'path':..., ...}] into:
      - out['S0'], out['S1'], out['S2'] : raw stacked tensors by order (C x H x W), with
        channels ordered deterministically across calls;
      - out['S1_magL1'] : per-(depth=1, j, theta, ...) L1 magnitude, grouping the 4 splits.
    """
    # Bucket and sort by order
    by_order = defaultdict(list)
    for it in items:
        depth = int(it.get('depth', 0))
        by_order[depth].append(it)

    out = {}

    # ---- Zeroth order (no split) ----
    if 0 in by_order:
        sorted_0 = _sort_items_stably(by_order[0])
        S0 = [torch.as_tensor(d['coef']) for d in sorted_0]
        out['S0'] = torch.stack([t if t.ndim >= 2 else t.reshape(1, 1) for t in S0], dim=0)

    # ---- First order ----
    if 1 in by_order:
        sorted_1 = _sort_items_stably(by_order[1])

        # Raw stacked (each split item is a channel) — with deterministic order
        S1_raw = [torch.as_tensor(d['coef']) for d in sorted_1]
        out['S1'] = torch.stack([t if t.ndim >= 2 else t.reshape(1, 1) for t in S1_raw], dim=0)

        # Group the four splits that belong to the same (j, theta, ...) path
        groups = defaultdict(dict)  # key -> {split_name: tensor}
        for d in sorted_1:  # sorted insert guarantees deterministic grouping traversal too
            key = _key_for_group(d)
            sname = _split_name(d)
            groups[key][sname] = torch.as_tensor(d['coef'])

        # Recombine by L1: |Re z| + |Im z| = re_pos + re_neg + im_pos + im_neg
        keys_sorted = sorted(groups.keys())  # <--- deterministic group order
        S1_magL1 = []
        for key in keys_sorted:
            parts = groups[key]
            re_pos = torch.as_tensor(parts.get('re_pos', 0.0))
            re_neg = torch.as_tensor(parts.get('re_neg', 0.0))
            im_pos = torch.as_tensor(parts.get('im_pos', 0.0))
            im_neg = torch.as_tensor(parts.get('im_neg', 0.0))
            magL1 = re_pos + re_neg + im_pos + im_neg
            S1_magL1.append(magL1 if magL1.ndim >= 2 else magL1.reshape(1, 1))
        if S1_magL1:
            out['S1_magL1'] = torch.stack(S1_magL1, dim=0)

    # ---- Second order (raw, deterministically ordered) ----
    if 2 in by_order:
        sorted_2 = _sort_items_stably(by_order[2])
        S2_raw = [torch.as_tensor(d['coef']) for d in sorted_2]
        out['S2'] = torch.stack([t if t.ndim >= 2 else t.reshape(1, 1) for t in S2_raw], dim=0)

    return out


def S_wrapper(scatter, device, x):
    """Forward pass returning a normalized dict (handles list/dict/tensor outputs)."""
    out = scatter(x.to(device))
    if isinstance(out, list):
        return _stack_list_output_with_meta(out)
    if isinstance(out, dict):
        norm = {}
        for k, v in out.items():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if torch.is_complex(t):
                t = t.abs()
            norm[k.upper()] = t
        return norm
    if isinstance(out, torch.Tensor):
        t = out.abs() if torch.is_complex(out) else out
        return {"S1": t}
    raise RuntimeError("Unknown scattering output type")


def order_norms(coeffs):
    """Return per-order (S0,S1,S2) norms if present, else 0."""
    n0 = coeffs.get('S0')
    n1 = coeffs.get('S1')
    n2 = coeffs.get('S2')
    n0 = n0.norm().item() if isinstance(n0, torch.Tensor) else 0.0
    n1 = n1.norm().item() if isinstance(n1, torch.Tensor) else 0.0
    n2 = n2.norm().item() if isinstance(n2, torch.Tensor) else 0.0
    return n0, n1, n2


def split_energy_profile(t: torch.Tensor) -> torch.Tensor:
    """
    Return a 4-vector with total energy per split: [sum r+, sum r-, sum i+, sum i-].
    If no 4-way split axis is present, returns zeros.
    We detect a dimension of size 4 as the split axis.
    """
    dims_of_4 = [d for d, sz in enumerate(t.shape) if sz == 4]
    if not dims_of_4:
        return torch.zeros(4, dtype=t.dtype, device=t.device)
    ax = dims_of_4[0]
    tt = t.movedim(ax, 0)  # [4, ...]
    return torch.stack([tt[i].abs().sum() for i in range(4)], dim=0)


# ---------------------------
# Individual test functions
# ---------------------------

def test_constant(S, H, W, J, L, max_order, tol_mean=1e-3):
    """Constant image: S0≈1; S1≈0; S2≈0."""
    const = torch.ones(H, W)
    C = S(const)
    s0_mean = C['S0'].mean().item() if 'S0' in C else 0.0
    n0, n1, n2 = order_norms(C)
    return [
        ("Constant: S0≈1", abs(s0_mean - 1.0) < tol_mean),
        ("Constant: S1≈0", n1 < 1e-5),
        ("Constant: S2≈0", n2 < 1e-5),
    ]


def test_delta_isotropy(S, H, W, J, L, max_order):
    """Impulse: first-order channels (per orientation at a scale) should be roughly isotropic."""
    delta = torch.zeros(H, W)
    delta[H // 2, W // 2] = 1.0
    D = S(delta)
    if 'S1' not in D:
        return [("Delta: S1 present", False)]
    S1 = D['S1'].view(D['S1'].size(0), -1)  # [C, M]
    means = S1.mean(dim=1)
    rel_std = (means.std() / (means.abs().mean() + 1e-12)).item()
    return [("Delta: S1 orientation isotropy (rel std < 0.5)", rel_std < 0.5)]


def test_phase_invariance(S, H, W, J, L, max_order):
    """
    Sinusoid: a phase shift should not change phase-invariant magnitudes.
    Compare spatially averaged per-path S1_magL1 (deterministic path ordering).
    """
    f = max(2, W // 16)
    x = torch.arange(W).float()
    sinusoid = torch.cos(2 * np.pi * f * x / W).repeat(H, 1)
    sin_shift = torch.cos(2 * np.pi * f * x / W + np.pi / 2).repeat(H, 1)

    A = S(sinusoid)
    B = S(sin_shift)

    # Prefer pooled magnitude if present; else fall back to S1 (less robust)
    if 'S1_magL1' in A and 'S1_magL1' in B:
        a = A['S1_magL1'].mean(dim=(-1, -2)).flatten()
        b = B['S1_magL1'].mean(dim=(-1, -2)).flatten()
    elif 'S1' in A and 'S1' in B:
        a = A['S1'].mean(dim=(-1, -2)).flatten()
        b = B['S1'].mean(dim=(-1, -2)).flatten()
    else:
        return [("Sinusoid: S1 present", False)]

    rel = (a - b).norm() / (a.norm() + 1e-12)
    return [
        ("Sinusoid: phase invariance (L1 pooled, rel < 1e-2)", float(rel) < 1e-2),
        (f"rel_phase: {float(rel):.3f}", True),
    ]


def test_orientation_selectivity(S, H, W, J, L, max_order):
    """
    Sinusoid (orientation selectivity): horizontal vs vertical gratings should differ.
    Check both pooled magnitude (phase-invariant) and split-profile difference.
    """
    f = max(2, W // 16)
    x = torch.arange(W).float()
    sinusoid_h = torch.cos(2 * np.pi * f * x / W).repeat(H, 1)           # varies along x
    sinusoid_v = torch.cos(2 * np.pi * f * x[:, None] / H).repeat(1, W)  # varies along y

    Sh = S(sinusoid_h)
    Sv = S(sinusoid_v)

    results = []
    if 'S1' in Sh and 'S1' in Sv:
        # (A) Phase-invariant orientation selectivity with pooled magnitude
        if 'S1_magL1' in Sh and 'S1_magL1' in Sv:
            Ah = Sh['S1_magL1'].flatten()
            Bv = Sv['S1_magL1'].flatten()
        else:
            Ah = Sh['S1'].flatten()
            Bv = Sv['S1'].flatten()
        cos_sim = float((Ah @ Bv) / (Ah.norm() * Bv.norm() + 1e-12))
        results.append(("Orientation selectivity (pooled |z|, cos sim < 0.99)", cos_sim < 0.99))

        # (B) Split-energy profile difference (if split axis exists)
        prof_h = split_energy_profile(Sh['S1'])
        prof_v = split_energy_profile(Sv['S1'])
        if prof_h.sum() > 0 and prof_v.sum() > 0:
            cos_sim_splits = float((prof_h @ prof_v) / (prof_h.norm() * prof_v.norm() + 1e-12))
            results.append(("Orientation selectivity (split-energy profiles differ)", cos_sim_splits < 0.99))
        else:
            results.append(("Orientation selectivity (split-energy profiles differ)", True))
    else:
        results.append(("Orientation selectivity: S1 present", False))
    return results


def test_translation_invariance(S, H, W, J, L, max_order):
    """Translation stability: shifts <=~ 2^J should cause small relative change after averaging."""
    f = max(2, W // 16)
    x = torch.arange(W).float()
    sinusoid = torch.cos(2 * np.pi * f * x / W).repeat(H, 1)

    shift_px = max(1, 2 ** J // 2)
    shifted = torch.roll(sinusoid, shifts=shift_px, dims=1)
    S_orig = S(sinusoid)
    S_shft = S(shifted)
    if 'S1' in S_orig and 'S1' in S_shft:
        v = S_orig['S1'].mean(dim=(-1, -2)).flatten()
        vt = S_shft['S1'].mean(dim=(-1, -2)).flatten()
        rel = (v - vt).norm() / (v.norm() + 1e-12)
        bound = 0.3 * (shift_px / max(1, 2 ** J))
        return [(f"Translation invariance (rel diff < {bound:.3f})", rel.item() < bound + 1e-3)]
    else:
        return [("Translation invariance: S1 present", False)]


def test_nonexpansive(S, H, W, J, L, max_order):
    """Non-expansiveness: ||Sx - Sy|| ≤ 1.1 ||x - y|| (tolerance for numerics)."""
    x1 = torch.randn(H, W)
    x2 = x1 + 1e-3 * torch.randn_like(x1)
    Sx1 = S(x1)
    Sx2 = S(x2)

    def vec(coeffs):
        parts = []
        for k in ('S0', 'S1_magL1', 'S1', 'S2'):
            if k in coeffs and isinstance(coeffs[k], torch.Tensor):
                parts.append(coeffs[k].reshape(-1))
        return torch.cat(parts) if parts else torch.zeros(1)

    lhs = (vec(Sx1) - vec(Sx2)).norm().item()
    rhs = (x1 - x2).norm().item()
    return [("Non-expansiveness (||Sx-Sy|| ≤ 1.1||x-y||)", lhs <= 1.1 * rhs + 1e-9)]


def test_checker(S, H, W, J, L, max_order):
    """Checkerboard (high-frequency): should elicit non-trivial first-order response."""
    checker = torch.tensor([[(i + j) % 2 for j in range(W)] for i in range(H)], dtype=torch.float32)
    S_checker = S(checker)
    if 'S1' in S_checker:
        s1max = S_checker['S1'].abs().max().item()
        return [("Checkerboard: S1 non-trivial (max > 1e-3)", s1max > 1e-3)]
    else:
        return [("Checkerboard: S1 present", False)]


# ---------------------------
# Runner
# ---------------------------

def run_tests(H, W, J, L, max_order):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scatter = Scattering2D(
        J=J, shape=(H, W), L=L, max_order=max_order,
        downsample=False, frontend='torch',
        model_kind='invertible_scattering',
        dilation_optimization=False,
        out_type="list",
    ).to(device)

    def S(x):
        return S_wrapper(scatter, device, x)

    results = []
    for name, fn in [
        ("Constant", test_constant),
        ("Delta isotropy", test_delta_isotropy),
        ("Sinusoid phase invariance", test_phase_invariance),
        ("Orientation selectivity", test_orientation_selectivity),
        ("Translation invariance", test_translation_invariance),
        ("Non-expansiveness", test_nonexpansive),
        ("Checkerboard", test_checker),
    ]:
        try:
            res = fn(S, H, W, J, L, max_order)
            results.extend(res)
        except Exception as e:
            results.append((f"{name} (exception)", False))
            results.append((f"  {type(e).__name__}: {e}", False))

    print("\n==== Scattering Transform Unit Tests ====")
    all_passed = True
    width = max(len(n) for n, _ in results) + 4
    for name, ok in results:
        print(f"{name:<{width}} {'OK' if ok else 'FAIL'}")
        all_passed &= ok
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--W", type=int, default=64)
    parser.add_argument("--J", type=int, default=2)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--max-order", type=int, default=2)
    args = parser.parse_args()

    success = run_tests(args.H, args.W, args.J, args.L, args.max_order)
    raise SystemExit(0 if success else 1)


