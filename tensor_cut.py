# -*- coding: utf-8 -*-
"""
Tensor-cut: پیاده‌سازی کامل روش «گراف‌کات مبتنی بر تانسور» برای بخش‌بندی عروق
مرجع: C. Wang et al., "Tensor-cut: A Tensor-based Graph-cut Blood Vessel
Segmentation Method and Its Application to Renal Artery Segmentation",
Medical Image Analysis, 2019.

مراحل (مطابق Fig.1 مقاله و فصل ۴ پایان‌نامه):
  1. فیلتر چندمقیاسه‌ی Hessian (Sato/Frangi) -> بهترین مقیاس و تانسور T^H یا T^D   (Eq.3-5)
  2. متر شبه‌ریمانی بین تانسورها، میانگین و واریانس فرشه (Eq.7 و پیوست)
  3. تولید خودکار گره‌های s/t با K-means روی خروجی vesselness                        (Eq.8-9)
  4. GMM برای شدت روشنایی (اقلیدسی) و برای تانسورها (ریمانی)                        (Eq.10)
  5. تابع انرژی MRF مرتبه‌ی اول و حل با min-cut/max-flow                            (Eq.11-12)
  6. معیارهای ارزیابی Se, Spc, CO, CE                                               (Eq.13-15)

وابستگی‌ها: numpy, scipy, scikit-image, scikit-learn  و (اختیاری، سریع‌تر) PyMaxflow
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

try:  # PyMaxflow سریع‌ترین گزینه است: pip install PyMaxflow
    import maxflow as _maxflow
    HAVE_MAXFLOW = True
except Exception:  # pragma: no cover
    HAVE_MAXFLOW = False


# =============================================================================
# پارامترها
# =============================================================================
@dataclass
class TensorCutParams:
    # --- فیلتر Hessian (بخش 2.2) ---
    scales: tuple = (0.7, 1.0, 1.5, 2.0, 2.5)   # بر حسب میلی‌متر (σ گاوسی)
    alpha_v: float = 0.25                        # α در Eq.4
    gamma12: float = 1.0                         # γ12
    gamma23: float = 1.0                         # γ23
    bright_vessels: bool = True                  # در CT با کنتراست، رگ روشن است
    # --- تانسور قطری T^D (Eq.5): λ3 >> λ2 ≈ λ1 > 0 (روی تانسورِ نرمال‌شده) ---
    td_eigs: tuple = (0.02, 0.02, 1.0)
    # --- متر ریمانی ---
    eps_spd: float = 1e-2        # حداقل مقدار ویژه برای SPD کردن (پایداری عددی)
    lam_floor: float = 1e-2      # کف قدرمطلق مقدار ویژه‌ی تعمیم‌یافته قبل از log
    # --- گره‌های ترمینال (Eq.8-9) ---
    kmeans_k: int = 5            # K
    bg_dilate_radius: int = 3    # r
    # --- GMM ---
    kg: int = 3                  # تعداد مؤلفه‌ها Kg
    em_iters: int = 8
    max_fit_samples: int = 20000
    # --- انرژی (Eq.11): سه پارامتر اصلی ---
    lambda_i: float = 1.0
    lambda_t: float = 1.0
    omega: float = 1.0
    connectivity: int = 6        # MRF مرتبه‌ی اول: 6 همسایه
    seed_mode: str = "both"     # قید سخت روی گره‌های اولیه: both (s و t) | fg (فقط s) | none (نرم)
    seed: int = 0


# =============================================================================
# 1) میدان تانسوری (بخش 2.2)
# =============================================================================
def _hessian_at_scale(vol: np.ndarray, sigma_mm: float, spacing) -> np.ndarray:
    """ماتریس هسین نرمال‌شده با σ² در مقیاس s. خروجی: (6, *shape) به ترتیب xx,xy,xz,yy,yz,zz"""
    sig = [sigma_mm / s for s in spacing]                    # σ بر حسب وکسل
    sc = sigma_mm ** 2
    out = []
    for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
        order = [0, 0, 0]
        order[i] += 1
        order[j] += 1
        # مشتق نسبت به مختصات فیزیکی: تقسیم بر spacing_i * spacing_j
        h = ndi.gaussian_filter(vol, sig, order=order, mode="nearest")
        out.append(sc * h / (spacing[i] * spacing[j]))
    return np.stack(out).astype(np.float32)


def _pack_hessian(h6: np.ndarray) -> np.ndarray:
    """(6,...) -> (...,3,3) متقارن"""
    shp = h6.shape[1:]
    H = np.empty(shp + (3, 3), dtype=np.float32)
    H[..., 0, 0] = h6[0]; H[..., 0, 1] = H[..., 1, 0] = h6[1]
    H[..., 0, 2] = H[..., 2, 0] = h6[2]; H[..., 1, 1] = h6[3]
    H[..., 1, 2] = H[..., 2, 1] = h6[4]; H[..., 2, 2] = h6[5]
    return H


def sato_vesselness(lam: np.ndarray, alpha=0.25, g12=1.0, g23=1.0) -> np.ndarray:
    """Eq.4. lam: (...,3) مرتب‌شده‌ی نزولی λ1>=λ2>=λ3 (λ3 منفی‌ترین). رگ روشن."""
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]
    V = np.zeros(l1.shape, dtype=np.float32)
    tiny = 1e-12
    valid = (l3 < l2) & (l2 < 0) & (l3 < 0)
    ratio = np.where(valid, l2 / np.where(l3 == 0, -tiny, l3), 0.0)   # λ2/λ3 >0
    base = np.abs(l3) * np.power(np.clip(ratio, 0, None), g23)
    c1 = valid & (l1 <= 0)
    c2 = valid & (l1 > 0) & (l1 < np.abs(l2) / alpha)
    t1 = np.power(np.clip(1 + l1 / np.maximum(np.abs(l2), tiny), 0, None), g12)
    t2 = np.power(np.clip(1 - alpha * l1 / np.maximum(np.abs(l2), tiny), 0, None), g12)
    V = np.where(c1, base * t1, V)
    V = np.where(c2, base * t2, V)
    return V.astype(np.float32)


def build_tensor_field(vol, spacing, p: TensorCutParams, mask=None, chunk=400_000):
    """
    فیلتر چندمقیاسه‌ی Hessian -> V_max و s' برای هر وکسل -> میدان تانسوری T (Eq.5).
    خروجی:  T (N,3,3) برای وکسل‌های mask ، vmax (حجمی) ، idx (اندیس‌های خطی mask)
    """
    vol = vol.astype(np.float32)
    if not p.bright_vessels:
        vol = -vol
    shape = vol.shape
    if mask is None:
        mask = np.ones(shape, bool)
    idx = np.flatnonzero(mask.ravel())
    n = idx.size
    vmax = np.zeros(n, np.float32)
    Hbest = np.zeros((n, 3, 3), np.float32)

    for s in p.scales:
        h6 = _hessian_at_scale(vol, s, spacing).reshape(6, -1)[:, idx]
        for a in range(0, n, chunk):
            sl = slice(a, min(a + chunk, n))
            hh = h6[:, sl]
            Hm = np.empty((hh.shape[1], 3, 3), np.float32)
            Hm[:, 0, 0] = hh[0]; Hm[:, 0, 1] = Hm[:, 1, 0] = hh[1]
            Hm[:, 0, 2] = Hm[:, 2, 0] = hh[2]; Hm[:, 1, 1] = hh[3]
            Hm[:, 1, 2] = Hm[:, 2, 1] = hh[4]; Hm[:, 2, 2] = hh[5]
            lam = np.linalg.eigvalsh(Hm)[:, ::-1]          # نزولی: λ1>=λ2>=λ3
            V = sato_vesselness(lam, p.alpha_v, p.gamma12, p.gamma23)
            better = V > vmax[sl]
            vmax[sl] = np.where(better, V, vmax[sl])
            view = Hbest[sl]                                # برش پایه = view
            view[better] = Hm[better]                       # s' = argmax_s V_s
    # نرمال‌سازی سراسری تا T^D با ثابت‌های td_eigs هم‌مقیاس شود
    tub = vmax > 0
    if tub.any():
        nrm = np.linalg.norm(Hbest[tub], axis=(1, 2))
        scale = np.percentile(nrm, 95) + 1e-12
    else:
        scale = 1.0
    Hbest /= scale
    T = Hbest.copy()
    TD = np.diag(np.asarray(p.td_eigs, np.float32))
    T[~tub] = TD                                           # Eq.5
    vfull = np.zeros(shape, np.float32).ravel()
    vfull[idx] = vmax
    return T, vfull.reshape(shape), idx


# =============================================================================
# 2) هندسه‌ی ریمانی روی تانسورها (بخش 2.3 + پیوست)
# =============================================================================
def spd_abs(T: np.ndarray, eps: float) -> np.ndarray:
    """Eq.6: T+ = U diag(|d_i|) U^T  (به‌علاوه‌ی کف eps)  -> SPD"""
    w, U = np.linalg.eigh(T.astype(np.float64))
    w = np.maximum(np.abs(w), eps)
    return np.einsum("...ij,...j,...kj->...ik", U, w, U)


def _logm_spd(S):
    w, U = np.linalg.eigh(S)
    return np.einsum("...ij,...j,...kj->...ik", U, np.log(w), U)


def _expm_sym(S):
    w, U = np.linalg.eigh(S)
    return np.einsum("...ij,...j,...kj->...ik", U, np.exp(w), U)


def _sqrtm_spd(S, inv=False):
    w, U = np.linalg.eigh(S)
    e = -0.5 if inv else 0.5
    return np.einsum("...ij,...j,...kj->...ik", U, w ** e, U)


def semi_riemannian_dist2(A: np.ndarray, B_spd: np.ndarray, lam_floor=1e-2) -> np.ndarray:
    """
    Eq.7:  d^2 = sum_i log^2 |λ_i(A,B)|  ، λ_i مقادیر ویژه‌ی تعمیم‌یافته‌ی A v = λ B v.
    A می‌تواند نامعین (هسین خام) باشد؛ B باید SPD باشد (برای پایداری).
    """
    M = np.linalg.solve(B_spd, A.astype(np.float64))         # B^{-1} A
    ev = np.linalg.eigvals(M)
    lam = np.maximum(np.abs(ev), lam_floor)
    return np.sum(np.log(lam) ** 2, axis=-1)


def frechet_mean(Ts_spd: np.ndarray, weights=None, eps=1e-3, max_iter=30, init=None):
    """
    Algorithm 2 (گرادیان‌کاهشی روی منیفلد): میانگین هندسی/فرشه.
    Ts_spd: (N,3,3) SPD ؛ weights: (N,) وزن‌های نرمال‌شده (برای EM).
    """
    N = Ts_spd.shape[0]
    w = np.full(N, 1.0 / N) if weights is None else weights / (weights.sum() + 1e-300)
    M = Ts_spd[0].copy() if init is None else init.copy()
    for _ in range(max_iter):
        Mh = _sqrtm_spd(M)
        Mih = _sqrtm_spd(M, inv=True)
        Y = _logm_spd(Mih @ Ts_spd @ Mih)                      # log_M(T_i) در فضای مماس (منتقل‌شده)
        X = np.tensordot(w, Y, axes=(0, 0))                    # گرادیان
        M = Mh @ _expm_sym(X) @ Mh                             # exp_M(X)
        if np.linalg.norm(X) < eps:
            break
    return M


def frechet_variance(Ts_raw, mean, weights=None, lam_floor=1e-2):
    """Eq.20: σ² = E[d²(T̄, T)]"""
    d2 = semi_riemannian_dist2(Ts_raw, mean, lam_floor)
    w = np.full(d2.shape, 1.0 / d2.size) if weights is None else weights / (weights.sum() + 1e-300)
    return float(np.sum(w * d2)), d2


# =============================================================================
# 3) مدل آمیخته‌ی گاوسی روی تانسورها (Eq.10 ، سمت تانسور)
# =============================================================================
class RiemannianGMM:
    """
    GMM روی منیفلد: هر مؤلفه N(T | T̄_k, σ_k²) ∝ exp(-d²/(2σ_k²)) / (2πσ_k²)^{D/2} ، D=6.
    وزن‌ها با EM؛ M-step با میانگین فرشه‌ی وزن‌دار.
    """
    D = 6  # بعد Sym(3)

    def __init__(self, k=3, iters=8, eps=1e-2, lam_floor=1e-2, seed=0):
        self.k, self.iters, self.eps, self.lam_floor, self.seed = k, iters, eps, lam_floor, seed

    def _logpdf(self, T_raw):
        out = np.empty((T_raw.shape[0], self.k))
        for j in range(self.k):
            d2 = semi_riemannian_dist2(T_raw, self.means[j], self.lam_floor)
            s2 = self.var[j]
            out[:, j] = -d2 / (2 * s2) - 0.5 * self.D * np.log(2 * np.pi * s2)
        return out

    def fit(self, T_raw, T_spd):
        n = T_raw.shape[0]
        k = min(self.k, max(1, n // 10))
        self.k = k
        # مقداردهی اولیه: K-means روی لگاریتم ماتریس (log-Euclidean)
        feat = _logm_spd(T_spd)[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
        lab = KMeans(k, n_init=3, random_state=self.seed).fit_predict(feat) if k > 1 else np.zeros(n, int)
        R = np.eye(k)[lab] + 1e-6
        R /= R.sum(1, keepdims=True)
        self.means = [None] * k
        self.var = np.ones(k)
        self.pi = np.ones(k) / k
        for it in range(self.iters + 1):
            # M-step
            nk = R.sum(0) + 1e-12
            self.pi = nk / nk.sum()
            for j in range(k):
                self.means[j] = frechet_mean(T_spd, R[:, j], max_iter=10,
                                             init=self.means[j])
                v, _ = frechet_variance(T_raw, self.means[j], R[:, j], self.lam_floor)
                self.var[j] = max(v, 1e-3)
            if it == self.iters:
                break
            # E-step
            lp = self._logpdf(T_raw) + np.log(self.pi)[None]
            lp -= lp.max(1, keepdims=True)
            R = np.exp(lp)
            R /= R.sum(1, keepdims=True)
        return self

    def logpdf(self, T_raw, chunk=100_000):
        res = np.empty(T_raw.shape[0])
        for a in range(0, T_raw.shape[0], chunk):
            lp = self._logpdf(T_raw[a:a + chunk]) + np.log(self.pi)[None]
            m = lp.max(1)
            res[a:a + chunk] = m + np.log(np.exp(lp - m[:, None]).sum(1))
        return res


# =============================================================================
# 4) گره‌های ترمینال با K-means (Eq.8-9)
# =============================================================================
def terminal_nodes(vesselness: np.ndarray, mask: np.ndarray, p: TensorCutParams):
    """s-nodes: خوشه‌ای با بیشترین میانگین |I_V| ؛ t-nodes: مکمل s-nodes منبسط‌شده با شعاع r."""
    vals = np.abs(vesselness[mask]).reshape(-1, 1)
    km = KMeans(p.kmeans_k, n_init=3, random_state=p.seed).fit(vals)
    means = [np.mean(np.abs(vals[km.labels_ == c])) for c in range(p.kmeans_k)]
    cmax = int(np.argmax(means))
    fg = np.zeros(mask.shape, bool)
    fg[mask] = km.labels_ == cmax
    fg &= vesselness > 0
    struct = ndi.generate_binary_structure(3, 1)
    dil = ndi.binary_dilation(fg, structure=struct, iterations=p.bg_dilate_radius)
    bg = (~dil) & mask
    return fg, bg


# =============================================================================
# 5) گراف‌کات
# =============================================================================
def _estimate_xi(d2_values: np.ndarray) -> float:
    m = float(np.mean(d2_values)) + 1e-12
    return 1.0 / (2.0 * m)          # ξ از «سطح نویز» (مانند GrabCut)


def _solve_mincut(shape, unary_fg, unary_bg, edges, hard_fg, hard_bg):
    """
    unary_fg/bg: هزینه‌ی برچسب F / B ؛ edges: لیست (idx_a, idx_b, w) روی اندیس‌های خطی.
    خروجی: ماسک بولی (True=فورگراند).
    """
    N = int(np.prod(shape))
    BIG = 1e9
    cs = unary_bg.copy()            # ظرفیت source: هزینه‌ی افتادن در سمت sink (=برچسب B)
    ct = unary_fg.copy()            # ظرفیت sink: هزینه‌ی برچسب F
    cs[hard_fg.ravel()] = BIG; ct[hard_fg.ravel()] = 0
    ct[hard_bg.ravel()] = BIG;  cs[hard_bg.ravel()] = 0

    if HAVE_MAXFLOW:
        g = _maxflow.GraphFloat()
        nid = g.add_grid_nodes(shape)
        g.add_grid_tedges(nid, cs.reshape(shape), ct.reshape(shape))
        for (a, b, w) in edges:
            try:                                            # نسخه‌ی برداری (PyMaxflow >= 1.2.12)
                g.add_edges(a.astype(np.int64), b.astype(np.int64),
                            w.astype(np.float64), w.astype(np.float64))
            except AttributeError:                          # نسخه‌ی قدیمی: حلقه
                for ai, bi, wi in zip(a, b, w):
                    g.add_edge(int(ai), int(bi), float(wi), float(wi))
        g.maxflow()
        return ~g.get_grid_segments(nid)                    # سمت source = فورگراند

    # --- جایگزین بدون PyMaxflow: scipy.sparse.csgraph.maximum_flow ---
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order
    S, Tn = N, N + 1
    scale = 1e4 / max(1e-12, np.percentile(np.concatenate([cs[cs < BIG], ct[ct < BIG], 
                                      np.concatenate([e[2] for e in edges])]), 99))
    cap = lambda x: np.minimum(np.round(x * scale), 2**30).astype(np.int32)
    rows, cols, data = [], [], []
    ia = np.arange(N)
    rows += [np.full(N, S), ia]; cols += [ia, np.full(N, Tn)]; data += [cap(cs), cap(ct)]
    for (a, b, w) in edges:
        c = cap(w)
        rows += [a, b]; cols += [b, a]; data += [c, c]
    A = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N + 2, N + 2)).tocsr()
    A.sum_duplicates()
    res = maximum_flow(A, S, Tn)
    resid = (A - res.flow).tocsr()
    resid.data[resid.data < 0] = 0
    resid.eliminate_zeros()
    reach = breadth_first_order(resid, S, directed=True, return_predecessors=False)
    seg = np.zeros(N + 2, bool); seg[reach] = True
    return seg[:N].reshape(shape)


# =============================================================================
# 6) الگوریتم اصلی Tensor-cut (Algorithm 1)
# =============================================================================
def tensor_cut(vol, spacing=(1.0, 1.0, 1.0), mask=None, params: TensorCutParams | None = None,
               use_tensor_term=True, verbose=True):
    """
    vol: حجم CT (z,y,x یا هر ترتیب سه‌بعدی؛ spacing باید با همان ترتیب باشد)
    mask: ماسک ناحیه‌ی مورد نظر (کبد/کلیه) ؛ اگر None کل حجم
    خروجی: ماسک بخش‌بندی عروق (bool) و دیکشنری اطلاعات میانی
    use_tensor_term=False  ->  GC ساده (ω=0)
    """
    p = params or TensorCutParams()
    log = (lambda *a: print("[tensor-cut]", *a)) if verbose else (lambda *a: None)
    rng = np.random.default_rng(p.seed)
    vol = vol.astype(np.float32)
    if mask is None:
        mask = np.ones(vol.shape, bool)

    # کراپ به جعبه‌ی محیطی ماسک (صرفه‌جویی حافظه)
    sl = ndi.find_objects(mask.astype(np.uint8))[0]
    vol_c, mask_c = vol[sl], mask[sl]
    shape = vol_c.shape
    # نرمال‌سازی شدت به [0,1] برای GMM و ξ
    lo, hi = np.percentile(vol_c[mask_c], [0.5, 99.5])
    I = np.clip((vol_c - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)

    # --- مرحله ۱: میدان تانسوری ---
    log("building tensor field ...")
    T, vesselness, idx = build_tensor_field(vol_c, spacing, p, mask_c)
    T_spd = spd_abs(T, p.eps_spd)

    # --- مرحله ۲: گره‌های s/t ---
    log("terminal nodes (K-means) ...")
    fg, bg = terminal_nodes(vesselness, mask_c, p)
    log(f"  |s-nodes|={fg.sum()}  |t-nodes|={bg.sum()}")
    if fg.sum() < 20:
        raise RuntimeError("ناحیه‌ی فورگراند اولیه خیلی کوچک است؛ مقیاس‌ها/مقداردهی را بررسی کنید.")

    flat_pos = -np.ones(mask_c.size, np.int64)
    flat_pos[idx] = np.arange(idx.size)          # نگاشت اندیس خطی -> اندیس در آرایه‌ی تانسور
    fg_i = flat_pos[np.flatnonzero(fg.ravel())]
    bg_i = flat_pos[np.flatnonzero(bg.ravel())]

    def sub(ix):
        return ix[rng.permutation(ix.size)[:p.max_fit_samples]]

    # --- مرحله ۳: GMMها ---
    log("fitting GMMs ...")
    gi_f = GaussianMixture(p.kg, random_state=p.seed).fit(I.ravel()[idx[sub(fg_i)]].reshape(-1, 1))
    gi_b = GaussianMixture(p.kg, random_state=p.seed).fit(I.ravel()[idx[sub(bg_i)]].reshape(-1, 1))
    Ii = I.ravel()[idx].reshape(-1, 1)
    lf_I, lb_I = gi_f.score_samples(Ii), gi_b.score_samples(Ii)

    if use_tensor_term:
        sf, sb = sub(fg_i), sub(bg_i)
        gt_f = RiemannianGMM(p.kg, p.em_iters, p.eps_spd, p.lam_floor, p.seed).fit(T[sf], T_spd[sf])
        gt_b = RiemannianGMM(p.kg, p.em_iters, p.eps_spd, p.lam_floor, p.seed).fit(T[sb], T_spd[sb])
        lf_T, lb_T = gt_f.logpdf(T), gt_b.logpdf(T)
    else:
        lf_T = lb_T = np.zeros_like(lf_I)

    # احتمال پسین با قانون بیز و پیشین برابر:  -log Pr(L|.) = -(l_L - logsumexp(l_F,l_B))
    def neg_log_post(lF, lB):
        m = np.maximum(lF, lB)
        lse = m + np.log(np.exp(lF - m) + np.exp(lB - m))
        return -(lF - lse), -(lB - lse)

    dF_I, dB_I = neg_log_post(lf_I, lb_I)
    dF_T, dB_T = neg_log_post(lf_T, lb_T)
    w_t = p.omega if use_tensor_term else 0.0
    unary_fg = np.zeros(mask_c.size); unary_bg = np.zeros(mask_c.size)
    unary_fg[idx] = dF_I + w_t * dF_T
    unary_bg[idx] = dB_I + w_t * dB_T
    outside = ~mask_c.ravel()
    unary_fg[outside] = 1e6; unary_bg[outside] = 0        # بیرون ماسک = پس‌زمینه

    # --- مرحله ۴: پیوندهای pairwise (Eq.12) ---
    log("pairwise terms ...")
    Iflat = I.ravel()
    raw_edges = []                   # (a, b, dI², dT², dist)
    offs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if p.connectivity == 26:
        offs = [o for o in np.ndindex(3, 3, 3)]
        offs = [(a - 1, b - 1, c - 1) for a, b, c in offs if (a - 1, b - 1, c - 1) > (0, 0, 0)]
    mflat = mask_c.ravel()
    ids = np.arange(mask_c.size).reshape(shape)
    for (dz, dy, dx) in offs:
        s1 = tuple(slice(0 if o >= 0 else -o, n - o if o > 0 else n) for o, n in zip((dz, dy, dx), shape))
        s2 = tuple(slice(o if o > 0 else 0, n if o >= 0 else n + o) for o, n in zip((dz, dy, dx), shape))
        a, b = ids[s1].ravel(), ids[s2].ravel()
        ok = mflat[a] & mflat[b]
        a, b = a[ok], b[ok]
        dist = float(np.sqrt(sum((o * s) ** 2 for o, s in zip((dz, dy, dx), spacing))))
        dI2 = (Iflat[a] - Iflat[b]) ** 2
        if use_tensor_term:
            pa, pb = flat_pos[a], flat_pos[b]
            dT2 = np.empty(a.size)
            for c0 in range(0, a.size, 200_000):
                c1 = c0 + 200_000
                dT2[c0:c1] = semi_riemannian_dist2(T[pa[c0:c1]], T_spd[pb[c0:c1]], p.lam_floor)
        else:
            dT2 = np.zeros(a.size)
        raw_edges.append((a, b, dI2, dT2, dist))

    xi_I = _estimate_xi(np.concatenate([e[2] for e in raw_edges]))
    xi_T = _estimate_xi(np.concatenate([e[3] for e in raw_edges])) if use_tensor_term else 0.0
    edges = []
    for (a, b, dI2, dT2, dist) in raw_edges:
        V = np.exp(-xi_I * dI2) / dist
        w = p.lambda_i * V
        if use_tensor_term:
            U = np.exp(-xi_T * dT2) / dist
            w = w + p.omega * p.lambda_t * U
        edges.append((a, b, w))

    # --- مرحله ۵: min-cut/max-flow ---
    log("max-flow ...", "(PyMaxflow)" if HAVE_MAXFLOW else "(scipy fallback)")
    hard_fg = fg if p.seed_mode in ("both", "fg") else np.zeros(shape, bool)
    hard_bg = bg if p.seed_mode == "both" else np.zeros(shape, bool)
    seg_c = _solve_mincut(shape, unary_fg, unary_bg, edges, hard_fg, hard_bg)
    seg_c &= mask_c

    out = np.zeros(vol.shape, bool)
    out[sl] = seg_c
    info = dict(vesselness=vesselness, fg_seed=fg, bg_seed=bg, crop=sl, xi_I=xi_I, xi_T=xi_T,
                idx=idx, logodds_I=(dB_I - dF_I), logodds_T=((dB_T - dF_T) if use_tensor_term else None))
    return out, info


# =============================================================================
# 7) معیارهای ارزیابی (Eq.13-15)
# =============================================================================
def sensitivity_specificity(seg, gt):
    seg, gt = seg.astype(bool), gt.astype(bool)
    tp = np.sum(seg & gt); fn = np.sum(~seg & gt)
    tn = np.sum(~seg & ~gt); fp = np.sum(seg & ~gt)
    return tp / (tp + fn + 1e-12), tn / (tn + fp + 1e-12)


def largest_component(m):
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def centerline(m):
    from skimage.morphology import skeletonize
    return skeletonize(m.astype(bool))


def centerline_metrics(seg, gt, spacing=(1, 1, 1), dilate_vox=2, largest_only=True):
    """CO (Eq.14) و CE (Eq.15). مقاله فقط بزرگ‌ترین مؤلفه‌ی همبند را می‌سنجد (largest_only=True)؛
    با largest_only=False همه‌ی مؤلفه‌ها حساب می‌شوند (برای خروجی‌های تکه‌تکه مفیدتر است)."""
    seg = seg.astype(bool); gt = gt.astype(bool)
    if largest_only:
        seg = largest_component(seg); gt = largest_component(gt)
    cg, cs = centerline(gt), centerline(seg)
    if cg.sum() == 0 or cs.sum() == 0:
        return 0.0, np.inf
    U = ndi.binary_dilation(cg, structure=ndi.generate_binary_structure(3, 1), iterations=dilate_vox)
    overlap = np.sum(cs & U)                         # O_Ω
    CO = 2.0 * overlap / (cg.sum() + cs.sum())
    sp = np.asarray(spacing, float)
    pg, ps = np.argwhere(cg) * sp, np.argwhere(cs) * sp
    h_gs = cKDTree(ps).query(pg)[0].mean()
    h_sg = cKDTree(pg).query(ps)[0].mean()
    return float(CO), float((h_gs + h_sg) / 2)


# =============================================================================
# 8) فانتوم عروقی ساده (جایگزین VascuSynth برای آزمون سریع)
# =============================================================================
def make_phantom(shape=(64, 64, 64), n_levels=4, noise_sigma=0.3, smooth_mm=1.0,
                 spacing=(1.0, 1.0, 1.0), seed=0):
    """
    درخت دوشاخه‌ای تصادفی؛ شعاع‌ها با قانون مورِی (r_child = r_parent / 2^(1/3)).
    خروجی: حجم نویزی، ground-truth باینری.
    """
    rng = np.random.default_rng(seed)
    gt = np.zeros(shape, bool)
    zz, yy, xx = np.indices(shape, sparse=True)

    def draw(p0, p1, r):
        lo = np.floor(np.minimum(p0, p1) - r - 2).astype(int).clip(0)
        hi = np.ceil(np.maximum(p0, p1) + r + 3).astype(int)
        hi = np.minimum(hi, shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        g = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1)
        d = p1 - p0
        t = np.clip(((g - p0) @ d) / (d @ d + 1e-12), 0, 1)
        dist = np.linalg.norm(g - (p0 + t[..., None] * d), axis=-1)
        gt[sl] |= dist <= r

    def grow(p0, direction, r, level, length):
        p1 = p0 + direction * length
        if np.any(p1 < 2) or np.any(p1 > np.array(shape) - 3):
            return
        draw(p0, p1, r)
        if level >= n_levels:
            return
        for _ in range(2):
            dnew = direction + rng.normal(0, 0.6, 3)
            dnew /= np.linalg.norm(dnew)
            grow(p1, dnew, max(r / 2 ** (1 / 3), 0.9), level + 1, length * 0.8)

    root = np.array([shape[0] * 0.1, shape[1] * 0.5, shape[2] * 0.5])
    grow(root, np.array([1.0, 0, 0]), 3.0, 0, shape[0] * 0.3)
    img = gt.astype(np.float32)
    img = ndi.gaussian_filter(img, smooth_mm / np.asarray(spacing))
    img = img + rng.normal(0, noise_sigma * img.max() * 0.35, shape).astype(np.float32)
    return img, gt


# =============================================================================
# 9) ROC / پارامتر جستجو (grid search مطابق بخش 3.2)
# =============================================================================
def grid_search(vol, gt, spacing, mask=None, lambdas_i=(0.5, 1, 2), lambdas_t=(0.5, 1, 2),
                omegas=(0.5, 1, 2), base: TensorCutParams | None = None, verbose=False):
    base = base or TensorCutParams()
    rows = []
    for li in lambdas_i:
        for lt in lambdas_t:
            for om in omegas:
                q = TensorCutParams(**{**base.__dict__, "lambda_i": li, "lambda_t": lt, "omega": om})
                seg, _ = tensor_cut(vol, spacing, mask, q, verbose=verbose)
                se, sp = sensitivity_specificity(seg, gt)
                co, ce = centerline_metrics(seg, gt, spacing)
                rows.append(dict(lambda_i=li, lambda_t=lt, omega=om, Se=se, Spc=sp, CO=co, CE=ce))
    return rows
