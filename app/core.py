"""裁剪工具核心：比例转换 + 自动识别并去掉全身人物的头。全部本地计算，不联网。"""
import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes, gaussian_filter,
                           label, find_objects)
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg

# 打包成 exe 后，模型文件解压在 sys._MEIPASS 里
HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
LUMA = np.array([0.299, 0.587, 0.114])


# ---------------------------------------------------------------- 读写

def load_image(path):
    im = Image.open(path)
    info = {"dpi": im.info.get("dpi"), "icc_profile": im.info.get("icc_profile")}
    return np.asarray(im.convert("RGB")).copy(), info


def save_png(arr, path, info):
    kw = {}
    if info.get("dpi"):
        kw["dpi"] = info["dpi"]
    if info.get("icc_profile"):
        kw["icc_profile"] = info["icc_profile"]
    tmp = path + ".tmp.png"
    Image.fromarray(arr).save(tmp, **kw)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 比例转换

def parse_ratio(text):
    a, b = text.replace("：", ":").split(":")
    return float(a) / float(b)


def _edge_colors(arr, side, width=20, tol=22):
    """Per-row background colour along one edge (rows covered by clothes are interpolated)."""
    H, W, _ = arr.shape
    strip = arr[:, :width] if side == "left" else arr[:, W - width:]
    strip = strip.astype(np.float64)
    ref = np.median(strip.reshape(-1, 3), axis=0)
    good = np.abs(strip - ref).max(2) < tol
    cols = np.full((H, 3), np.nan)
    rows = good.any(1)
    for y in np.nonzero(rows)[0]:
        cols[y] = strip[y][good[y]].mean(0)
    idx = np.arange(H)
    for c in range(3):
        v = cols[:, c]
        ok = ~np.isnan(v)
        cols[:, c] = np.interp(idx, idx[ok], v[ok]) if ok.any() else ref[c]
    return gaussian_filter(cols, (40, 0))


def pad_to_ratio(arr, ratio, mode="black"):
    """Add bars so that width/height == ratio. Original pixels are copied untouched."""
    H, W, _ = arr.shape
    if abs(W / H - ratio) < 1e-4:
        return arr.copy(), (0, 0)
    if W / H < ratio:
        new_w, new_h = round(H * ratio), H
    else:
        new_w, new_h = W, round(W / ratio)
    left, top = (new_w - W) // 2, (new_h - H) // 2
    out = np.zeros((new_h, new_w, 3), np.uint8)
    if mode == "bg":
        if new_w > W:
            lc, rc = _edge_colors(arr, "left"), _edge_colors(arr, "right")
            out[:, :left] = np.clip(np.round(lc), 0, 255).astype(np.uint8)[:, None, :]
            out[:, left + W:] = np.clip(np.round(rc), 0, 255).astype(np.uint8)[:, None, :]
        else:
            tc = _edge_colors(arr.transpose(1, 0, 2), "left")
            bc = _edge_colors(arr.transpose(1, 0, 2), "right")
            out[:top] = np.clip(np.round(tc), 0, 255).astype(np.uint8)[None, :, :]
            out[top + H:] = np.clip(np.round(bc), 0, 255).astype(np.uint8)[None, :, :]
    out[top:top + H, left:left + W] = arr
    return out, (left, top)


# ---------------------------------------------------------------- 识别头部

def _face_detector(w, h):
    import cv2
    buf = np.frombuffer(open(MODEL, "rb").read(), np.uint8)  # 读进内存，绕开中文路径问题
    return cv2.FaceDetectorYN.create("onnx", buf, np.array([], np.uint8), (w, h), 0.75, 0.3, 200)


def detect_faces(img):
    import cv2
    H, W = img.shape[:2]
    found = []
    for target in (1600, 2400):
        s = min(2.0, target / max(W, H))
        small = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        det = _face_detector(bgr.shape[1], bgr.shape[0])
        _, faces = det.detect(bgr)
        if faces is None:
            continue
        for f in faces:
            v = f[:14] / s
            found.append(v)
    # merge detections of the same face from the two scales
    merged = []
    for v in found:
        cx, cy, w = v[0] + v[2] / 2, v[1] + v[3] / 2, v[2]
        for m in merged:
            if abs(m["cx"] - cx) < 0.5 * w and abs(m["cy"] - cy) < 0.5 * w:
                m["vals"].append(v)
                break
        else:
            merged.append({"cx": cx, "cy": cy, "vals": [v]})
    faces = []
    for m in merged:
        v = np.mean(m["vals"], axis=0)
        x, y, w, h = v[:4]
        eye_y = (v[5] + v[7]) / 2
        mouth_y = (v[11] + v[13]) / 2
        faces.append({"x": x, "y": y, "w": w, "h": h, "eye_y": eye_y, "mouth_y": mouth_y,
                      "cut": mouth_y + 0.8 * (mouth_y - eye_y)})
    return faces


def detect_heads(img, max_face_frac=0.12):
    """Return boxes (x0, y0, x1, cut_y) for the heads of full-body figures (small faces)."""
    H, W = img.shape[:2]
    faces = [f for f in detect_faces(img) if f["h"] < max_face_frac * H]
    # figures standing side by side in one sheet share the same cut height
    for f in faces:
        group = [g["cut"] for g in faces
                 if abs(g["eye_y"] - f["eye_y"]) < 0.03 * H and 0.7 < g["h"] / f["h"] < 1.4]
        f["cut_g"] = float(np.median(group))
    boxes = []
    for f in faces:
        cut = int(round(f["cut_g"]))
        cx, fw, fh = f["x"] + f["w"] / 2, f["w"], f["h"]
        r0, c0 = max(0, int(f["y"] - 1.5 * fh)), max(0, int(cx - 3 * fw))
        c1 = min(W, int(cx + 3 * fw))
        if cut - r0 < 8 or c1 - c0 < 8:
            continue
        k = 4
        sub = img[r0:cut:k, c0:c1:k].astype(np.float64)
        border = np.concatenate([sub[0], sub[:, 0], sub[:, -1]])
        bgc = np.median(border, axis=0)
        fg = np.abs(sub - bgc).max(2) > 18
        fg = binary_dilation(fg, iterations=2)
        lab, _ = label(fg)
        fy0, fy1 = int((f["y"] - r0) / k), int((f["y"] + fh - r0) / k)
        fx0, fx1 = int((f["x"] - c0) / k), int((f["x"] + fw - c0) / k)
        ids = np.unique(lab[max(0, fy0):fy1, max(0, fx0):fx1])
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue
        ys, xs = np.nonzero(np.isin(lab, ids))
        m = int(0.08 * fh) + 10
        x0 = max(0, c0 + xs.min() * k - m)
        x1 = min(W, c0 + xs.max() * k + m)
        y0 = max(0, r0 + ys.min() * k - m)
        boxes.append([int(x0), int(y0), int(x1), cut])
    boxes.sort(key=lambda b: b[0])
    return boxes


# ---------------------------------------------------------------- 填补算法

def _cg_solve(vals, U, ok, init, maxiter):
    """Harmonic fill of U: neighbours in `ok` are fixed values, other neighbours act as walls."""
    h, w = U.shape
    ys, xs = np.nonzero(U)
    n = len(ys)
    idx = -np.ones((h, w), np.int64)
    idx[ys, xs] = np.arange(n)
    diag = np.zeros(n)
    b = np.zeros((n, 3))
    rr, cc = [], []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        inb = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nyc, nxc = np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)
        nU = inb & U[nyc, nxc]
        nK = inb & ~U[nyc, nxc] & ok[nyc, nxc]
        diag += nU | nK
        rr.append(np.nonzero(nU)[0])
        cc.append(idx[nyc[nU], nxc[nU]])
        b[nK] += vals[nyc[nK], nxc[nK]]
    diag[diag == 0] = 1
    rr, cc = np.concatenate(rr), np.concatenate(cc)
    A = diags(diag) - csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n))
    M = diags(1.0 / diag)
    out = vals.copy()
    for c in range(3):
        x0 = init[ys, xs, c]
        sol, _ = cg(A, b[:, c], x0=x0, rtol=1e-6, maxiter=maxiter, M=M)
        out[ys, xs, c] = sol
    return out


def _harmonic(vals, U, ok, depth=0):
    """Coarse-to-fine harmonic fill (fast for large holes)."""
    h, w = U.shape
    n = int(U.sum())
    if n == 0:
        return vals
    if n > 6000 and min(h, w) > 32 and depth < 6:
        hp, wp = h + h % 2, w + w % 2
        def pad(a, v):
            return np.pad(a, ((0, hp - h), (0, wp - w)) + ((0, 0),) * (a.ndim - 2), constant_values=v)
        Up, okp, vp = pad(U, False), pad(ok & ~U, False), pad(vals, 0)
        U2 = Up.reshape(hp // 2, 2, wp // 2, 2).any((1, 3))
        cnt = okp.reshape(hp // 2, 2, wp // 2, 2).sum((1, 3))
        ok2 = (cnt > 0) & ~U2
        s = (vp * okp[..., None]).reshape(hp // 2, 2, wp // 2, 2, 3).sum((1, 3))
        v2 = s / np.maximum(cnt, 1)[..., None]
        sol2 = _harmonic(v2, U2, ok2, depth + 1)
        init = np.repeat(np.repeat(sol2, 2, 0), 2, 1)[:h, :w]
        init = gaussian_filter(init, (1, 1, 0))
        return _cg_solve(vals, U, ok, init, maxiter=150)
    ring = binary_dilation(U) & ~U & ok
    mean = vals[ring].mean(0) if ring.any() else vals[ok].mean(0) if ok.any() else np.full(3, 200.0)
    init = np.broadcast_to(mean, vals.shape).copy()
    return _cg_solve(vals, U, ok, init, maxiter=3000)


def _grain(shape, U, ref_vals, mono, seed):
    rng = np.random.default_rng(seed)
    if len(ref_vals) < 50:
        return np.zeros(shape)
    sd = 1.4826 * np.median(np.abs(ref_vals), axis=0)
    nz = rng.standard_normal(shape[:2] + ((1,) if mono else (3,)))
    nz = gaussian_filter(nz, (0.6, 0.6, 0))
    nz /= nz[U].std(0) + 1e-9
    return nz * sd


def fill_region(f, U, ok, mono=False, seed=1):
    """f: float image crop. Fill U smoothly from `ok` pixels and add matching grain."""
    out = _harmonic(f, U, ok)
    hp = f - gaussian_filter(f, (3, 3, 0))
    ring = binary_dilation(U, iterations=20) & ~U & ok
    out[U] += _grain(f.shape, U, hp[ring], mono, seed)[U]
    return out


# ---------------------------------------------------------------- 去头

def remove_heads(img, boxes, progress=None, debug=None):
    """img uint8 HxWx3; boxes [(x0, y0, x1, cut_y)]. Returns new image (only head areas change)."""
    out = img.copy()
    H, W = img.shape[:2]
    for bi, (x0, y0, x1, cut) in enumerate(boxes):
        if progress:
            progress(f"正在去头 {bi + 1}/{len(boxes)} …")
        x0, x1 = sorted((int(x0), int(x1)))
        y0, cut = sorted((int(y0), int(cut)))
        x0, x1, y0, cut = max(0, x0), min(W, x1), max(0, y0), min(H, cut)
        if x1 - x0 < 4 or cut - y0 < 4:
            continue
        head_h = cut - y0
        band = max(10, int(0.35 * head_h))
        m = 40
        X0, X1 = max(0, x0 - m), min(W, x1 + m)
        Y0, Y1 = max(0, y0 - m), min(H, cut + 3 * band + m)
        f = out[Y0:Y1, X0:X1].astype(np.float64)
        h, w = f.shape[:2]
        lum = f @ LUMA
        yy = np.arange(h)[:, None] + Y0
        xx = np.arange(w)[None, :] + X0
        inbox = (xx >= x0) & (xx < x1) & (yy >= y0)
        above = inbox & (yy < cut)
        below_cols = (xx >= x0 - m // 2) & (xx < x1 + m // 2)

        # background model (from pixels around the head that look like plain backdrop)
        ring = np.zeros((h, w), bool)
        ring[:max(1, y0 - Y0)] = True
        ring[:, :max(1, x0 - X0)] = True
        ring[:, min(w - 1, x1 - X0):] = True
        ring &= (yy < cut)
        bgc = np.median(f[ring].reshape(-1, 3), axis=0)
        m0 = np.abs(f - bgc).max(2) < 12
        wgt = gaussian_filter(m0.astype(np.float64), 20)
        bg_ref = gaussian_filter(f * m0[..., None], (20, 20, 0)) / (wgt[..., None] + 1e-6)
        dev = np.abs(f - bg_ref).max(2)
        is_bg = m0 & (dev < 5)
        fg = ~is_bg

        # 1) everything of the head above the cut line (including anything enclosed by it,
        #    e.g. bright skin that happens to be close to the backdrop colour)
        U = binary_dilation(fg & above, iterations=12) | (yy >= cut)
        U = binary_fill_holes(U) & above

        # 2) short hair hanging just below the cut (long hair over clothes is left alone)
        region = below_cols & (yy >= cut) & (yy < cut + 3 * band)
        dark = (lum < 60) & region
        lab, _ = label(binary_dilation(dark, iterations=1) & region)
        hair = np.zeros((h, w), bool)
        for i, sl in enumerate(find_objects(lab), start=1):
            if sl is None:
                continue
            top, bot = sl[0].start + Y0, sl[0].stop - 1 + Y0
            if top <= cut + 2 and bot <= cut + band:
                hair |= (lab == i) & dark
        notch = np.zeros((h, w), bool)
        if hair.any():
            near = binary_dilation(hair, iterations=6) & (yy >= cut)
            for _ in range(4):
                hair |= binary_dilation(hair) & (lum < 120) & near
            hair |= binary_dilation(hair) & (lum < 150) & near
            # what is left of the clothes below the cut
            K = ~is_bg & ~hair & ~U & (yy >= cut)
            # thin bits with removed hair on both sides (earring chains, collar slivers)
            Kopen = binary_dilation(binary_erosion(K, iterations=6), iterations=7) & K
            def sides(mask, r=12):
                """Does `mask` occur within r px to the left / right of each pixel?"""
                cs = np.pad(np.cumsum(mask, axis=1), ((0, 0), (r + 1, r + 1)), mode="edge")
                cs[:, :r + 1] = 0
                xi = np.arange(w)
                return (cs[:, xi + r] - cs[:, xi]) > 0, (cs[:, xi + 2 * r + 1] - cs[:, xi + r + 1]) > 0
            hl, hr = sides(hair)
            el, er = sides(~K)
            thin = K & ~Kopen & el & er & (hl | hr)
            K &= ~thin
            # small fragments no longer attached to the body (e.g. a floating bead)
            lab_k, _ = label(K, structure=np.ones((3, 3)))
            keep = np.zeros(lab_k.max() + 1, bool)
            keep[np.unique(lab_k[-1][lab_k[-1] > 0])] = True
            sizes = np.bincount(lab_k.ravel())
            keep |= sizes >= 2000
            keep[0] = False
            debris = (lab_k > 0) & ~keep[lab_k] & binary_dilation(hair | U, iterations=15)
            K &= ~debris
            hair |= thin | debris
            # narrow gaps between two pieces of clothing: fill with cloth, not backdrop
            closed = binary_erosion(binary_dilation(K, iterations=10), iterations=10)
            notch = closed & hair
        U |= hair

        # pieces with no backdrop around them are filled from their surroundings instead
        ok_bg = is_bg & ~U
        lab1, n1 = label(U & ~notch)
        touch = np.unique(lab1[binary_dilation(ok_bg) & U])
        enclosed = (lab1 > 0) & ~np.isin(lab1, touch[touch > 0])
        P2 = enclosed | notch
        U1 = U & ~P2
        res = fill_region(f, U1, ok_bg, seed=bi + 1)
        if P2.any():
            res = fill_region(res, P2, ~P2, mono=True, seed=bi + 100)
        block = out[Y0:Y1, X0:X1]
        block[U] = np.clip(np.round(res[U]), 0, 255).astype(np.uint8)
        if debug is not None:
            debug.append({"origin": (X0, Y0), "U1": U1, "P2": P2})
    return out


def repair(img, mask, progress=None):
    """Fill painted pixels from their surroundings (for earrings / leftovers)."""
    out = img.copy()
    if not mask.any():
        return out
    H, W = img.shape[:2]
    lab, n = label(binary_dilation(mask, iterations=1))
    for i, sl in enumerate(find_objects(lab), start=1):
        if progress:
            progress(f"正在修补 {i}/{n} …")
        y0, y1 = max(0, sl[0].start - 30), min(H, sl[0].stop + 30)
        x0, x1 = max(0, sl[1].start - 30), min(W, sl[1].stop + 30)
        U = lab[y0:y1, x0:x1] == i
        f = out[y0:y1, x0:x1].astype(np.float64)
        res = fill_region(f, U, ~U, mono=True, seed=1000 + i)
        out[y0:y1, x0:x1][U] = np.clip(np.round(res[U]), 0, 255).astype(np.uint8)
    return out
