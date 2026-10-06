# =====================================================================
#  obj3d Kaggle runner：由 iPhone App 自動上傳並執行
#  輸入：Kaggle Dataset 中的拍攝資料（影格 + ARKit 相機位置）
#  輸出：/kaggle/working/ 下的 model.glb、model_mm.stl、model_mm.ply、
#        preview.bin（App 預覽用）、result_meta.json
# =====================================================================
JOB_ID = "__JOB_ID__"
RUNNER_VERSION = "2026.10.09-3dgs-clean"   # 每次修改運算程式時更新
SMOOTH_LEVEL = "__SMOOTH__"      # low / medium / high（由 App 設定）

import os, sys, json, time, glob, shutil, subprocess, zipfile, traceback
T0 = time.time()
WORK = "/kaggle/working"
TMP = "/tmp/obj3d"
WARN = []

POISSON_DEPTH = 9
SMOOTH_PRESETS = {"low": (0.7, 3, 0), "medium": (1.0, 6, 4), "high": (1.4, 12, 12)}   # (高斯模糊, 實體平滑, 額外平滑)
if SMOOTH_LEVEL not in SMOOTH_PRESETS:
    SMOOTH_LEVEL = "medium"
SMOOTH_BLUR, SMOOTH_TAUBIN, SMOOTH_ITERS = SMOOTH_PRESETS[SMOOTH_LEVEL]
MAX_BATCHES = 3                  # T4 一次約 50 張，分 3 批 → 最多約 150 張
USE_TSDF = True                  # 深度圖用 TSDF 融合（平均掉雜訊，表面更平滑）
TEXTURE_SIZE = 2048              # 貼圖解析度
TSDF_MAX_RES = 384               # TSDF 格子數上限（每邊）

CONF_DROP_PERCENT = 40
HR_TOL = 0.04
MASK_ERODE_PX = 4
MAX_FRAMES = 0          # 0 = 依 GPU 記憶體自動決定

# ---- App「開發者參數」：由 App 填入 JSON；0 或缺少 = 用程式預設值 ----
PARAMS_JSON = r"""__PARAMS__"""
try:
    PARAMS = json.loads(PARAMS_JSON) if not PARAMS_JSON.startswith("__") else {}
except Exception:
    PARAMS = {}


def _p(name, default):
    try:
        v = float(PARAMS.get(name))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def _flag(name, default=True):
    v = PARAMS.get(name)
    if v is None:
        return default
    try:
        return bool(float(v))
    except (TypeError, ValueError):
        return bool(v)


CONF_DROP_PERCENT = _p("conf_drop_percent", CONF_DROP_PERCENT)
MASK_ERODE_PX = int(_p("mask_erode_px", MASK_ERODE_PX))
HR_TOL = _p("hr_tol_pct", HR_TOL * 100) / 100
CONS_TOL = _p("consistency_tol_pct", 2.5) / 100
POISSON_DEPTH = int(_p("poisson_depth", POISSON_DEPTH))
TSDF_MAX_RES = int(_p("tsdf_max_res", TSDF_MAX_RES))
TEXTURE_SIZE = int(_p("texture_size", TEXTURE_SIZE))
MAX_BATCHES = int(_p("batches", MAX_BATCHES))
DEV_FRAMES_PER_BATCH = int(_p("max_frames_per_batch", 0))
DEV_SOLID_MM = _p("solid_voxel_mm", 0)
DEV_TRUNC_MM = _p("tsdf_trunc_mm", 0)
DEV_BASE_MARGIN = _p("base_margin_cm", 0)
SMOOTH_ITERS = int(_p("extra_smooth_iters", SMOOTH_ITERS))      # 0.5 代表「0 次」
USE_TSDF = _flag("use_tsdf", USE_TSDF)
USE_TEXTURE = _flag("use_texture", True)
USE_HIRES = _flag("use_hires", True)
DEPTH_REFINE = _flag("depth_refine", True)
USE_BASE = _flag("use_base", True)
SCALE_SOURCE = {0: "auto", 1: "arkit", 2: "ruler"}.get(int(_p("scale_source", 0)), "auto")
USE_GS = _flag("gs_enable", True)
USE_ARKIT = _flag("use_arkit", True)          # 是否使用手機動作追蹤（ARKit）的位置資料
GS_ITERS = int(_p("gs_iters", 7000))
GS_RES = int(_p("gs_res", 1280))
GS_MAX = int(_p("gs_max_gaussians", 400_000))
GS_CLEAN = _flag("gs_clean", True)              # 去毛邊：剔除輪廓外的漂浮點、針狀與太淡的高斯
GS_MIN_OPACITY = _p("gs_min_opacity", 0.05)
PARAMS_USED = {k: v for k, v in PARAMS.items()
               if k != "capture_fps" and ((isinstance(v, bool) and not v) or (not isinstance(v, bool) and v not in (0, 0.0, None)))}

SOLID_VOXEL = 0.06      # 封閉實體的體素大小（cm，以 12 cm 物體為基準，會依物體大小縮放）


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def write_meta(**kw):
    kw.setdefault("jobId", JOB_ID)
    kw["runner_version"] = RUNNER_VERSION
    kw["elapsed_s"] = round(time.time() - T0, 1)
    kw["warnings"] = WARN
    with open(f"{WORK}/result_meta.json", "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2)


def sh(cmd):
    log("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def install():
    sh("pip -q install open3d trimesh scikit-image einops safetensors huggingface_hub")
    sh("SAM2_BUILD_CUDA=0 pip -q install git+https://github.com/facebookresearch/sam2.git")
    if not os.path.isdir("/tmp/vggt"):
        sh("git clone -q --depth 1 https://github.com/facebookresearch/vggt /tmp/vggt")
    sys.path.insert(0, "/tmp/vggt")


NO_RUN = os.environ.get("OBJ3D_NO_RUN") == "1"     # 本機測試用
os.makedirs(TMP, exist_ok=True)
if not NO_RUN:
    try:
        install()
    except Exception as e:
        traceback.print_exc()
        write_meta(status="error", error="install", message=str(e)[:500])
        raise SystemExit(0)

import numpy as np
import cv2
from PIL import Image


# =====================================================================
#  幾何工具
# =====================================================================
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def cam_centers_dirs(extr):
    """extr: (S,3,4) OpenCV cam-from-world → 相機中心 C (S,3)、視線方向 fwd (S,3)、旋轉 R (S,3,3)"""
    R = extr[:, :, :3]
    t = extr[:, :, 3]
    C = -np.einsum('sji,sj->si', R, t)
    fwd = R[:, 2, :]
    return C, fwd, R


def nearest_point_to_lines(C, d):
    """所有相機視線最接近的共同點 ≈ 環繞拍攝的中心（小熊位置）"""
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, u in zip(C, d):
        P = np.eye(3) - np.outer(u, u)
        A += P; b += P @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]


def fit_table_plane(pts, center, cams, radius, dist_thr, up_prior=None, n_sample=400_000, seed=0, tries=8):
    """在小熊附近 RANSAC 找「接近水平」的桌面平面，回傳 (unit normal n, d)，n·x + d = 0，相機在正側"""
    sub = pts[np.linalg.norm(pts - center, axis=1) < radius]
    if len(sub) > n_sample:
        sub = sub[np.random.default_rng(seed).choice(len(sub), n_sample, replace=False)]
    n0 = len(sub)
    for _ in range(tries):
        if len(sub) < 500:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub))
        (a, b, c, d), inl = pc.segment_plane(dist_thr, 3, 3000)
        n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        if up_prior is None or abs(n @ up_prior) > 0.75:      # 必須大致水平
            P = sub[inl]; mu = P.mean(0)
            _, _, Vt = np.linalg.svd(P - mu, full_matrices=False)
            n = Vt[2]
            if np.mean((cams - mu) @ n) < 0:
                n = -n
            return n, -n @ mu, len(inl) / n0
        sub = np.delete(sub, inl, 0)                           # 這是牆面之類的，移除後再找
    raise RuntimeError("找不到水平的桌面平面：請確認 ⑤-c 的遮罩正確，或增加 CONF_DROP_PERCENT")


def align_transform(n, d, center, cam0):
    """建立 world → aligned 的剛體轉換：桌面 z=0、z 軸朝上、原點在環繞中心正下方"""
    o = center - (n @ center + d) * n
    x = cam0 - o
    x = x - (x @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    Ra = np.stack([x, y, n])
    return Ra, o


def apply_align(P, Ra, o):
    return (P - o) @ Ra.T


def pixel_ray(u, v, K, Rcw, C):
    """像素 (u,v) → 世界座標射線 (origin, dir)"""
    dc = np.linalg.inv(K) @ np.array([u, v, 1.0])
    dw = Rcw.T @ dc
    return C, dw / np.linalg.norm(dw)


def ray_plane_z(Co, dw, z0=0.0):
    s = (z0 - Co[2]) / dw[2]
    return Co + s * dw


def make_base(center_xy, radius, shape, thickness, table_pts, table_cols, step=0.15, n_sect=256):
    """產生底座（桌面）網格：頂面為密集網格，顏色取自真實桌面點；z=0 為桌面"""
    nr = max(4, int(np.ceil(radius / step)))
    th = np.linspace(0, 2 * np.pi, n_sect, endpoint=False)
    if shape == 'square':
        k = 1.0 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
    else:
        k = np.ones_like(th)
    verts = [[center_xy[0], center_xy[1], 0.0]]
    for i in range(1, nr + 1):
        r = radius * i / nr * k
        verts += list(np.stack([center_xy[0] + r * np.cos(th),
                                center_xy[1] + r * np.sin(th),
                                np.zeros_like(th)], 1))
    verts = np.array(verts)
    ring = lambda i, j: 1 + (i - 1) * n_sect + (j % n_sect)
    tris = []
    for j in range(n_sect):
        tris.append([0, ring(1, j), ring(1, j + 1)])
    for i in range(1, nr):
        for j in range(n_sect):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            tris += [[a, c, d], [a, d, b]]
    # 頂面顏色：附近真實桌面點的平均
    tree = cKDTree(table_pts[:, :2])
    dist, idx = tree.query(verts[:, :2], k=8, distance_upper_bound=1.0)
    fallback = np.median(table_cols, 0)
    cols = np.empty((len(verts), 3))
    for vi in range(len(verts)):
        ok = np.isfinite(dist[vi])
        cols[vi] = table_cols[idx[vi][ok]].mean(0) if ok.any() else fallback
    tris = np.array(tris)
    if thickness > 0:
        nv = len(verts)
        rim_top = np.array([ring(nr, j) for j in range(n_sect)])
        rim_bot = verts[rim_top].copy(); rim_bot[:, 2] = -thickness
        bot_c = np.array([[center_xy[0], center_xy[1], -thickness]])
        verts = np.vstack([verts, rim_bot, bot_c])
        side_col = fallback * 0.85
        cols = np.vstack([cols, np.tile(side_col, (n_sect + 1, 1))])
        rb = lambda j: nv + (j % n_sect)
        bc = nv + n_sect
        extra = []
        for j in range(n_sect):
            a, b = rim_top[j], rim_top[(j + 1) % n_sect]
            extra += [[a, rb(j), rb(j + 1)], [a, rb(j + 1), b]]   # 側面（法向量朝外）
            extra.append([bc, rb(j + 1), rb(j)])                   # 底面
        tris = np.vstack([tris, np.array(extra)])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    m.compute_vertex_normals()
    return m


def orient_normals_by_source_cam(pcd, src_pts, src_cam_centers):
    """每個點的法向量朝向「拍到它的那台相機」"""
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)
    _, idx = cKDTree(src_pts).query(P, k=1)
    to_cam = src_cam_centers[idx] - P
    flip = np.einsum('ij,ij->i', N, to_cam) < 0
    N[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(N)
    return pcd


def poisson_mesh(pcd, depth, trim_q, bbox_min, bbox_max, keep_largest=False, max_dist=None, scale=1.1, crop=True):
    """Poisson 重建。max_dist：只刪掉離資料點超過這個距離的曲面（比用密度修剪更不容易開出大洞）"""
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=False)
    if max_dist is not None:
        d = cKDTree(np.asarray(pcd.points)).query(np.asarray(mesh.vertices), k=1)[0]
        mesh.remove_vertices_by_mask(d > max_dist)
    elif trim_q > 0:
        dens = np.asarray(dens)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_q))
    if crop:
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max))
    tri_clusters, counts, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters); counts = np.asarray(counts)
    if len(counts):
        keep = (np.arange(len(counts)) == counts.argmax()) if keep_largest else counts >= max(200, 0.02 * counts.max())
        mesh.remove_triangles_by_mask(~keep[tri_clusters])
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def project_colors(P, N, mesh, frame_paths, masks, intrinsic, Rcam, Ra, Cc, sx, sy, padL, padT,
                   sharp, fallback, topk=5, min_cos=0.2, eps=0.08):
    """把 3D 點投影回原始解析度影格取色：遮擋判斷後，取「最正面、最近、最清晰」的前 topk 個視角，
    再取各色頻的中位數（自動排除反光亮點）。P、N、Cc 的單位與 mesh 相同（公分）。"""
    import cv2
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    n = len(P)
    best_w = np.zeros((n, topk), np.float32)
    best_c = np.zeros((n, topk, 3), np.float32)
    sharp = np.asarray(sharp, float); sharp = np.sqrt(sharp / np.median(sharp))
    kd = np.ones((3, 3), np.uint8)
    for fi, path in enumerate(frame_paths):
        if masks is not None and not masks[fi].any():
            continue
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        H0, W0 = img.shape[:2]
        K = intrinsic[fi]
        fx, fy = K[0, 0] / sx, K[1, 1] / sy
        cx, cy = (K[0, 2] - padL + 0.5) / sx - 0.5, (K[1, 2] - padT + 0.5) / sy - 0.5
        R2 = Rcam[fi] @ Ra.T
        D = P - Cc[fi]
        dist = np.linalg.norm(D, axis=1)
        Xc = D @ R2.T
        z = Xc[:, 2]
        zs = np.maximum(z, 1e-6)
        u = fx * Xc[:, 0] / zs + cx
        v = fy * Xc[:, 1] / zs + cy
        cosv = np.einsum('ij,ij->i', N, -D) / np.maximum(dist, 1e-9)
        ok = (z > 0) & (u >= 0) & (u <= W0 - 1) & (v >= 0) & (v <= H0 - 1) & (cosv > min_cos)
        if masks is not None:
            md = cv2.dilate(masks[fi].astype(np.uint8), kd, iterations=6) > 0
            ok[ok] &= md[np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)]
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        dirs = D[idx] / dist[idx, None]
        rays = np.hstack([np.broadcast_to(Cc[fi], (len(idx), 3)), dirs]).astype(np.float32)
        t = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        idx = idx[t > dist[idx] - eps]
        if len(idx) == 0:
            continue
        m_ = len(idx); w_ = int(np.ceil(m_ / 4096)) * 4096
        mu = np.zeros(w_, np.float32); mv = np.zeros(w_, np.float32)
        mu[:m_], mv[:m_] = u[idx], v[idx]
        col = cv2.remap(img, mu.reshape(-1, 4096), mv.reshape(-1, 4096), cv2.INTER_LINEAR).reshape(-1, 3)[:m_]
        w = (cosv[idx] ** 2 / dist[idx] * sharp[fi]).astype(np.float32)
        allw = np.concatenate([best_w[idx], w[:, None]], 1)
        allc = np.concatenate([best_c[idx], col[:, None]], 1)
        order = np.argsort(-allw, 1)[:, :topk]
        best_w[idx] = np.take_along_axis(allw, order, 1)
        best_c[idx] = np.take_along_axis(allc, order[..., None], 1)
    out = np.array(fallback, dtype=np.float64)
    has = best_w[:, 0] > 0
    if has.any():
        c = best_c[has].copy()
        c[best_w[has] <= 0] = np.nan
        out[has] = np.nanmedian(c, axis=1)
    return out, has


def color_from_frames(mesh, vidx, frame_paths, masks, intrinsic, Rcam, Ra, Cc, sx, sy, padL, padT,
                      sharp, fallback, topk=5, min_cos=0.2):
    """網格頂點取色（中位數，去反光）"""
    V = np.asarray(mesh.vertices)[vidx]
    N = np.asarray(mesh.vertex_normals)[vidx]
    out, has = project_colors(V, N, mesh, frame_paths, masks, intrinsic, Rcam, Ra, Cc, sx, sy, padL, padT,
                              sharp, fallback, topk=topk, min_cos=min_cos)
    print(f"影像取色：{has.mean():.1%} 的頂點有可見視角")
    return out


def bake_texture(mesh, color_fn, tex_size=2048, max_faces=200_000):
    """UV 展開（Open3D UVAtlas）→ 在貼圖上逐像素算出對應的 3D 位置 → 用 color_fn(P, N) 取色。
    回傳 (頂點, 三角面, uv[0..1，v 向下], 貼圖 uint8, 精簡後的網格)"""
    import cv2
    m = o3d.geometry.TriangleMesh(mesh)
    if len(m.triangles) > max_faces:
        m = m.simplify_quadric_decimation(max_faces)
    m.remove_unreferenced_vertices()
    m.remove_degenerate_triangles()
    m.compute_vertex_normals()
    mt = o3d.t.geometry.TriangleMesh.from_legacy(m)
    mt.compute_uvatlas(size=tex_size, parallel_partitions=4)       # Open3D 內建 UVAtlas 展開
    TUV = mt.triangle.texture_uvs.numpy().astype(np.float64)        # (面數, 3, 2)
    V = np.asarray(m.vertices); F = np.asarray(m.triangles)
    corner_v = F.reshape(-1)
    corner_uv = TUV.reshape(-1, 2)
    key = np.c_[corner_v, np.round(corner_uv * 1e6)].astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V2 = V[corner_v[first]]
    N2 = np.asarray(m.vertex_normals)[corner_v[first]]
    UV = corner_uv[first]
    Fi = inv.reshape(-1, 3)
    R = tex_size
    tri_id = np.full((R, R), -1, np.int32)
    pts = np.round(UV[Fi] * R * 16).astype(np.int32)
    for t in range(len(Fi)):
        cv2.fillConvexPoly(tri_id, pts[t], int(t), shift=4)
    ys, xs = np.nonzero(tri_id >= 0)
    tid = tri_id[ys, xs]
    p = np.stack([(xs + 0.5) / R, (ys + 0.5) / R], 1)
    a, b, c = UV[Fi[tid, 0]], UV[Fi[tid, 1]], UV[Fi[tid, 2]]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    bu = 1 - bv - bw
    bary = np.clip(np.stack([bu, bv, bw], 1), 0, 1)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-9)
    P = (V2[Fi[tid]] * bary[..., None]).sum(1)
    Nn = (N2[Fi[tid]] * bary[..., None]).sum(1)
    Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)
    cols = color_fn(P, Nn)
    tex = np.zeros((R, R, 3), np.float32)
    tex[ys, xs] = cols
    # 把整張貼圖的空白處都填上「最近的已上色像素」：縮圖（mipmap）時就不會混進黑色，避免黑點與黑邊
    from scipy import ndimage
    empty = tri_id < 0
    if empty.any():
        idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
        tex = tex[idx[0], idx[1]]
    tex8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    return V2, Fi.astype(np.int32), UV.astype(np.float32), tex8, m


def voxel_solid(mesh, vox, close_iters=2, base=None, sink=0.15, min_comp=0.05, max_voxels=40e6, clip_lo=None, clip_hi=None,
                blur=0.8, taubin=5):
    """把表面網格轉成「保證封閉」的實體：
    1) 網格體素化，先切掉桌面以下並加上底座（把底部封起來）
    2) 填滿內部（小破洞先用較強的閉運算封住再判斷內部）
    3) 用體積方式裁掉資料範圍外的部分，Marching Cubes 轉回表面。base = (中心 xy, 半徑, 厚度) 或 None"""
    from scipy import ndimage
    from skimage import measure
    lo = np.asarray(mesh.get_min_bound(), float).copy(); hi = np.asarray(mesh.get_max_bound(), float).copy()
    if base is not None:
        cxy, R, T = base
        lo = np.minimum(lo, [cxy[0] - R, cxy[1] - R, -T]); hi = np.maximum(hi, [cxy[0] + R, cxy[1] + R, 0.0])
    vox = max(vox, (np.prod(hi - lo) / max_voxels) ** (1 / 3))
    pad = close_iters + 3
    lo = lo - pad * vox
    shape = np.ceil((hi - lo) / vox).astype(int) + pad + 1
    n = int(np.clip(mesh.get_surface_area() / (vox * 0.5) ** 2, 2e5, 1.5e7))
    P = np.asarray(mesh.sample_points_uniformly(n).points)
    idx = np.clip(np.floor((P - lo) / vox).astype(int), 0, shape - 1)
    occ = np.zeros(shape, bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    st = ndimage.generate_binary_structure(3, 1)
    occ = ndimage.binary_dilation(occ, structure=st, iterations=1)
    if close_iters > 0:
        occ = ndimage.binary_closing(occ, structure=st, iterations=close_iters)
    xc = lo[0] + (np.arange(shape[0]) + 0.5) * vox
    yc = lo[1] + (np.arange(shape[1]) + 0.5) * vox
    zc = lo[2] + (np.arange(shape[2]) + 0.5) * vox
    # 先切掉桌面以下、加上底座：Poisson 在看不到的底部常是開口的，必須先封住才能填滿內部
    base_occ = None
    if base is not None:
        disk = ((xc[:, None] - cxy[0]) ** 2 + (yc[None, :] - cxy[1]) ** 2) <= R * R
        zsel = (zc >= -T) & (zc <= 0)
        occ[:, :, zc < -sink] = False
        occ[:, :, zsel] |= disk[:, :, None]
        occ[:, :, zc < -T] = False
        base_occ = (disk[:, :, None] & zsel[None, None, :])
    else:
        occ[:, :, zc < 0] = False
        k0 = int(np.argmax(zc >= 0))
        for k in range(k0, min(k0 + 3, shape[2])):
            occ[:, :, k] = ndimage.binary_fill_holes(occ[:, :, k])
    heavy = max(close_iters, int(round(0.5 / vox)))
    sealed = ndimage.binary_closing(np.pad(occ, heavy), structure=st, iterations=heavy)[heavy:-heavy, heavy:-heavy, heavy:-heavy]
    inside = ndimage.binary_fill_holes(sealed) & ~sealed
    occ = ndimage.binary_fill_holes(occ) | inside
    occ = ndimage.binary_erosion(occ, structure=st, iterations=1)   # 抵銷前面膨脹的一層
    if base_occ is not None:
        occ |= base_occ
    if clip_lo is not None:                                    # 體積裁切：資料範圍外的 Poisson 外插部分
        keep_z = (zc <= clip_hi[2])
        if base_occ is not None:
            occ &= (((xc >= clip_lo[0]) & (xc <= clip_hi[0]))[:, None, None] & ((yc >= clip_lo[1]) & (yc <= clip_hi[1]))[None, :, None]
                    & keep_z[None, None, :]) | base_occ
        else:
            occ &= ((xc >= clip_lo[0]) & (xc <= clip_hi[0]))[:, None, None]
            occ &= ((yc >= clip_lo[1]) & (yc <= clip_hi[1]))[None, :, None]
            occ &= keep_z[None, None, :]
    lab, nlab = ndimage.label(occ)
    if nlab > 1:
        sizes = ndimage.sum(occ, lab, range(1, nlab + 1))
        occ = np.isin(lab, 1 + np.nonzero(sizes >= min_comp * sizes.max())[0])
    field = ndimage.gaussian_filter(np.pad(occ, 3).astype(np.float32), blur)   # 平滑後的等值面不會有非流形邊
    v, f, _, _ = measure.marching_cubes(field, level=0.5)
    V = lo + (v - 3 + 0.5) * vox
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(f.astype(np.int32)))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    Fm = np.asarray(m.triangles)                          # 移除正反重疊的重複面（會造成非流形邊）
    _, inv_, cnt_ = np.unique(np.sort(Fm, axis=1), axis=0, return_inverse=True, return_counts=True)
    if (cnt_ > 1).any():
        m.remove_triangles_by_mask(cnt_[inv_.ravel()] > 1); m.remove_unreferenced_vertices()
    tc, cnt, _ = m.cluster_connected_triangles()
    tc = np.asarray(tc); cnt = np.asarray(cnt)
    if len(cnt) > 1:
        m.remove_triangles_by_mask(~(cnt >= max(100, 0.01 * cnt.max()))[tc]); m.remove_unreferenced_vertices()
    import trimesh
    if trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False).volume < 0:
        m.triangles = o3d.utility.Vector3iVector(np.asarray(m.triangles)[:, ::-1])
    if taubin > 0:
        m = m.filter_smooth_taubin(number_of_iterations=taubin)
    m.compute_vertex_normals()
    return m, vox


# =====================================================================
#  讀取拍攝資料
# =====================================================================
def find_capture():
    metas = glob.glob("/kaggle/input/**/meta.json", recursive=True)
    if not metas:
        os.makedirs(f"{TMP}/cap", exist_ok=True)
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            zipfile.ZipFile(z).extractall(f"{TMP}/cap")
        metas = glob.glob(f"{TMP}/cap/**/meta.json", recursive=True)
    if not metas:
        raise RuntimeError("no_capture")
    for m in metas:
        meta = json.load(open(m, encoding="utf-8"))
        if meta.get("jobId") == JOB_ID:
            return os.path.dirname(m), meta
    return None, json.load(open(metas[0], encoding="utf-8"))


def sharpness(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, None, fx=0.5, fy=0.5)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def pad_geometry(W0, H0):
    if W0 >= H0:
        nw, nh = 518, round(H0 * 518 / W0 / 14) * 14
    else:
        nh, nw = 518, round(W0 * 518 / H0 / 14) * 14
    return nw, nh, (518 - nw) // 2, (518 - nh) // 2, nw / W0, nh / H0


def umeyama(src, dst):
    """找相似轉換 dst ≈ s R src + t"""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((xs ** 2).sum() / len(src))
    return s, R, md - s * R @ ms


# =====================================================================
#  GPU 模型（VGGT、SAM 2）
# =====================================================================
def run_vggt(paths, mode, with_camera):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    images = load_and_preprocess_images(paths, mode=mode).to("cuda")
    out = {}
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            tokens, ps_idx = model.aggregator(images[None])
        # 各預測頭用 float32 權重：只把會用到的幾層轉成 float32，其餘層直接釋放（同時省下不少 GPU 記憶體）
        # 修正：在支援 bfloat16 的 GPU 上會出現「mat1 and mat2 must have the same dtype」錯誤
        tokens = list(tokens)
        need = set(getattr(model.depth_head, "intermediate_layer_idx", [4, 11, 17, 23])) | {len(tokens) - 1}
        for i in range(len(tokens)):
            tokens[i] = tokens[i].float() if i in need else None
        torch.cuda.empty_cache()
        if with_camera:
            pose_enc = model.camera_head(tokens)[-1]
            e, k = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            out["extrinsic"] = e[0].float().cpu().numpy()
            out["intrinsic"] = k[0].float().cpu().numpy()
        d, c = model.depth_head(tokens, images[None], ps_idx)
    out["depth"] = d[0, ..., 0].float().cpu().numpy()
    out["conf"] = c[0].float().cpu().numpy()
    out["imgs"] = images.permute(0, 2, 3, 1).float().cpu().numpy()
    del model, tokens, images, d, c
    torch.cuda.empty_cache()
    return out


def run_sam(sel_dir, n, W0, H0, prompts):
    """prompts = {影格編號: (x, y)}：物體在該影格上的位置（由 3D 定位算出）"""
    return run_sam_objects(sel_dir, n, W0, H0, {1: {int(fi): [(x, y)] for fi, (x, y) in prompts.items()}})[1]


def find_prompts(extrinsic, intrinsic, depth, W0, H0, padL, padT, sx, sy, n_prompts=4):
    """用 3D 找物體：所有相機視線最集中的點＝物體位置，再投影回每張影格。
    不依賴「物體剛好在畫面中央」，拍攝時偶爾偏掉也沒關係。"""
    C, fwd, _ = cam_centers_dirs(extrinsic)
    d = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    P = nearest_point_to_lines(C, d)
    for _ in range(4):                                   # 加權重算，降低偏掉的影格的影響
        v = P - C
        dist = np.linalg.norm(v - (v * d).sum(1, keepdims=True) * d, axis=1)
        w = 1.0 / (dist + np.median(dist) + 1e-9)
        A = np.zeros((3, 3)); b = np.zeros(3)
        for c_, u_, wi in zip(C, d, w):
            Mx = wi * (np.eye(3) - np.outer(u_, u_)); A += Mx; b += Mx @ c_
        P = np.linalg.lstsq(A, b, rcond=None)[0]
    n = len(extrinsic)
    info = [None] * n
    for i in range(n):
        Xc = extrinsic[i][:, :3] @ P + extrinsic[i][:, 3]
        if Xc[2] <= 0:
            continue
        K = intrinsic[i]
        up = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
        vp = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
        uo = (up - padL + 0.5) / sx - 0.5
        vo = (vp - padT + 0.5) / sy - 0.5
        if not (0 <= uo < W0 and 0 <= vo < H0):
            continue
        ui = int(np.clip(round(up), 2, depth.shape[2] - 3)); vi = int(np.clip(round(vp), 2, depth.shape[1] - 3))
        dpix = np.median(depth[i][vi - 2:vi + 3, ui - 2:ui + 3])
        if dpix > Xc[2] * 1.05:                          # 這個位置看到的是物體後面的背景
            continue
        dc = np.hypot((uo - W0 / 2) / (W0 / 2), (vo - H0 / 2) / (H0 / 2))
        info[i] = (dc, float(uo), float(vo))
    prompts = {}
    for bidx in np.array_split(np.arange(n), n_prompts):
        cands = [(info[i][0], i) for i in bidx if info[i] is not None and info[i][0] < 0.7]
        if cands:
            _, i = min(cands)
            prompts[int(i)] = (info[i][1], info[i][2])
    return prompts


def save_mask_diagnostic(frame_paths, masks, prompts, bad, path):
    from PIL import ImageDraw
    idx = np.linspace(0, len(frame_paths) - 1, 12).astype(int)
    tw, th = 240, 320
    sheet = Image.new("RGB", (tw * 4, th * 3), (40, 40, 40))
    for k, i in enumerate(idx):
        im = np.asarray(Image.open(frame_paths[i]).convert("RGB").resize((tw, th))).astype(np.float32)
        m = np.asarray(Image.fromarray(masks[i].astype(np.uint8) * 255).resize((tw, th))) > 127
        im[~m] *= 0.3
        tile = Image.fromarray(im.astype(np.uint8))
        dr = ImageDraw.Draw(tile)
        sxr, syr = tw / masks.shape[2], th / masks.shape[1]
        if i in prompts:
            x, y = prompts[i]
            dr.ellipse([x * sxr - 9, y * syr - 9, x * sxr + 9, y * syr + 9], outline=(0, 255, 0), width=4)
        label = f"#{i}" + (" BAD" if i in bad else "") + (" PROMPT" if i in prompts else "")
        dr.rectangle([0, 0, 8 * len(label) + 8, 16], fill=(0, 0, 0))
        dr.text((4, 2), label, fill=(255, 255, 0) if i in bad else (255, 255, 255))
        sheet.paste(tile, ((k % 4) * tw, (k // 4) * th))
    # 提示影格另外補上（若不在上面 12 張裡）
    sheet.save(path, quality=88)


def save_camera_diagnostic(Cc, Pb, Pg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    sb = Pb[rng.choice(len(Pb), min(30000, len(Pb)), replace=False)]
    near = np.linalg.norm(Pg[:, :2], axis=1) < max(30, 1.5 * np.ptp(sb[:, :2], axis=0).max())
    pg = Pg[near]
    sg = pg[rng.choice(len(pg), min(40000, len(pg)), replace=False)] if len(pg) else pg
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (a, b), title in zip(axs, ((0, 1), (0, 2)), ("Top view (cm)", "Side view (cm)")):
        if len(sg):
            ax.scatter(sg[:, a], sg[:, b], s=0.2, c="0.6")
        ax.scatter(sb[:, a], sb[:, b], s=0.3, c="red")
        ax.plot(Cc[:, a], Cc[:, b], "b.-", lw=0.8, ms=4)
        ax.set_title(title + "  red=object  blue=camera"); ax.axis("equal"); ax.grid(alpha=0.3)
    axs[1].axhline(0, color="k", lw=0.8)
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


def gpu_memory_gb():
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 1e9


# =====================================================================
#  主流程
# =====================================================================
def run_sam_objects(sel_dir, n, W0, H0, objs):
    """多物體追蹤：objs = {物體編號: {影格: [(x, y), ...]}}，回傳 {物體編號: 遮罩 (n,H0,W0)}"""
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    def _track(use_bf16):
        masks = {int(o): np.zeros((n, H0, W0), bool) for o in objs}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            st = pred.init_state(sel_dir, offload_video_to_cpu=True)
            for oid, frs in objs.items():
                for fi, pts in frs.items():
                    pred.add_new_points_or_box(st, frame_idx=int(fi), obj_id=int(oid),
                                               points=np.array(pts, np.float32), labels=np.ones(len(pts), np.int32))
            for rev in (False, True):
                for fidx, obj_ids, lg in pred.propagate_in_video(st, reverse=rev):
                    for k, oid in enumerate(obj_ids):
                        masks[int(oid)][fidx] |= (lg[k, 0] > 0).cpu().numpy()
        return masks
    try:
        masks = _track(bf16)
    except RuntimeError as e:
        if not bf16 or "dtype" not in str(e):
            raise
        log("SAM 2 bfloat16 失敗，改用 float32 重跑：", str(e)[:120])
        torch.cuda.empty_cache()
        masks = _track(False)
    del pred
    torch.cuda.empty_cache()
    return masks


def run_vggt_batches_anchor(paths, n_batch):
    """沒有 ARKit 軌跡可用時（轉盤模式）的分批推論：每批都包含同一組「錨點影格」，
    用錨點的相機位置把各批對齊到第一批的座標系。"""
    n = len(paths)
    B = int(min(MAX_BATCHES, np.ceil(n / n_batch)))
    if B <= 1:
        v = run_vggt(paths, "pad", True)
        log(f"VGGT（1 批，{n} 張）完成")
        return (v["extrinsic"].astype(np.float32), v["intrinsic"], v["depth"], v["conf"], v["imgs"],
                np.zeros(n, int), 1)
    n_anchor = int(np.clip(n_batch // 5, 6, 12))
    anchors = sorted(set(np.linspace(0, n - 1, n_anchor).astype(int).tolist()))
    aset = set(anchors)
    rest = [i for i in range(n) if i not in aset]
    per = max(1, n_batch - len(anchors))
    B = int(min(MAX_BATCHES, np.ceil(len(rest) / per)))
    groups = [rest[b::B] for b in range(B)]
    ext = np.zeros((n, 3, 4), np.float32); Ks = np.zeros((n, 3, 3), np.float32)
    depth = conf = imgs = None
    batch_of = np.zeros(n, int)
    ref = None
    for b, g in enumerate(groups):
        idx = anchors + g
        v = run_vggt([paths[i] for i in idx], "pad", True)
        C, _, _ = cam_centers_dirs(v["extrinsic"])
        Ca = C[:len(anchors)]
        if ref is None:
            ref = Ca.copy(); s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            s, R, t = umeyama(Ca, ref)
            res = np.linalg.norm(s * Ca @ R.T + t - ref, axis=1)
            log(f"第 {b+1} 批錨點對齊殘差（相對單位）中位數 {np.median(res):.4f}")
        if depth is None:
            depth = np.zeros((n,) + v["depth"].shape[1:], np.float32)
            conf = np.zeros_like(depth); imgs = np.zeros((n,) + v["imgs"].shape[1:], np.float32)
        for k, i in enumerate(idx):
            if b > 0 and k < len(anchors):
                continue
            Rj = v["extrinsic"][k][:, :3].astype(np.float64); tj = v["extrinsic"][k][:, 3].astype(np.float64)
            Rn = Rj @ R.T
            ext[i] = np.hstack([Rn, (s * tj - Rn @ t)[:, None]])
            Ks[i] = v["intrinsic"][k]
            depth[i] = v["depth"][k] * s; conf[i] = v["conf"][k]; imgs[i] = v["imgs"][k]
            batch_of[i] = b
        log(f"VGGT 第 {b+1}/{B} 批（{len(idx)} 張，含 {len(anchors)} 張錨點）完成")
        del v
    return ext, Ks, depth, conf, imgs, batch_of, B


def ruler_scale(marks_sel, extrinsic, intrinsic, depth, RMASK, sx, sy, padL, padT, L_true_cm):
    """比例尺：沿著使用者標記的兩端取深度，擬合 3D 直線，再求兩端點在直線上的位置，得到長度（相對單位）。
    回傳 (每單位幾公尺, 比例尺的 3D 點（世界座標）, 各影格量到的長度)"""
    Ls, pts_world = [], []
    for fi, p1, p2 in marks_sel:
        K = intrinsic[fi].astype(np.float64); Kinv = np.linalg.inv(K)
        p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
        to_proc = lambda q: np.array([(q[0] + 0.5) * sx - 0.5 + padL, (q[1] + 0.5) * sy - 0.5 + padT])
        H0m, W0m = RMASK[fi].shape
        P = []
        for t in np.linspace(0.04, 0.96, 80):
            q = p1 + (p2 - p1) * t
            qi = (int(np.clip(round(q[1]), 0, H0m - 1)), int(np.clip(round(q[0]), 0, W0m - 1)))
            if RMASK[fi].any() and not RMASK[fi][qi]:
                continue
            u, v = to_proc(q)
            ui, vi = int(round(u)), int(round(v))
            win = depth[fi][max(vi - 1, 0):vi + 2, max(ui - 1, 0):ui + 2]
            win = win[win > 0]
            if len(win) == 0:
                continue
            P.append(np.median(win) * (Kinv @ np.array([u, v, 1.0])))
        if len(P) < 12:
            continue
        P = np.array(P)
        for _ in range(2):                                   # 擬合直線並排除離群點
            c = P.mean(0)
            dirv = np.linalg.svd(P - c, full_matrices=False)[2][0]
            res = np.linalg.norm((P - c) - ((P - c) @ dirv)[:, None] * dirv, axis=1)
            P = P[res <= max(np.percentile(res, 80), 1e-12)]
        c = P.mean(0)
        dirv = np.linalg.svd(P - c, full_matrices=False)[2][0]

        def on_line(q):
            rv = Kinv @ np.array([*to_proc(q), 1.0]); rv /= np.linalg.norm(rv)
            b_ = dirv @ rv; d_ = dirv @ c; e_ = rv @ c
            den = 1 - b_ * b_
            s_ = (b_ * e_ - d_) / den if abs(den) > 1e-9 else 0.0
            return c + s_ * dirv
        Ls.append(float(np.linalg.norm(on_line(p1) - on_line(p2))))
        pts_world.append((P - extrinsic[fi][:, 3]) @ extrinsic[fi][:, :3])
    if not Ls:
        raise RuntimeError("找不到比例尺：請確認標記的兩端在比例尺上，而且比例尺放在轉盤上跟著物體一起轉")
    L_units = float(np.median(Ls))
    return (L_true_cm / 100.0) / L_units, np.concatenate(pts_world), Ls


def gravity_up(T_ar_sel, extrinsic):
    """轉盤模式的「上方」：用 ARKit 記錄的手機姿態算出每張畫面中重力的方向（物體只繞垂直軸轉，所以重力在物體座標中不變）"""
    M_ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])   # ARKit 相機 → 直式影像相機
    g_w = np.array([0, -1.0, 0])
    downs = []
    for T, E in zip(T_ar_sel, extrinsic):
        g_cv = M_ @ (T[:3, :3].T @ g_w)
        g = E[:, :3].astype(np.float64).T @ g_cv
        downs.append(g / np.linalg.norm(g))
    downs = np.array(downs)
    d = downs.mean(0); d /= np.linalg.norm(d)
    keep = downs @ d > np.cos(np.radians(20))
    if keep.sum() >= 3:
        d = downs[keep].mean(0); d /= np.linalg.norm(d)
    spread = float(np.degrees(np.arccos(np.clip(downs @ d, -1, 1))).mean())
    return -d, spread


def up_from_cameras(extrinsic):
    """沒有 iPhone 姿態資料時（從「照片」匯入的影片）判斷「上方」：
    相機繞物體（或物體旋轉）時，相機位置大致在一個水平圓上 → 圓所在平面的法向量就是上下方向；
    正負號由「畫面上方」平均方向決定（拍攝時手機通常是正拿的）。"""
    C, fwd, R = cam_centers_dirs(extrinsic)
    img_up = -R[:, 1, :]
    u0 = img_up.mean(0); u0 /= np.linalg.norm(u0)
    c = C.mean(0)
    _, sv, Vt = np.linalg.svd(C - c, full_matrices=False)
    n = Vt[2]
    planar = sv[2] / max(sv[1], 1e-12)
    if planar < 0.35 and abs(n @ u0) > 0.5:
        return (n if n @ u0 > 0 else -n), "相機軌跡平面"
    return u0, "畫面方向"


def frame_support(extrinsic, intrinsic, depth, masks, box, stride=6, tol=0.03, n_ref=20):
    """不靠 ARKit 的相機位置檢查：每張畫面的物體點，有多少比例能在其他畫面找到吻合的深度。
    相機位置算錯的畫面，比例會明顯偏低。"""
    padL, padT, nw, nh = box
    S = len(depth)
    refs = np.linspace(0, S - 1, min(n_ref, S)).astype(int)
    frac = np.full(S, np.nan)
    for i in range(S):
        if not masks[i].any():
            continue
        g = np.zeros(depth.shape[1:], bool)
        g[padT:padT + nh, padL:padL + nw] = cv2.resize(masks[i].astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        g &= depth[i] > 0
        vv, uu = np.nonzero(g[::stride, ::stride]); uu = uu * stride; vv = vv * stride
        if len(uu) < 30:
            continue
        z = depth[i][vv, uu]; K = intrinsic[i]
        Xc = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], 1)
        P = (Xc - extrinsic[i][:, 3]) @ extrinsic[i][:, :3]
        sup = np.zeros(len(P), int)
        for j in refs:
            if j == i:
                continue
            Y = P @ extrinsic[j][:, :3].T + extrinsic[j][:, 3]
            zj = Y[:, 2]; Kj = intrinsic[j]; ok = zj > 1e-9
            zs = np.where(ok, zj, 1)
            ui = np.round(Kj[0, 0] * Y[:, 0] / zs + Kj[0, 2]).astype(int); vi = np.round(Kj[1, 1] * Y[:, 1] / zs + Kj[1, 2]).astype(int)
            ok &= (ui >= padL) & (ui < padL + nw) & (vi >= padT) & (vi < padT + nh)
            idx = np.nonzero(ok)[0]
            d = depth[j][vi[idx], ui[idx]]
            sup[idx[(d > 0) & (np.abs(zj[idx] - d) < tol * d)]] += 1
        frac[i] = float(np.mean(sup >= 2))
    return frac


def ruler_triangulate(marks_sel, extrinsic, intrinsic, sx, sy, padL, padT):
    """比例尺兩端在 2 張以上畫面的點選 → 三角測量出 3D 端點（世界座標）。自動處理某張點反順序。"""
    import itertools
    def ray(fi, q):
        u, v = (q[0] + 0.5) * sx - 0.5 + padL, (q[1] + 0.5) * sy - 0.5 + padT
        R_ = extrinsic[fi][:, :3].astype(np.float64); t_ = extrinsic[fi][:, 3].astype(np.float64)
        d = R_.T @ (np.linalg.inv(intrinsic[fi].astype(np.float64)) @ np.array([u, v, 1.0]))
        return -R_.T @ t_, d / np.linalg.norm(d)
    def dist(P, rays):
        return np.mean([np.linalg.norm((P - c) - ((P - c) @ d) * d) for c, d in rays])
    best = None
    for flips in itertools.product([0, 1], repeat=len(marks_sel) - 1):
        fl = (0,) + flips
        Ar = [ray(fi, (p1, p2)[f]) for (fi, p1, p2), f in zip(marks_sel, fl)]
        Br = [ray(fi, (p1, p2)[1 - f]) for (fi, p1, p2), f in zip(marks_sel, fl)]
        A_ = nearest_point_to_lines(np.array([c for c, _ in Ar]), np.array([d for _, d in Ar]))
        B_ = nearest_point_to_lines(np.array([c for c, _ in Br]), np.array([d for _, d in Br]))
        res = dist(A_, Ar) + dist(B_, Br)
        if best is None or res < best[0]:
            best = (res, A_, B_, Ar)
    res, A_, B_, Ar = best
    ang = max(np.degrees(np.arccos(np.clip(abs(d1 @ d2), -1, 1))) for (_, d1), (_, d2) in itertools.combinations(Ar, 2))
    return A_, B_, res / max(np.linalg.norm(A_ - B_), 1e-12), ang


def _qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def train_3dgs(frame_paths, masks, viewmats, Ks, W0, H0, ds, P0, C0, iters, log_every=1000):
    """以 gsplat 訓練 3DGS（只學物體：物體外要透明）。座標單位：公分（轉正後，z 軸朝上）"""
    import torch, torch.nn.functional as TF
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    dev = "cuda"
    Wt, Ht = int(round(W0 * ds)), int(round(H0 * ds))
    IMG, MSK, MSKA = [], [], []
    for p, m in zip(frame_paths, masks):
        IMG.append(torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((Wt, Ht)), np.float32) / 255))
        mm = cv2.resize(cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)), (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        MSK.append(torch.from_numpy(mm.astype(np.float32))[..., None])
        ma = cv2.resize(m.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_AREA).astype(np.float32)   # 透明度目標：不外擴、邊緣柔和
        MSKA.append(torch.from_numpy(ma)[..., None])
    VM = [torch.tensor(v, dtype=torch.float32) for v in viewmats]
    KS = [torch.tensor(k, dtype=torch.float32) for k in Ks]
    cam_c = np.array([-(v[:3, :3].T @ v[:3, 3]) for v in viewmats])
    scene_scale = float(np.linalg.norm(cam_c - cam_c.mean(0), axis=1).max()) * 1.1
    N = len(P0)
    d3 = cKDTree(P0).query(P0, k=4)[0][:, 1:].mean(1)
    obj_size = float((np.percentile(P0, 98, 0) - np.percentile(P0, 2, 0)).max())
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(P0, dtype=torch.float32, device=dev)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(np.clip(d3, 1e-3, None), dtype=torch.float32, device=dev))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(TF.normalize(torch.randn(N, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(torch.tensor((C0 - 0.5) / 0.28209479177387814, dtype=torch.float32, device=dev)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=dev)),
    })
    lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15) for k in lrs}
    strat = DefaultStrategy(verbose=False, refine_start_iter=500, refine_stop_iter=int(iters * 0.6),
                            reset_every=max(1000, int(iters * 0.4)), refine_every=100)
    strat.check_sanity(params, opts)
    st = strat.initialize_state(scene_scale=scene_scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / iters))

    def ssim(x, y):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        g = torch.exp(-(torch.arange(11, device=x.device, dtype=torch.float32) - 5) ** 2 / (2 * 1.5 ** 2)); g = g / g.sum()
        k = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
        f = lambda t: TF.conv2d(t, k, padding=5, groups=3)
        mx, my = f(x), f(y)
        vx, vy, cxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
        return (((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))).mean()

    rng = np.random.default_rng(1); t0 = time.time()
    for step in range(iters):
        j = int(rng.integers(len(IMG)))
        img, msk, mska = IMG[j].to(dev), MSK[j].to(dev), MSKA[j].to(dev)
        rc, ra, info = rasterization(means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
                                     opacities=torch.sigmoid(params["opacities"]),
                                     colors=torch.cat([params["sh0"], params["shN"]], 1),
                                     viewmats=VM[j][None].to(dev), Ks=KS[j][None].to(dev), width=Wt, height=Ht,
                                     sh_degree=min(step // 1000, 3), packed=False)
        strat.step_pre_backward(params, opts, st, step, info)
        tgt = img * msk
        loss = 0.8 * (rc[0] - tgt).abs().mean() + 0.2 * (1 - ssim(rc[0].permute(2, 0, 1)[None], tgt.permute(2, 0, 1)[None]))
        loss = loss + 0.3 * (ra[0] - mska).abs().mean()                 # 物體外要透明（不外擴的遮罩）
        sc_ = torch.exp(params["scales"])
        smax, smin = sc_.max(1).values, sc_.min(1).values.clamp_min(1e-6)
        loss = loss + 0.01 * TF.relu(smax / smin - 10).mean()          # 抑制細長的針狀高斯
        loss = loss + 0.1 * TF.relu(smax - 0.03 * obj_size).mean() / max(obj_size, 1e-6)   # 單一高斯不要太大
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        sched.step()
        strat.step_post_backward(params, opts, st, step, info, packed=False)
        if step % log_every == 0 or step == iters - 1:
            log(f"3DGS {step}/{iters}  loss {loss.item():.4f}  元素 {len(params['means']):,}  {time.time() - t0:.0f}s")
    out = {k: v.detach().cpu().numpy() for k, v in params.items()}
    del params, opts, IMG, MSK
    torch.cuda.empty_cache()
    return out


def clean_gaussians(g, viewmats, Ks, masks, W0, H0, ds, min_opacity=0.05, min_out=3):
    """去毛邊：(1) 中心點在 min_out 張以上畫面落在物體輪廓外 → 漂浮點；(2) 太淡；(3) 又大又細長的針狀"""
    Wt, Ht = int(round(W0 * ds)), int(round(H0 * ds))
    m = g["means"]
    outside = np.zeros(len(m), np.int32)
    k5 = np.ones((5, 5), np.uint8)
    for V, K, mk in zip(viewmats, Ks, masks):
        md = cv2.resize(cv2.dilate(mk.astype(np.uint8), k5), (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 0
        Xc = m @ V[:3, :3].T + V[:3, 3]
        z = Xc[:, 2]; ok = z > 1e-6; zs = np.where(ok, z, 1)
        u = np.round(K[0, 0] * Xc[:, 0] / zs + K[0, 2]).astype(int); v = np.round(K[1, 1] * Xc[:, 1] / zs + K[1, 2]).astype(int)
        ok &= (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)
        ii = np.nonzero(ok)[0]
        outside[ii] += ~md[v[ii], u[ii]]
    op = 1 / (1 + np.exp(-g["opacities"]))
    s = np.exp(g["scales"]); smax, smin = s.max(1), np.maximum(s.min(1), 1e-9)
    needle = (smax / smin > 25) & (smax > 3 * np.median(smax))
    keep = (outside < min_out) & (op >= min_opacity) & ~needle
    log(f"3DGS 去毛邊：輪廓外 {int((outside >= min_out).sum()):,}、太淡 {int((op < min_opacity).sum()):,}、"
        f"針狀 {int(needle.sum()):,}，保留 {keep.mean():.0%}")
    return {k: v[keep] for k, v in g.items()}


def export_gaussians(g, max_n, ply_path, splat_path):
    """匯出：完整 .ply（INRIA 格式，給電腦）＋精簡 .splat（每個 32 bytes，給手機）。座標：公尺、y 軸朝上（與 GLB 相同）"""
    m = g["means"] * 0.01; s = g["scales"] + np.log(0.01)
    q = g["quats"] / np.linalg.norm(g["quats"], axis=1, keepdims=True)
    o = g["opacities"]; f0 = g["sh0"][:, 0, :]; fr = g["shN"].transpose(0, 2, 1).reshape(len(m), -1)
    imp = 1 / (1 + np.exp(-o)) * np.exp(s.sum(1))
    order = np.argsort(-imp)[:max_n]
    m, s, q, o, f0, fr = m[order], s[order], q[order], o[order], f0[order], fr[order]
    m = np.stack([m[:, 0], m[:, 2], -m[:, 1]], 1)                      # z 朝上 → y 朝上
    qx = np.array([[np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0]])     # 繞 x 軸 -90°
    q = _qmul(np.repeat(qx, len(q), 0), q)
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(fr.shape[1])] + \
            ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.concatenate([m, np.zeros_like(m), f0, fr, o[:, None], s, q], 1).astype(np.float32)
    with open(ply_path, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(data)).encode())
        fh.write("".join(f"property float {n}\n" for n in names).encode()); fh.write(b"end_header\n")
        fh.write(data.tobytes())
    rec = np.zeros(len(m), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = m; rec["s"] = np.exp(s)
    rgb = np.clip(0.5 + 0.28209479177387814 * f0, 0, 1)
    rec["c"] = (np.concatenate([rgb, 1 / (1 + np.exp(-o[:, None]))], 1) * 255).astype(np.uint8)
    rec["r"] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    rec.tofile(splat_path)
    return len(m)


def run_vggt_batches(frame_paths, C_ar, n_batch):
    """分批跑 VGGT（交錯分組），每批各自用 ARKit 軌跡對齊到同一個公尺座標系後合併。"""
    n = len(frame_paths)
    B = int(min(MAX_BATCHES, np.ceil(n / n_batch)))
    groups = [list(range(b, n, B)) for b in range(B)]
    ext = np.zeros((n, 3, 4), np.float32); Ks = np.zeros((n, 3, 3), np.float32)
    depth = conf = imgs = None
    batch_of = np.zeros(n, int)
    for b, g in enumerate(groups):
        v = run_vggt([frame_paths[i] for i in g], "pad", True)
        C, _, _ = cam_centers_dirs(v["extrinsic"])
        s, R, t = umeyama(C, C_ar[g])
        res = np.linalg.norm((s * C @ R.T + t) - C_ar[g], axis=1)
        kf = res <= np.percentile(res, 80)
        s, R, t = umeyama(C[kf], C_ar[g][kf])
        for k, i in enumerate(g):
            Rj = v["extrinsic"][k][:, :3].astype(np.float64); tj = v["extrinsic"][k][:, 3].astype(np.float64)
            Rn = Rj @ R.T
            ext[i] = np.hstack([Rn, (s * tj - Rn @ t)[:, None]])
            Ks[i] = v["intrinsic"][k]
        if depth is None:
            depth = np.zeros((n,) + v["depth"].shape[1:], np.float32)
            conf = np.zeros_like(depth); imgs = np.zeros((n,) + v["imgs"].shape[1:], np.float32)
        depth[g] = v["depth"] * s; conf[g] = v["conf"]; imgs[g] = v["imgs"]
        batch_of[g] = b
        log(f"VGGT 第 {b+1}/{B} 批（{len(g)} 張）完成")
        del v
    return ext, Ks, depth, conf, imgs, batch_of, B


def batch_points(idx, ext, Ks, depth, conf, mask, stride=2):
    pts = []
    for i in idx:
        d = depth[i][::stride, ::stride]; m = mask[i][::stride, ::stride] & (d > 1e-6)
        if m.sum() == 0:
            continue
        vv, uu = np.nonzero(m); uu = uu * stride; vv = vv * stride
        z = d[m]; K = Ks[i]
        Xc = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], 1)
        pts.append((Xc - ext[i][:, 3]) @ ext[i][:, :3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def icp_refine_batches(ext, Ks, depth, conf, mask, batch_of, B):
    """以第 1 批為基準，用 ICP 微調其他批的位置（ARKit 對齊後的殘餘誤差）"""
    if B < 2:
        return ext
    P_ref = batch_points(np.nonzero(batch_of == 0)[0], ext, Ks, depth, conf, mask)
    if len(P_ref) < 500:
        return ext
    sz = float((np.percentile(P_ref, 98, 0) - np.percentile(P_ref, 2, 0)).max())   # 物體大小（任何單位都適用）
    def cloud(b):
        P = batch_points(np.nonzero(batch_of == b)[0], ext, Ks, depth, conf, mask)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.02 * sz)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1 * sz, max_nn=30))
        return pc
    ref = cloud(0)
    if len(ref.points) < 500:
        return ext
    for b in range(1, B):
        src = cloud(b)
        if len(src.points) < 500:
            continue
        reg = o3d.pipelines.registration.registration_icp(
            src, ref, 0.06 * sz, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = reg.transformation
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        mv = np.linalg.norm(T[:3, 3])
        log(f"ICP 第 {b+1} 批：吻合度 {reg.fitness:.2f}，修正 {mv / sz:.1%}（相對物體大小）/ {ang:.2f}°")
        if reg.fitness < 0.3 or mv > 0.2 * sz or ang > 4:
            WARN.append(f"第 {b+1} 批的 ICP 對齊不可靠，沿用 ARKit 對齊")
            continue
        Rt, tt = T[:3, :3], T[:3, 3]
        for i in np.nonzero(batch_of == b)[0]:
            Rn = ext[i][:, :3] @ Rt.T
            ext[i] = np.hstack([Rn, (ext[i][:, 3] - Rn @ tt)[:, None]])
    return ext


def tsdf_fuse(frames, extrinsic, P_ref):
    """TSDF 融合：在 3D 格子中把所有影格看到的深度做加權平均，隨機雜訊會被大幅抵銷。
    （Open3D 0.20 的 ScalableTSDFVolume 對浮點深度圖有問題，這裡用 UniformTSDFVolume）"""
    lo, hi = np.percentile(P_ref, 1, 0), np.percentile(P_ref, 99, 0)
    size = float((hi - lo).max())
    center = (lo + hi) / 2
    length = size * 1.3 + 0.02
    res = int(np.clip(length / (size / 220), 128, TSDF_MAX_RES))
    vox = length / res
    trunc = float(np.clip(0.04 * size, 4 * vox, 0.012))
    if DEV_TRUNC_MM > 0:
        trunc = max(DEV_TRUNC_MM / 1000.0, 2 * vox)
    log(f"TSDF 截斷距離 {trunc*1000:.1f} mm")
    vol = o3d.pipelines.integration.UniformTSDFVolume(
        length=length, resolution=res, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=(center - length / 2).reshape(3, 1))
    for fi, d, (fx, fy, cx, cy), col in frames:
        H, W = d.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(col)), o3d.geometry.Image(np.ascontiguousarray(d)),
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
        E = np.eye(4); E[:3, :4] = extrinsic[fi]
        vol.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(W, H, float(fx), float(fy), float(cx), float(cy)), E)
    pc = vol.extract_point_cloud()
    return np.asarray(pc.points), np.asarray(pc.colors), vox, vol.extract_triangle_mesh()


def refine_depth_frames(frames, extrinsic, ref_mesh):
    """逐張深度校正：把每張深度圖對齊到初步融合出的共同表面，
    擬合 d_ref ≈ a·d + b + c·u + e·v（倍數、偏移、左右／上下傾斜），用穩健迴歸排除離群值。"""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ref_mesh))
    out, before, after = [], [], []
    for fi, d, (fx, fy, cx, cy), col in frames:
        H, W = d.shape
        vv, uu = np.nonzero(d > 0)
        if len(uu) < 500:
            out.append((fi, d, (fx, fy, cx, cy), col)); continue
        R = extrinsic[fi][:, :3].astype(np.float64); t = extrinsic[fi][:, 3].astype(np.float64)
        C = -R.T @ t
        dc = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(len(uu))], 1)
        dw = dc @ R                                       # z_cam = 1 → 命中距離 t 就是深度
        rays = np.hstack([np.broadcast_to(C, dw.shape), dw]).astype(np.float32)
        tr = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
        z = d[vv, uu].astype(np.float64)
        good = np.isfinite(tr) & (np.abs(tr - z) < 0.15 * z)
        if good.sum() < 300:
            out.append((fi, d, (fx, fy, cx, cy), col)); continue
        un, vn = (uu - W / 2) / W, (vv - H / 2) / H
        X = np.stack([z, np.ones_like(z), un, vn], 1)[good]; y = tr[good]
        w = np.ones(len(y))
        for _ in range(4):                                # Huber 加權最小平方
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
            res = y - X @ coef
            s = 1.4826 * np.median(np.abs(res)) + 1e-9
            w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(res), 1e-12))
        a_, b_, c_, e_ = coef
        if not (0.85 < a_ < 1.15) or abs(b_) > 0.03:
            out.append((fi, d, (fx, fy, cx, cy), col)); continue
        before.append(np.median(np.abs(y - X[:, 0])))
        after.append(np.median(np.abs(res)))
        dn = np.zeros_like(d)
        dn[vv, uu] = (a_ * z + b_ + c_ * un + e_ * vn).astype(np.float32)
        out.append((fi, dn, (fx, fy, cx, cy), col))
    if before:
        log(f"逐張深度校正：與共同表面的差距中位數 {np.median(before)*1000:.2f} mm → {np.median(after)*1000:.2f} mm")
    return out


def consistency_filter(P, F, extrinsic, intrinsic, depth, frame_ok, box, tol=0.025, min_support=2):
    """多視角一致性：點要被其他至少 min_support 個視角看到「同一個深度」才保留；
    若點出現在其他相機看到的表面前方很多（重影），就剔除。"""
    padL, padT, nw, nh = box
    S = len(extrinsic)
    support = np.zeros(len(P), np.int16)
    viol = np.zeros(len(P), np.int16)
    for j in range(S):
        if not frame_ok[j]:
            continue
        Xc = P @ extrinsic[j][:, :3].T + extrinsic[j][:, 3]
        z = Xc[:, 2]
        K = intrinsic[j]
        ok = z > 1e-6
        zs = np.where(ok, z, 1.0)
        ui = np.round(K[0, 0] * Xc[:, 0] / zs + K[0, 2]).astype(np.int64)
        vi = np.round(K[1, 1] * Xc[:, 1] / zs + K[1, 2]).astype(np.int64)
        inb = ok & (ui >= padL) & (ui < padL + nw) & (vi >= padT) & (vi < padT + nh) & (F != j)
        idx = np.nonzero(inb)[0]
        if len(idx) == 0:
            continue
        d = depth[j][vi[idx], ui[idx]]
        good = d > 1e-6
        rel = (z[idx] - d) / np.maximum(d, 1e-9)
        support[idx[good & (np.abs(rel) < tol)]] += 1
        viol[idx[good & (rel < -2.5 * tol)]] += 1
    keep = (support >= min_support) & (viol <= np.maximum(1, support // 2))
    return keep


def main():
    cap_dir, meta = find_capture()
    if cap_dir is None:
        write_meta(status="error", error="stale_capture",
                   message=f"資料集還是舊的拍攝（{meta.get('jobId')}）")
        return
    frames = meta["frames"]
    if len(frames) < 12:
        raise RuntimeError("影格太少，請拍久一點（至少繞物體半圈以上）")
    paths = [os.path.join(cap_dir, f["file"]) for f in frames]
    T_ar = np.array([np.array(f["transform"], float).reshape(4, 4).T for f in frames])   # column-major
    C_ar_all = T_ar[:, :3, 3]
    TT = meta.get("mode") in ("turntable", "video")      # 物體旋轉模式，或從「照片」匯入的影片（都沒有可用的手機軌跡）
    VIDEO = meta.get("mode") == "video"
    span = np.ptp(C_ar_all, axis=0).max()
    if not TT and USE_ARKIT and span < 0.15:
        raise RuntimeError(f"手機移動範圍太小（{span*100:.0f} cm），請繞著物體走一圈；如果是旋轉物體的拍法，請在 App 選「物體旋轉」模式")

    # ---- 1. 挑影格 ----
    gb = gpu_memory_gb()
    N_batch = int(np.clip((gb - 3.5) / 0.23, 16, 150))
    if DEV_FRAMES_PER_BATCH > 0:
        N_batch = DEV_FRAMES_PER_BATCH
    N = MAX_FRAMES or N_batch * MAX_BATCHES
    N = min(N, len(paths))
    sc = np.array([sharpness(p) for p in paths])
    pick = [int(b[np.argmax(sc[b])]) for b in np.array_split(np.arange(len(paths)), N) if len(b)]
    marks = meta.get("scale_marks") or []
    objp = meta.get("object_point")
    if (TT or not USE_ARKIT) and not marks:
        WARN.append("沒有比例尺，也沒有可用的手機位置資料：模型沒有真實尺寸（相對比例）")
    need = {m["frame"] for m in marks} | ({objp["frame"]} if objp else set())
    if need:
        pick = sorted(set(pick) | {i for i, f in enumerate(frames) if f["file"] in need})
    SEL = f"{TMP}/sel"
    shutil.rmtree(SEL, ignore_errors=True); os.makedirs(SEL)
    frame_paths = []
    for k, i in enumerate(pick):
        dst = f"{SEL}/{k:03d}.jpg"
        shutil.copy(paths[i], dst); frame_paths.append(dst)
    C_ar = C_ar_all[pick]
    W0, H0 = Image.open(frame_paths[0]).size
    nw, nh, padL, padT, sx, sy = pad_geometry(W0, H0)
    log(f"影格 {len(paths)} → 選用 {len(pick)}（GPU {gb:.0f} GB），{W0}x{H0}")

    crop_src = frame_paths
    if TT:
        # ---- 2T. 轉盤模式：先用 SAM 2 分出「物體」和「比例尺」，把背景遮掉後再交給 VGGT ----
        sel_of = {frames[i]["file"]: k for k, i in enumerate(pick)}
        if objp and objp["frame"] in sel_of:
            obj_prompts = {sel_of[objp["frame"]]: [tuple(objp["p"])]}
            target_method = "使用者標記"
        else:
            obj_prompts = {int(i): [(W0 / 2, H0 / 2)] for i in sorted(set(np.linspace(0, len(frame_paths) - 1, 4).astype(int)))}
            target_method = "畫面中央"
        ruler_prompts, marks_sel = {}, []
        for m in marks:
            if m["frame"] in sel_of:
                k = sel_of[m["frame"]]
                p1, p2 = np.array(m["p1"], float), np.array(m["p2"], float)
                ruler_prompts[k] = [tuple(p1 * 0.95 + p2 * 0.05), tuple(p1 * 0.05 + p2 * 0.95)]   # 只用兩端附近（中段可能被物體擋住）
                marks_sel.append((k, p1, p2))
        prompts = {k: v[0] for k, v in obj_prompts.items()}
        mk = run_sam_objects(SEL, len(frame_paths), W0, H0, {1: obj_prompts, 2: ruler_prompts} if ruler_prompts else {1: obj_prompts})
        MASKS = mk[1]; RMASK = mk.get(2, np.zeros_like(MASKS))
        MSEL = f"{TMP}/msel"
        shutil.rmtree(MSEL, ignore_errors=True); os.makedirs(MSEL)
        k15 = np.ones((15, 15), np.uint8)
        crop_src = []
        for k, pth in enumerate(frame_paths):
            im = np.asarray(Image.open(pth).convert("RGB")).copy()
            keepm = cv2.dilate((MASKS[k] | RMASK[k]).astype(np.uint8), k15) > 0
            im[~keepm] = 128                                  # 背景塗成灰色
            q = f"{MSEL}/{k:03d}.jpg"
            Image.fromarray(im).save(q, quality=95)
            crop_src.append(q)
        log(f"轉盤模式：物體與比例尺遮罩完成（物體提示：{target_method}）")
        extrinsic, intrinsic, depth, conf, imgs, batch_of, NB = run_vggt_batches_anchor(crop_src, N_batch)
        if marks_sel:
            m_per_unit, ruler_pts, Ls = ruler_scale(marks_sel, extrinsic, intrinsic, depth, RMASK, sx, sy, padL, padT,
                                                    float(meta.get("scale_length_cm", 15.0)))
        else:                                             # 沒有比例尺：先用 1，最後再換成相對比例
            m_per_unit, ruler_pts, Ls = 1.0, np.zeros((0, 3)), []
        if len(Ls) > 1 and (max(Ls) - min(Ls)) / np.median(Ls) > 0.06:
            WARN.append(f"各影格量到的比例尺長度差異 {(max(Ls)-min(Ls))/np.median(Ls):.0%}，尺寸可能不準")
        extrinsic[:, :, 3] *= m_per_unit; depth *= m_per_unit     # 換成公尺
        ruler_pts = ruler_pts * m_per_unit
        if VIDEO or not USE_ARKIT or np.allclose(T_ar[pick][:, :3, :3], np.eye(3)):
            up_tt, how = up_from_cameras(extrinsic)
            log(f"比例尺：{len(Ls)} 張影格；上方方向依據：{how}")
            if how == "畫面方向":
                WARN.append("無法從相機軌跡判斷上下方向，改用畫面方向，模型可能稍微傾斜")
        else:
            up_tt, g_spread = gravity_up(T_ar[pick], extrinsic)
            log(f"比例尺：{len(Ls)} 張影格；重力方向一致性 {g_spread:.1f}°")
            if g_spread > 8:
                WARN.append(f"重力方向估計的分散度 {g_spread:.0f}°，物體可能不是繞垂直軸旋轉")
        Ccam, fwd, Rcam = cam_centers_dirs(extrinsic)
        frame_ok = np.ones(len(frame_paths), bool); bad_pose = []; rms_cm = 0.0
        log(f"物體定位（{target_method}）：提示影格 {sorted(prompts)}")
    else:
        # ---- 2. VGGT 第一次 + SAM 2 ----
        if USE_ARKIT:
            extrinsic, intrinsic, depth, conf, imgs, batch_of, NB = run_vggt_batches(frame_paths, C_ar, N_batch)
            log(f"VGGT 第一次完成（{NB} 批，共 {len(frame_paths)} 張）")
            # 用 ARKit 軌跡檢查 VGGT 的相機位置：差太多的影格不拿來建模
            Ccam, fwd, Rcam = cam_centers_dirs(extrinsic)
            s1, R1, t1 = umeyama(Ccam, C_ar)
            res = np.linalg.norm((s1 * Ccam @ R1.T + t1) - C_ar, axis=1)
            keep_fit = res <= np.percentile(res, 80)
            s1, R1, t1 = umeyama(Ccam[keep_fit], C_ar[keep_fit])
            res = np.linalg.norm((s1 * Ccam @ R1.T + t1) - C_ar, axis=1)
            rms_cm = float(np.sqrt(np.mean(res[keep_fit] ** 2)) * 100)
            pose_thr = max(0.02, 3.0 * np.median(res))
            frame_ok = res <= pose_thr
            bad_pose = [int(i) for i in np.nonzero(~frame_ok)[0]]
            log(f"ARKit 對齊：殘差 {rms_cm:.2f} cm；相機位置不一致而略過的影格 {bad_pose}")
            if rms_cm > 3:
                WARN.append(f"相機軌跡對齊誤差偏大（{rms_cm:.1f} cm），尺寸可能不準")
            if len(bad_pose) > 0.3 * len(frame_ok):
                WARN.append(f"有 {len(bad_pose)} 張影格的相機位置不可靠，建議放慢速度、保持距離重拍")
        else:
            extrinsic, intrinsic, depth, conf, imgs, batch_of, NB = run_vggt_batches_anchor(frame_paths, N_batch)
            log(f"未使用手機動作追蹤（ARKit）：相機位置只由畫面推算（{NB} 批，共 {len(frame_paths)} 張）")
            Ccam, fwd, Rcam = cam_centers_dirs(extrinsic)
            frame_ok = np.ones(len(frame_paths), bool); bad_pose = []; rms_cm = 0.0
        prompts = find_prompts(extrinsic, intrinsic, depth, W0, H0, padL, padT, sx, sy)
        target_method = "3D 自動定位"
        if not prompts:
            target_method = "畫面中央（3D 定位失敗）"
            WARN.append("無法用 3D 定位物體，改用畫面中央")
            prompts = {int(i): (W0 / 2, H0 / 2) for i in sorted(set(np.linspace(0, len(frame_paths) - 1, 4).astype(int)))}
        log(f"物體定位（{target_method}）：提示影格 {sorted(prompts)}")
        MASKS = run_sam(SEL, len(frame_paths), W0, H0, prompts)
    area = MASKS.reshape(len(MASKS), -1).mean(1)
    med = float(np.median(area))
    bad = [int(i) for i in np.nonzero((area < 0.2 * med) | (area > 3.0 * med))[0]]
    for i in bad:                                         # 遮罩異常的影格不拿來建物體
        MASKS[i] = False
    if not TT and not USE_ARKIT:                          # 沒有 ARKit：用畫面之間的一致性找出相機位置算錯的影格
        fs = frame_support(extrinsic, intrinsic, depth, MASKS, (padL, padT, nw, nh))
        med_fs = float(np.nanmedian(fs))
        bad_pose = [int(i) for i in np.nonzero(fs < max(0.3, med_fs - 0.25))[0]]
        frame_ok = np.ones(len(frame_paths), bool); frame_ok[bad_pose] = False
        log(f"畫面一致性檢查（不用 ARKit）：中位數 {med_fs:.0%}；相機位置不一致而略過的影格 {bad_pose}")
    for i in bad_pose:                                    # 相機位置不可靠的影格也不拿來建物體
        MASKS[i] = False
    log(f"SAM 2 完成，遮罩面積中位數 {med:.1%}，異常影格 {bad}")
    save_mask_diagnostic(frame_paths, MASKS, prompts, set(bad), f"{WORK}/diag_masks.jpg")
    if med < 0.005:
        raise RuntimeError("找不到物體：請讓物體保持在畫面中，並繞著物體拍一圈")
    if med > 0.8:
        WARN.append("遮罩幾乎佔滿畫面，可能選到背景")
    if len(bad) > 0.3 * len(MASKS):
        WARN.append(f"有 {len(bad)} 張影格的遮罩異常，請看診斷圖")

    crop_info, hr_paths, hr_fids = {}, [], []
    v2 = None
    if USE_HIRES:
        # ---- 3. 高解析第二次推論（只裁物體附近）----
        HR = f"{TMP}/crops"
        shutil.rmtree(HR, ignore_errors=True); os.makedirs(HR)
        for fi, m in enumerate(MASKS):
            ys, xs = np.nonzero(m)
            if len(xs) < 500:
                continue
            side = int(min(max(np.ptp(xs), np.ptp(ys)) * 1.3 + 20, W0, H0))
            x0 = int(np.clip((xs.min() + xs.max()) / 2 - side / 2, 0, W0 - side))
            y0 = int(np.clip((ys.min() + ys.max()) / 2 - side / 2, 0, H0 - side))
            p = f"{HR}/{fi:03d}.png"
            Image.open(crop_src[fi]).convert("RGB").crop((x0, y0, x0 + side, y0 + side)).resize((518, 518), Image.BICUBIC).save(p)
            crop_info[fi] = (x0, y0, side); hr_paths.append(p); hr_fids.append(fi)
        v2 = {"depth": [], "conf": [], "imgs": []}
        hr_order = []
        for b in range(NB):
            sel = [j for j, fi in enumerate(hr_fids) if batch_of[fi] == b]
            if not sel:
                continue
            vb = run_vggt([hr_paths[j] for j in sel], "crop", False)
            for k, j in enumerate(sel):
                v2["depth"].append(vb["depth"][k]); v2["conf"].append(vb["conf"][k]); v2["imgs"].append(vb["imgs"][k])
                hr_order.append(j)
            del vb
        order = np.argsort(hr_order)
        v2 = {k: np.stack([v2[k][o] for o in order]) for k in v2}
        log("VGGT 第二次完成")
    else:
        log("已在開發者參數中關閉高解析第二次推論")

    # ---- 4. 點雲（VGGT 座標）----
    S_, H, W = depth.shape
    k3 = np.ones((3, 3), np.uint8)
    BMASK = np.zeros(depth.shape, bool); BDIL = np.zeros(depth.shape, bool)
    for i, m in enumerate(MASKS):
        m8 = m.astype(np.uint8)
        BMASK[i, padT:padT + nh, padL:padL + nw] = cv2.resize(cv2.erode(m8, k3, iterations=MASK_ERODE_PX), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        BDIL[i, padT:padT + nh, padL:padL + nw] = cv2.resize(cv2.dilate(m8, k3, iterations=12), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    extrinsic = icp_refine_batches(extrinsic, intrinsic, depth, conf, BMASK, batch_of, NB)
    Ccam, fwd, Rcam = cam_centers_dirs(extrinsic)
    u_, v_ = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    cam_pts = np.stack([(u_ - cx) * depth / fx, (v_ - cy) * depth / fy, depth], -1)
    world = np.einsum('shwj,sjk->shwk', cam_pts - extrinsic[:, None, None, :, 3], extrinsic[:, :, :3])
    del cam_pts
    valid = np.zeros(depth.shape, bool)
    valid[:, padT + 2:padT + nh - 2, padL + 2:padL + nw - 2] = True
    valid &= depth > 1e-6
    bg_px = valid & ~BDIL & frame_ok[:, None, None]
    if TT:
        bg_px[:] = False                                  # 轉盤模式背景已遮掉，沒有桌面點
    elif bg_px.any():
        bg_px &= conf >= np.percentile(conf[bg_px], CONF_DROP_PERCENT)
    P_bg0 = world[bg_px].astype(np.float64)
    COL_bg = np.clip(imgs[bg_px].astype(np.float64), 0, 1)
    obj_px = valid & BMASK
    obj_px &= conf >= np.percentile(conf[obj_px], CONF_DROP_PERCENT)
    P_obj1 = world[obj_px].astype(np.float64)
    del world

    # 高解析點（對齊回第一次推論的相機）
    uu, vv = np.meshgrid(np.arange(518), np.arange(518))
    hp, hc, hf, ratios = [], [], [], []
    tsdf_in = []
    for j, fi in enumerate(hr_fids):
        x0, y0, side = crop_info[fi]; s_c = 518 / side
        uo = (uu + 0.5) / s_c - 0.5 + x0; vo = (vv + 0.5) / s_c - 0.5 + y0
        up = ((uo + 0.5) * sx - 0.5 + padL).astype(np.float32)
        vp = ((vo + 0.5) * sy - 0.5 + padT).astype(np.float32)
        d1 = cv2.remap(depth[fi].astype(np.float32), up, vp, cv2.INTER_LINEAR)
        m = cv2.erode(MASKS[fi].astype(np.uint8), k3, iterations=MASK_ERODE_PX)[y0:y0 + side, x0:x0 + side]
        m = cv2.resize(m, (518, 518), interpolation=cv2.INTER_NEAREST) > 0
        d2, c2 = v2["depth"][j], v2["conf"][j]
        ok = m & (d1 > 1e-6) & (d2 > 1e-6)
        if ok.sum() < 300:
            continue
        a = np.median(d1[ok] / d2[ok]); ratios.append(a)
        d2a = d2 * a
        ok &= np.abs(d2a - d1) < HR_TOL * d1
        ok &= c2 >= np.percentile(c2[m], CONF_DROP_PERCENT)
        K = intrinsic[fi]
        fxc, fyc = K[0, 0] / sx * s_c, K[1, 1] / sy * s_c
        cxc = (((K[0, 2] - padL + 0.5) / sx - 0.5) - x0 + 0.5) * s_c - 0.5
        cyc = (((K[1, 2] - padT + 0.5) / sy - 0.5) - y0 + 0.5) * s_c - 0.5
        z_ = d2a[ok]
        Xc = np.stack([(uu[ok] - cxc) * z_ / fxc, (vv[ok] - cyc) * z_ / fyc, z_], 1)
        hp.append((Xc - extrinsic[fi][:, 3]) @ extrinsic[fi][:, :3])
        hc.append(np.clip(v2["imgs"][j][ok], 0, 1)); hf.append(np.full(len(z_), fi))
        tsdf_in.append((fi, np.where(ok, d2a, 0).astype(np.float32), (fxc, fyc, cxc, cyc),
                        (np.clip(v2["imgs"][j], 0, 1) * 255).astype(np.uint8)))
    if hp:
        P_obj0 = np.concatenate(hp).astype(np.float64)
        COL_obj = np.concatenate(hc).astype(np.float64)
        FID_obj = np.concatenate(hf)
    else:
        if USE_HIRES:
            WARN.append("高解析推論沒有可用的點，改用第一次推論")
        P_obj0, COL_obj, FID_obj = P_obj1, np.clip(imgs[obj_px], 0, 1).astype(np.float64), np.nonzero(obj_px)[0]
        tsdf_in = [(i, np.where(obj_px[i], depth[i], 0).astype(np.float32),
                    (intrinsic[i][0, 0], intrinsic[i][1, 1], intrinsic[i][0, 2], intrinsic[i][1, 2]),
                    (np.clip(imgs[i], 0, 1) * 255).astype(np.uint8)) for i in range(len(depth)) if obj_px[i].any()]
    ok_mask = frame_ok & np.array([not np.all(~m) for m in MASKS])
    keepc = consistency_filter(P_obj0, FID_obj, extrinsic, intrinsic, depth, ok_mask, (padL, padT, nw, nh), tol=CONS_TOL)
    log(f"多視角一致性：保留 {keepc.mean():.0%} 的物體點")
    if keepc.mean() < 0.15:
        WARN.append("多視角一致性過低，可能是拍攝太快或光線太暗")
    else:
        P_obj0, COL_obj, FID_obj = P_obj0[keepc], COL_obj[keepc], FID_obj[keepc]
    src_P, src_F = P_obj0, FID_obj                       # 用原始點的來源影格決定法向量朝向
    tsdf_used = False
    if USE_TSDF and len(tsdf_in) >= 8:
        try:
            Pt, Ct, tvox, tmesh = tsdf_fuse(tsdf_in, extrinsic, P_obj0)
            if DEPTH_REFINE and len(tmesh.triangles) > 1000:
                tsdf_in = refine_depth_frames(tsdf_in, extrinsic, tmesh)
                Pt, Ct, tvox, tmesh = tsdf_fuse(tsdf_in, extrinsic, P_obj0)
            del tmesh
            if len(Pt) > 5000:
                sub = np.random.default_rng(1).choice(len(src_P), min(len(src_P), 2_000_000), replace=False)
                src_P, src_F = src_P[sub], src_F[sub]
                P_obj0, COL_obj = Pt, Ct
                FID_obj = np.zeros(len(Pt), int)
                tsdf_used = True
                log(f"TSDF 融合：{len(Pt):,} 點（格子 {tvox*1000:.2f} mm）")
            else:
                log("TSDF 點數不足"); WARN.append("TSDF 融合點數不足，改用原本的點雲")
        except Exception as e:
            log("TSDF 失敗：", e)
            WARN.append("TSDF 融合失敗，改用原本的點雲")
    del tsdf_in
    log(f"物體點 {len(P_obj0):,}，背景點 {len(P_bg0):,}")

    if TT:
        # ---- 5T. 轉盤模式：世界座標已是公尺；上方＝重力反方向；桌面高度＝比例尺所在高度 ----
        center = np.median(P_obj0, 0)
        h_obj = float(np.percentile(P_obj0 @ up_tt, 0.5))     # 物體最低處
        h0 = min(float(np.median(ruler_pts @ up_tt)), h_obj) if len(ruler_pts) else h_obj   # 比例尺上表面（筆之類較厚時會偏高）
        Ra2, o2 = align_transform(up_tt, -h0, center, Ccam[0])
        A = Ra2
        to_al = lambda P: 100.0 * (P - o2) @ Ra2.T
        Pb_tmp = to_al(P_obj0)
        size0 = np.percentile(Pb_tmp, 98, 0) - np.percentile(Pb_tmp, 2, 0)
        U = float(np.clip(size0.max() / 12.0, 0.5, 6.0))
        has_table = True
        del Pb_tmp
    else:
        if USE_ARKIT:
            # ---- 5. 用 ARKit 相機位置換算真實尺寸（公尺）與重力方向（相似轉換已在前面算好）----
            M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)     # ARKit y-up → z-up
            to_cm1 = lambda P: 100.0 * ((s1 * P @ R1.T + t1) @ M.T)
            Pb1, Pg1, C1 = to_cm1(P_obj0), to_cm1(P_bg0), to_cm1(Ccam)

            center = np.median(Pb1, 0)
            size0 = np.percentile(Pb1, 98, 0) - np.percentile(Pb1, 2, 0)
            U = float(np.clip(size0.max() / 12.0, 0.5, 6.0))           # 長度單位：小熊（12 cm）= 1
            Dcam = np.median(np.linalg.norm(C1 - center, axis=1))
            has_table = True
            try:
                n_pl, d_pl, frac = fit_table_plane(Pg1, center, C1, radius=0.6 * Dcam,
                                                   dist_thr=max(0.3, 0.006 * Dcam), up_prior=np.array([0, 0, 1.0]))
            except RuntimeError:
                has_table = False
                WARN.append("找不到桌面，以物體最低點為底")
                n_pl = np.array([0, 0, 1.0]); d_pl = -np.percentile(Pb1[:, 2], 1)
            Ra2, o2 = align_transform(n_pl, d_pl, center, C1[0])
            k_al = 100.0 * s1
            A = Ra2 @ M @ R1
            b_al = Ra2 @ (100.0 * M @ t1 - o2)
            to_al = lambda P: k_al * P @ A.T + b_al
        else:
            # ---- 5. 未使用 ARKit：桌面平面定上下（畫面「上方」當參考），尺寸之後由比例尺換算 ----
            center = np.median(P_obj0, 0)
            Dcam = np.median(np.linalg.norm(Ccam - center, axis=1))
            up_prior = -Rcam[:, 1, :].mean(0); up_prior /= np.linalg.norm(up_prior)
            has_table = True
            try:
                n_pl, d_pl, frac = fit_table_plane(P_bg0, center, Ccam, radius=0.6 * Dcam, dist_thr=0.006 * Dcam, up_prior=up_prior)
            except RuntimeError:
                has_table = False
                WARN.append("找不到桌面，以物體最低點為底")
                n_pl = up_prior; d_pl = -np.percentile(P_obj0 @ n_pl, 1)
            Ra2, o2 = align_transform(n_pl, d_pl, center, Ccam[0])
            A = Ra2
            to_al = lambda P: (P - o2) @ Ra2.T                       # 暫時的單位（VGGT），比例尺換算後變公分
            size0 = np.percentile(to_al(P_obj0), 98, 0) - np.percentile(to_al(P_obj0), 2, 0)
            U = float(size0.max() / 12.0)
    # ---- 比例尺（手機繞物體模式，選用）：三角測量比例尺兩端，換算或驗證 ARKit 的尺寸 ----
    scale_info = dict(scale_source="arkit" if not TT else "ruler", use_arkit=USE_ARKIT)
    marks_o = meta.get("scale_marks") or []
    if not TT and marks_o and (SCALE_SOURCE != "arkit" or not USE_ARKIT):
        sel_of_o = {frames[i]["file"]: k for k, i in enumerate(pick)}
        ms_ = [(sel_of_o[m["frame"]], np.array(m["p1"], float), np.array(m["p2"], float)) for m in marks_o if m["frame"] in sel_of_o]
        if len(ms_) >= 2:
            A_w, B_w, rel_res, ang_ = ruler_triangulate(ms_, extrinsic, intrinsic, sx, sy, padL, padT)
            L_al = float(np.linalg.norm(to_al(A_w[None])[0] - to_al(B_w[None])[0]))
            L_true = float(meta.get("scale_length_cm") or 15.0)
            f_r = L_true / L_al
            if USE_ARKIT:
                log(f"比例尺三角測量：{len(ms_)} 張，夾角 {ang_:.0f}°，交會誤差 {rel_res:.1%}；ARKit 量到 {L_al:.2f} cm（實際 {L_true:.2f}），差 {f_r - 1:+.1%}")
                scale_info.update(ruler_measured_by_arkit_cm=round(L_al, 2), ruler_vs_arkit=round(f_r - 1, 4))
            else:
                log(f"比例尺三角測量：{len(ms_)} 張，夾角 {ang_:.0f}°，交會誤差 {rel_res:.1%}；以比例尺 {L_true:.2f} cm 換算尺寸")
            if ang_ >= 8 and rel_res < 0.05:
                to_al0 = to_al
                to_al = lambda P, f=f_r, g=to_al0: f * g(P)
                scale_info["scale_source"] = "ruler"
                U = U * f_r
            else:
                WARN.append("比例尺標記的角度差太小或點不準，改用 ARKit 尺寸")
        else:
            WARN.append("比例尺至少要在 2 張畫面標記，改用 ARKit 尺寸")
        if not USE_ARKIT and scale_info["scale_source"] != "ruler":
            WARN.append("比例尺無法使用（標記角度差太小或點不準），模型沒有真實尺寸（相對比例）")
        if SCALE_SOURCE == "ruler" and scale_info["scale_source"] != "ruler":
            WARN.append("指定使用比例尺，但比例尺無法使用，改用 ARKit 尺寸")
    unitless = (TT and not marks) or (not TT and not USE_ARKIT and scale_info.get("scale_source") != "ruler")
    if unitless:                                          # 沒有真實尺寸：把物體最大邊設成 10（相對單位）
        ext0 = np.percentile(to_al(P_obj0), 98, 0) - np.percentile(to_al(P_obj0), 2, 0)
        f_u = 10.0 / max(float(ext0.max()), 1e-9)
        to_al_u = to_al
        to_al = lambda P, f=f_u, g=to_al_u: f * g(P)
        U = 10.0 / 12.0
        scale_info["scale_source"] = "none"
        log("沒有真實尺寸：以相對比例建模（物體最大邊 = 10）")
    Pc_b, Pc_g, Cc = to_al(P_obj0), to_al(P_bg0), to_al(Ccam)
    Pc_src = to_al(src_P)
    del P_obj0, P_bg0

    try:
        save_camera_diagnostic(Cc, Pc_b, Pc_g, f"{WORK}/diag_cameras.png")
    except Exception as e:
        log("相機診斷圖失敗：", e)

    # ---- 6. 網格 ----
    mesh, bp, info, vox = build_mesh(Pc_b, COL_obj, FID_obj, Pc_g, COL_bg, Cc, U, has_table, Pc_src, src_F)
    is_obj = np.asarray(mesh.vertices)[:, 2] > 1.5 * vox
    cols = np.asarray(mesh.vertex_colors).copy()
    cols[is_obj] = color_from_frames(mesh, np.nonzero(is_obj)[0], frame_paths, MASKS, intrinsic, Rcam, A,
                                     Cc, sx, sy, padL, padT, sc[pick], cols[is_obj])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    log("頂點上色完成")

    # ---- 7. 貼圖 ----
    tex_pack = None
    tm_ = np.ones(len(Pc_g), bool) if info.get("has_base") else np.zeros(len(Pc_g), bool)
    tm_ &= (np.abs(Pc_g[:, 2]) < 0.3 * U) & (np.linalg.norm(Pc_g[:, :2] - np.median(bp[:, :2], 0), axis=1) < 3 * np.ptp(bp[:, :2], axis=0).max() + 5)
    table_xy, table_col = Pc_g[tm_][:, :2].copy(), COL_bg[tm_].copy()
    # 釋放大型陣列，留記憶體給貼圖步驟
    del Pc_g, COL_bg, Pc_b, v2, conf, imgs, depth, BMASK, BDIL, valid, bg_px, obj_px
    import gc; gc.collect()
    try:
        tree_g = cKDTree(table_xy) if len(table_xy) > 50 else None
        base_side = np.median(table_col, 0) * 0.85 if len(table_col) > 50 else np.full(3, 0.6)

        def color_fn(P, Nn):
            out = np.zeros((len(P), 3))
            ob = P[:, 2] > 1.5 * vox
            if ob.any():
                fb = np.asarray(mesh.vertex_colors)[cKDTree(np.asarray(mesh.vertices)).query(P[ob], k=1)[1]]
                out[ob], _ = project_colors(P[ob], Nn[ob], mesh, frame_paths, MASKS, intrinsic, Rcam, A,
                                            Cc, sx, sy, padL, padT, sc[pick], fb)
            top = ~ob & (P[:, 2] > -0.75 * vox)
            if tree_g is not None and top.any():
                out[top] = table_col[tree_g.query(P[top][:, :2], k=8)[1]].mean(1)
            out[~ob & ~top] = base_side
            return out

        if USE_TEXTURE:
            tex_pack = bake_texture(mesh, color_fn, tex_size=TEXTURE_SIZE)
            log(f"貼圖完成（{TEXTURE_SIZE}×{TEXTURE_SIZE}，{len(tex_pack[1]):,} 面）")
        else:
            log("已在開發者參數中關閉貼圖，改用頂點顏色")
    except Exception as e:
        traceback.print_exc()
        WARN.append(f"貼圖失敗，改用頂點顏色（{str(e)[:80]}）")

    # ---- 8. 匯出 ----
    export_all(mesh, tex_pack)
    # ---- 9. 3DGS 擬真模型（選用）----
    gs_info = {}
    if USE_GS:
        try:
            try:
                import gsplat  # noqa
            except ImportError:
                sh("pip -q install ninja gsplat")
            import gsplat
            log("gsplat", gsplat.__version__, "（第一次執行會編譯 CUDA 程式，約 5~10 分鐘）")
            ok_ids = [i for i in range(len(frame_paths)) if MASKS[i].any()]
            b0 = to_al(np.zeros((1, 3)))[0]
            Alin = np.stack([to_al(np.eye(3)[k:k + 1])[0] - b0 for k in range(3)], 1)   # 轉正座標 = Alin·X + b0
            c_ = float(np.cbrt(np.linalg.det(Alin))); Rot = Alin / c_
            ds = min(1.0, GS_RES / max(W0, H0))
            VMs, KSs = [], []
            for i in ok_ids:
                Rv = extrinsic[i][:, :3].astype(np.float64); tv = extrinsic[i][:, 3].astype(np.float64)
                V4 = np.eye(4); V4[:3, :3] = Rv @ Rot.T; V4[:3, 3] = c_ * tv - Rv @ Rot.T @ b0
                Ki = intrinsic[i].astype(np.float64)
                cxo = (Ki[0, 2] - padL + 0.5) / sx - 0.5; cyo = (Ki[1, 2] - padT + 0.5) / sy - 0.5
                KSs.append(np.array([[Ki[0, 0] / sx * ds, 0, (cxo + 0.5) * ds - 0.5], [0, Ki[1, 1] / sy * ds, (cyo + 0.5) * ds - 0.5], [0, 0, 1]]))
                VMs.append(V4)
            # 初始點：網格上物體部分的表面取樣（含顏色）
            obj_m = o3d.geometry.TriangleMesh(mesh)
            Vm = np.asarray(obj_m.vertices); Fm = np.asarray(obj_m.triangles)
            obj_m.remove_triangles_by_mask((Vm[Fm][:, :, 2] <= 1.5 * vox).all(1)); obj_m.remove_unreferenced_vertices()
            spc = obj_m.sample_points_uniformly(200_000)
            P0, C0 = np.asarray(spc.points), np.clip(np.asarray(spc.colors), 0, 1)
            log(f"3DGS 訓練：{len(ok_ids)} 張影像（{int(W0 * ds)}×{int(H0 * ds)}），{GS_ITERS} 步，初始 {len(P0):,} 點")
            g = train_3dgs([frame_paths[i] for i in ok_ids], [MASKS[i] for i in ok_ids], VMs, KSs, W0, H0, ds, P0, C0, GS_ITERS)
            if GS_CLEAN:
                g = clean_gaussians(g, VMs, KSs, [MASKS[i] for i in ok_ids], W0, H0, ds, GS_MIN_OPACITY)
            n_out = export_gaussians(g, GS_MAX, f"{WORK}/gaussians.ply", f"{WORK}/model.splat")
            gs_info = dict(gs=True, gs_count=int(n_out), gs_mb=round(os.path.getsize(f"{WORK}/model.splat") / 1e6, 1))
            log(f"3DGS 完成：{n_out:,} 個元素，model.splat {gs_info['gs_mb']} MB")
        except Exception as e:
            traceback.print_exc()
            WARN.append(f"3DGS 失敗（網格模型仍可使用）：{str(e)[:120]}")
            gs_info = dict(gs=False)

    info.update(tsdf=tsdf_used, textured=tex_pack is not None,
                params_used=PARAMS_USED,
                params_summary=("、".join(f"{k}={v}" for k, v in PARAMS_USED.items()) or "全部預設"))
    ext = bp.max(0) - bp.min(0)
    info.update(mode=meta.get("mode") or "orbit", **scale_info, **gs_info)
    info.update(target_method=target_method, prompt_frames=len(prompts), bad_pose_frames=len(bad_pose),
                batches=int(NB), smooth_level=SMOOTH_LEVEL,
                consistency_keep=round(float(keepc.mean()), 3),
                mask_area_median=round(med, 4), bad_mask_frames=len(bad))
    write_meta(status="ok", size_cm=[round(float(ext[0]), 1), round(float(ext[1]), 1), round(float(bp[:, 2].max()), 1)],
               frames_used=len(frame_paths), scale_residual_cm=round(rms_cm, 2), **info)
    log("全部完成")


def build_mesh(Pc_b, COL_b, FID_b, Pc_g, COL_g, Cc, U, has_table, src_P=None, src_F=None):
    T_BASE = max(0.5 * U, 0.2)
    SINK = min(0.15 * U, T_BASE / 2)
    VOX = max(0.05, 0.05 * U)
    keep = Pc_b[:, 2] > 0.15 * U
    P_, C_, F_id = Pc_b[keep], COL_b[keep], FID_b[keep]
    coarse = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_)).voxel_down_sample(0.3 * U)
    lab = np.array(coarse.cluster_dbscan(eps=0.9 * U, min_points=5))
    if (lab >= 0).sum() == 0:
        raise RuntimeError("物體點太少，無法重建")
    main_c = np.asarray(coarse.points)[lab == np.bincount(lab[lab >= 0]).argmax()]
    near_ = cKDTree(main_c).query(P_, k=1)[0] < 0.45 * U
    P_, C_, F_id = P_[near_], C_[near_], F_id[near_]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_))
    pcd.colors = o3d.utility.Vector3dVector(C_)
    pcd = pcd.voxel_down_sample(VOX)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.4 * U, 6 * VOX), max_nn=40))
    if src_P is None:
        sub_ = np.random.default_rng(0).choice(len(P_), min(len(P_), 2_000_000), replace=False)
        src_P, src_F = P_[sub_], F_id[sub_]
    pcd = orient_normals_by_source_cam(pcd, src_P, Cc[src_F])
    bp = np.asarray(pcd.points)
    log(f"物體點雲 {len(bp):,} 點")

    zmin = bp[:, 2].min()
    fm = bp[:, 2] < zmin + 0.6 * U
    levels = np.arange(zmin, -0.45 * U, -0.2 * U)
    foot_pts = np.vstack([np.c_[bp[fm][:, :2], np.full(fm.sum(), z_)] for z_ in levels]) if len(levels) else np.zeros((0, 3))
    nb_ = np.asarray(pcd.normals)[fm].copy(); nb_[:, 2] = 0
    ln_ = np.linalg.norm(nb_, axis=1, keepdims=True)
    nb_ = np.where(ln_ > 0.2, nb_ / np.maximum(ln_, 1e-9), [0, 0, -1.0])
    pcd_p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([bp, foot_pts])))
    pcd_p.normals = o3d.utility.Vector3dVector(np.vstack([np.asarray(pcd.normals), np.tile(nb_, (len(levels), 1))]))

    pm = poisson_mesh(pcd_p, POISSON_DEPTH, 0, None, None, scale=1.5, crop=False)
    log(f"Poisson 完成：{len(pm.triangles):,} 面")

    bpx = bp[:, :2]
    cxy = (bpx.min(0) + bpx.max(0)) / 2
    R_base = np.linalg.norm(bpx - cxy, axis=1).max() + (DEV_BASE_MARGIN if DEV_BASE_MARGIN > 0 else max(1.0, 0.15 * np.ptp(bpx, axis=0).max()))
    tm = (np.abs(Pc_g[:, 2]) < 0.3 * U) & (np.linalg.norm(Pc_g[:, :2] - cxy, axis=1) < R_base + 2 * U)
    use_base = has_table and USE_BASE and (tm.sum() > 200 or len(Pc_g) == 0)
    margin = 0.3 * U
    solid_target = DEV_SOLID_MM / 10.0 if DEV_SOLID_MM > 0 else SOLID_VOXEL * U
    mesh, vox = voxel_solid(pm, solid_target, base=(cxy, R_base, T_BASE) if use_base else None, sink=SINK,
                            clip_lo=bp.min(0) - margin, clip_hi=bp.max(0) + margin,
                            blur=SMOOTH_BLUR, taubin=SMOOTH_TAUBIN)
    if SMOOTH_ITERS > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=SMOOTH_ITERS)
        mesh.compute_vertex_normals()
    V3 = np.asarray(mesh.vertices); F3 = np.asarray(mesh.triangles)
    log(f"封閉實體完成：{len(F3):,} 面（體素 {vox*10:.2f} mm）")

    cols = np.zeros((len(V3), 3))
    is_obj = V3[:, 2] > 1.5 * vox
    if use_base:
        top = ~is_obj & (V3[:, 2] > -0.75 * vox)
        if tm.sum() > 0:
            _, ti = cKDTree(Pc_g[tm][:, :2]).query(V3[top][:, :2], k=8)
            cols[top] = COL_g[tm][ti].mean(1)
            cols[~is_obj & ~top] = np.median(COL_g[tm], 0) * 0.85
        else:                                             # 沒有桌面點（轉盤模式）：底座用淺灰色
            cols[top] = 0.78
            cols[~is_obj & ~top] = 0.66
    else:
        cols[~is_obj] = 0.7
    _, fb = cKDTree(bp).query(V3[is_obj], k=4)
    cols[is_obj] = np.asarray(pcd.colors)[fb].mean(1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))

    import trimesh
    wt = trimesh.Trimesh(V3, F3, process=False)
    info = dict(watertight=bool(wt.is_watertight), bodies=int(wt.body_count),
                volume_cm3=round(float(abs(wt.volume)), 1) if wt.is_watertight else None,
                triangles=int(len(F3)), has_base=bool(use_base), solid_voxel_mm=round(vox * 10, 2))
    return mesh, bp, info, vox


def export_all(mesh, tex_pack=None):
    import trimesh
    from PIL import Image as PILImage
    V = np.asarray(mesh.vertices); F_ = np.asarray(mesh.triangles)
    VC = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    trimesh.Trimesh(V * 10, F_, vertex_colors=VC, process=False).export(f"{WORK}/model_mm.ply")
    trimesh.Trimesh(V * 10, F_, process=False).export(f"{WORK}/model_mm.stl")
    yup = lambda X: np.stack([X[:, 0], X[:, 2], -X[:, 1]], 1)
    if tex_pack is not None:
        V2, Fi, UV, tex8, _ = tex_pack
        img = PILImage.fromarray(tex8)
        img.save(f"{WORK}/preview_tex.jpg", quality=90)
        vis = trimesh.visual.TextureVisuals(uv=np.c_[UV[:, 0], 1 - UV[:, 1]], image=img)   # trimesh 的 uv 以左下為原點
        trimesh.Trimesh(yup(V2 * 0.01), Fi, visual=vis, process=False).export(f"{WORK}/model.glb")
        mt = trimesh.Trimesh(V2, Fi, process=False)
        write_preview_textured(V2, Fi, UV, np.asarray(mt.vertex_normals), f"{WORK}/preview.bin")
    else:
        trimesh.Trimesh(yup(V * 0.01), F_, vertex_colors=VC, process=False).export(f"{WORK}/model.glb")
        prev = mesh.simplify_quadric_decimation(150_000) if len(F_) > 150_000 else mesh
        prev.compute_vertex_normals()
        write_preview(prev, f"{WORK}/preview.bin")
    log("匯出完成：", sorted(os.listdir(WORK)))


def write_preview_textured(V, F_, UV, N, path):
    """格式：'O3DT' + u32 頂點數 + u32 三角面數 + f32 位置 + f32 法向量 + f32 uv（v 向下）+ u32 索引；貼圖為 preview_tex.jpg"""
    yup = lambda X: np.stack([X[:, 0], X[:, 2], -X[:, 1]], 1)
    Vy = yup(np.asarray(V) * 0.01).astype("<f4")
    Ny = yup(np.asarray(N)).astype("<f4")
    with open(path, "wb") as f:
        f.write(b"O3DT")
        f.write(np.array([len(Vy), len(F_)], "<u4").tobytes())
        for arr in (Vy, Ny, np.asarray(UV, "<f4"), np.asarray(F_, "<u4")):
            f.write(np.ascontiguousarray(arr).tobytes())


def write_preview(m, path):
    """格式：'O3DP' + u32 頂點數 + u32 三角面數 + f32 位置 + f32 法向量 + f32 顏色 + u32 索引（little-endian）"""
    V = np.asarray(m.vertices) * 0.01
    V = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1).astype("<f4")
    N = np.asarray(m.vertex_normals)
    N = np.stack([N[:, 0], N[:, 2], -N[:, 1]], 1).astype("<f4")
    C = (np.asarray(m.vertex_colors) if m.has_vertex_colors() else np.full((len(V), 3), 0.7)).astype("<f4")
    F_ = np.asarray(m.triangles).astype("<u4")
    with open(path, "wb") as f:
        f.write(b"O3DP")
        f.write(np.array([len(V), len(F_)], "<u4").tobytes())
        for arr in (V, N, C, F_):
            f.write(np.ascontiguousarray(arr).tobytes())


if not NO_RUN:
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        write_meta(status="error", error="exception", message=str(e)[:500])
