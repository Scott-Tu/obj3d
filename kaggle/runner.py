# =====================================================================
#  obj3d Kaggle runner：由 iPhone App 自動上傳並執行
#  輸入：Kaggle Dataset 中的拍攝資料（影格 + ARKit 相機位置）
#  輸出：/kaggle/working/ 下的 model.glb、model_mm.stl、model_mm.ply、
#        preview.bin（App 預覽用）、result_meta.json
# =====================================================================
JOB_ID = "__JOB_ID__"
RUNNER_VERSION = "2026.10.05-devparams"   # 每次修改運算程式時更新
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

# ---- App「開發者參數」：由 App 填入 JSON；0 或缺少 = 自動 ----
PARAMS_JSON = r"""__PARAMS__"""
try:
    PARAMS = json.loads(PARAMS_JSON) if not PARAMS_JSON.startswith("__") else {}
except Exception:
    PARAMS = {}
DEV_SOLID_MM = float(PARAMS.get("solid_voxel_mm") or 0)      # 封閉實體的體素大小（mm）
DEV_TRUNC_MM = float(PARAMS.get("tsdf_trunc_mm") or 0)       # TSDF 截斷距離（mm）
if int(PARAMS.get("batches") or 0) > 0:
    MAX_BATCHES = int(PARAMS["batches"])
if "use_tsdf" in PARAMS:
    USE_TSDF = bool(PARAMS["use_tsdf"])
USE_TEXTURE = bool(PARAMS.get("use_texture", True))
CONF_DROP_PERCENT = 40
HR_TOL = 0.04
MASK_ERODE_PX = 4
MAX_FRAMES = 0          # 0 = 依 GPU 記憶體自動決定
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
    filled = (tri_id >= 0).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(8):                                     # 往外擴幾圈，避免貼圖接縫露出黑邊
        grown = cv2.dilate(filled, k3)
        ring = (grown > 0) & (filled == 0)
        if not ring.any():
            break
        blur = cv2.blur(tex * filled[..., None], (3, 3))
        cnt = cv2.blur(filled.astype(np.float32), (3, 3))
        tex[ring] = blur[ring] / np.maximum(cnt[ring, None], 1e-6)
        filled = grown
    tex8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    return V2, Fi.astype(np.int32), UV.astype(np.float32), tex8, m


def voxel_solid(mesh, vox, close_iters=2, base=None, sink=0.15, min_comp=0.05, max_voxels=40e6, clip_lo=None, clip_hi=None,
                blur=0.8, taubin=5):
    """把表面網格轉成「保證封閉」的實體：
    1) 整個網格體素化並填滿內部（Poisson 未修剪的網格本身就是封閉的，填得起來）
    2) 用體積的方式裁掉資料範圍外的部分與桌面以下（裁切後仍是實心，不會開洞）
    3) 加上實心底座，Marching Cubes 轉回表面。base = (中心 xy, 半徑, 厚度) 或 None"""
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
    occ = ndimage.binary_fill_holes(occ)
    occ = ndimage.binary_erosion(occ, structure=st, iterations=1)
    xc = lo[0] + (np.arange(shape[0]) + 0.5) * vox
    yc = lo[1] + (np.arange(shape[1]) + 0.5) * vox
    zc = lo[2] + (np.arange(shape[2]) + 0.5) * vox
    if clip_lo is not None:                                    # 體積裁切：資料範圍外的 Poisson 外插部分
        occ &= ((xc >= clip_lo[0]) & (xc <= clip_hi[0]))[:, None, None]
        occ &= ((yc >= clip_lo[1]) & (yc <= clip_hi[1]))[None, :, None]
        occ &= (zc <= clip_hi[2])[None, None, :]
    occ[:, :, zc < -sink] = False
    if base is not None:
        disk = ((xc[:, None] - cxy[0]) ** 2 + (yc[None, :] - cxy[1]) ** 2) <= R * R
        zsel = (zc >= -T) & (zc <= 0)
        occ[:, :, zc < 0] = False
        occ[:, :, zsel] |= disk[:, :, None]
    else:
        occ[:, :, zc < 0] = False
    lab, nlab = ndimage.label(occ)
    if nlab > 1:
        sizes = ndimage.sum(occ, lab, range(1, nlab + 1))
        occ = np.isin(lab, 1 + np.nonzero(sizes >= min_comp * sizes.max())[0])
    field = ndimage.gaussian_filter(np.pad(occ, 3).astype(np.float32), blur)   # 平滑後的等值面不會有非流形邊
    v, f, _, _ = measure.marching_cubes(field, level=0.5)
    V = lo + (v - 3 + 0.5) * vox
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(f.astype(np.int32)))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    tc, cnt, _ = m.cluster_connected_triangles()
    tc = np.asarray(tc); cnt = np.asarray(cnt)
    if len(cnt) > 1:
        m.remove_triangles_by_mask(~(cnt >= 0.01 * cnt.max())[tc]); m.remove_unreferenced_vertices()
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
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    masks = np.zeros((n, H0, W0), bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        st = pred.init_state(sel_dir, offload_video_to_cpu=True)
        for fi, (x, y) in prompts.items():
            pred.add_new_points_or_box(st, frame_idx=int(fi), obj_id=1,
                                       points=np.array([[x, y]], np.float32),
                                       labels=np.array([1], np.int32))
        for rev in (False, True):
            for fidx, _, lg in pred.propagate_in_video(st, reverse=rev):
                masks[fidx] |= (lg[0, 0] > 0).cpu().numpy()
    del pred, st
    torch.cuda.empty_cache()
    return masks


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
    def cloud(b):
        P = batch_points(np.nonzero(batch_of == b)[0], ext, Ks, depth, conf, mask)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.002)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        return pc
    ref = cloud(0)
    if len(ref.points) < 500:
        return ext
    for b in range(1, B):
        src = cloud(b)
        if len(src.points) < 500:
            continue
        reg = o3d.pipelines.registration.registration_icp(
            src, ref, 0.006, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = reg.transformation
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        mv = np.linalg.norm(T[:3, 3])
        log(f"ICP 第 {b+1} 批：吻合度 {reg.fitness:.2f}，修正 {mv*100:.2f} cm / {ang:.2f}°")
        if reg.fitness < 0.3 or mv > 0.02 or ang > 4:
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
    return np.asarray(pc.points), np.asarray(pc.colors), vox


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
    span = np.ptp(C_ar_all, axis=0).max()
    if span < 0.15:
        raise RuntimeError(f"手機移動範圍太小（{span*100:.0f} cm），請繞著物體走一圈")

    # ---- 1. 挑影格 ----
    gb = gpu_memory_gb()
    N_batch = int(np.clip((gb - 3.5) / 0.23, 16, 150))
    N = MAX_FRAMES or N_batch * MAX_BATCHES
    N = min(N, len(paths))
    sc = np.array([sharpness(p) for p in paths])
    pick = [int(b[np.argmax(sc[b])]) for b in np.array_split(np.arange(len(paths)), N) if len(b)]
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

    # ---- 2. VGGT 第一次 + SAM 2 ----
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

    # ---- 3. 高解析第二次推論（只裁物體附近）----
    HR = f"{TMP}/crops"
    shutil.rmtree(HR, ignore_errors=True); os.makedirs(HR)
    crop_info, hr_paths, hr_fids = {}, [], []
    for fi, m in enumerate(MASKS):
        ys, xs = np.nonzero(m)
        if len(xs) < 500:
            continue
        side = int(min(max(np.ptp(xs), np.ptp(ys)) * 1.3 + 20, W0, H0))
        x0 = int(np.clip((xs.min() + xs.max()) / 2 - side / 2, 0, W0 - side))
        y0 = int(np.clip((ys.min() + ys.max()) / 2 - side / 2, 0, H0 - side))
        p = f"{HR}/{fi:03d}.png"
        Image.open(frame_paths[fi]).convert("RGB").crop((x0, y0, x0 + side, y0 + side)).resize((518, 518), Image.BICUBIC).save(p)
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
        WARN.append("高解析推論沒有可用的點，改用第一次推論")
        P_obj0, COL_obj, FID_obj = P_obj1, np.clip(imgs[obj_px], 0, 1).astype(np.float64), np.nonzero(obj_px)[0]
    ok_mask = frame_ok & np.array([not np.all(~m) for m in MASKS])
    keepc = consistency_filter(P_obj0, FID_obj, extrinsic, intrinsic, depth, ok_mask, (padL, padT, nw, nh))
    log(f"多視角一致性：保留 {keepc.mean():.0%} 的物體點")
    if keepc.mean() < 0.15:
        WARN.append("多視角一致性過低，可能是拍攝太快或光線太暗")
    else:
        P_obj0, COL_obj, FID_obj = P_obj0[keepc], COL_obj[keepc], FID_obj[keepc]
    src_P, src_F = P_obj0, FID_obj                       # 用原始點的來源影格決定法向量朝向
    tsdf_used = False
    if USE_TSDF and len(tsdf_in) >= 8:
        try:
            Pt, Ct, tvox = tsdf_fuse(tsdf_in, extrinsic, P_obj0)
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

        if not USE_TEXTURE:
            raise RuntimeError("已在開發者參數中關閉貼圖")
        tex_pack = bake_texture(mesh, color_fn, tex_size=TEXTURE_SIZE)
        log(f"貼圖完成（{TEXTURE_SIZE}×{TEXTURE_SIZE}，{len(tex_pack[1]):,} 面）")
    except Exception as e:
        traceback.print_exc()
        WARN.append(f"貼圖失敗，改用頂點顏色（{str(e)[:80]}）")

    # ---- 8. 匯出 ----
    export_all(mesh, tex_pack)
    info.update(tsdf=tsdf_used, textured=tex_pack is not None,
                params_summary=f"實體 {DEV_SOLID_MM or '自動'} mm、截斷 {DEV_TRUNC_MM or '自動'} mm、"
                               f"批次 {MAX_BATCHES}、TSDF {'開' if USE_TSDF else '關'}、貼圖 {'開' if USE_TEXTURE else '關'}")
    ext = bp.max(0) - bp.min(0)
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
    R_base = np.linalg.norm(bpx - cxy, axis=1).max() + max(1.0, 0.15 * np.ptp(bpx, axis=0).max())
    tm = (np.abs(Pc_g[:, 2]) < 0.3 * U) & (np.linalg.norm(Pc_g[:, :2] - cxy, axis=1) < R_base + 2 * U)
    use_base = has_table and tm.sum() > 200
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
        _, ti = cKDTree(Pc_g[tm][:, :2]).query(V3[top][:, :2], k=8)
        cols[top] = COL_g[tm][ti].mean(1)
        cols[~is_obj & ~top] = np.median(COL_g[tm], 0) * 0.85
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
