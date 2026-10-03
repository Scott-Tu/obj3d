# =====================================================================
#  obj3d Kaggle runner：由 iPhone App 自動上傳並執行
#  輸入：Kaggle Dataset 中的拍攝資料（影格 + ARKit 相機位置）
#  輸出：/kaggle/working/ 下的 model.glb、model_mm.stl、model_mm.ply、
#        preview.bin（App 預覽用）、result_meta.json
# =====================================================================
JOB_ID = "__JOB_ID__"

import os, sys, json, time, glob, shutil, subprocess, zipfile, traceback
T0 = time.time()
WORK = "/kaggle/working"
TMP = "/tmp/obj3d"
WARN = []

POISSON_DEPTH = 9
SMOOTH_ITERS = 3
CONF_DROP_PERCENT = 40
HR_TOL = 0.04
MASK_ERODE_PX = 4
MAX_FRAMES = 0          # 0 = 依 GPU 記憶體自動決定


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def write_meta(**kw):
    kw.setdefault("jobId", JOB_ID)
    kw["elapsed_s"] = round(time.time() - T0, 1)
    kw["warnings"] = WARN
    with open(f"{WORK}/result_meta.json", "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2)


def sh(cmd):
    log("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def install():
    sh("pip -q install open3d trimesh pymeshfix manifold3d einops safetensors huggingface_hub")
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


def poisson_mesh(pcd, depth, trim_q, bbox_min, bbox_max, keep_largest=False):
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=1.1, linear_fit=False)
    dens = np.asarray(dens)
    if trim_q > 0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_q))
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max))
    tri_clusters, counts, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters); counts = np.asarray(counts)
    if len(counts):
        keep = (np.arange(len(counts)) == counts.argmax()) if keep_largest else counts >= max(200, 0.02 * counts.max())
        mesh.remove_triangles_by_mask(~keep[tri_clusters])
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def color_from_frames(mesh, vidx, frame_paths, masks, intrinsic, Rcam, Ra, Cc, sx, sy, padL, padT,
                      sharp, fallback, topk=3, min_cos=0.2):
    """從原始解析度影片幀投影取色：遮擋判斷 + 取「最正面、最清晰」的前 topk 個視角加權平均"""
    import cv2
    V = np.asarray(mesh.vertices)[vidx]
    N = np.asarray(mesh.vertex_normals)[vidx]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    best_w = np.zeros((len(V), topk), np.float32)
    best_c = np.zeros((len(V), topk, 3), np.float32)
    sharp = np.asarray(sharp, float); sharp = np.sqrt(sharp / np.median(sharp))
    kd = np.ones((3, 3), np.uint8)
    for fi, path in enumerate(frame_paths):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        H0, W0 = img.shape[:2]
        K = intrinsic[fi]
        fx, fy = K[0, 0] / sx, K[1, 1] / sy
        cx, cy = (K[0, 2] - padL + 0.5) / sx - 0.5, (K[1, 2] - padT + 0.5) / sy - 0.5
        R2 = Rcam[fi] @ Ra.T                     # aligned(cm) → camera 的旋轉
        D = V - Cc[fi]
        dist = np.linalg.norm(D, axis=1)
        Xc = D @ R2.T
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy
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
        vis = t > dist[idx] - 0.08
        idx = idx[vis]
        if len(idx) == 0:
            continue
        n = len(idx); w_ = int(np.ceil(n / 4096)) * 4096
        mu = np.zeros(w_, np.float32); mv = np.zeros(w_, np.float32)
        mu[:n], mv[:n] = u[idx], v[idx]
        col = cv2.remap(img, mu.reshape(-1, 4096), mv.reshape(-1, 4096), cv2.INTER_LINEAR).reshape(-1, 3)[:n]
        w = (cosv[idx] ** 2 / dist[idx] * sharp[fi]).astype(np.float32)
        # 插入 top-k
        allw = np.concatenate([best_w[idx], w[:, None]], 1)
        allc = np.concatenate([best_c[idx], col[:, None]], 1)
        order = np.argsort(-allw, 1)[:, :topk]
        best_w[idx] = np.take_along_axis(allw, order, 1)
        best_c[idx] = np.take_along_axis(allc, order[..., None], 1)
    ws = best_w.sum(1)
    out = np.array(fallback, dtype=np.float64)
    has = ws > 0
    out[has] = (best_c[has] * best_w[has, :, None]).sum(1) / ws[has, None]
    print(f"影像取色：{has.mean():.1%} 的物體頂點有可見視角")
    return out



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


def run_sam(sel_dir, n, W0, H0):
    """自動提示：物體在畫面中央（App 拍攝時有十字準心）"""
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    masks = np.zeros((n, H0, W0), bool)
    pf = sorted(set(np.linspace(0, n - 1, 4).astype(int).tolist()))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        st = pred.init_state(sel_dir, offload_video_to_cpu=True)
        for fi in pf:
            pred.add_new_points_or_box(st, frame_idx=int(fi), obj_id=1,
                                       points=np.array([[W0 / 2, H0 / 2]], np.float32),
                                       labels=np.array([1], np.int32))
        for rev in (False, True):
            for fidx, _, lg in pred.propagate_in_video(st, reverse=rev):
                masks[fidx] |= (lg[0, 0] > 0).cpu().numpy()
    del pred, st
    torch.cuda.empty_cache()
    return masks


def gpu_memory_gb():
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 1e9


# =====================================================================
#  主流程
# =====================================================================
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
    N = MAX_FRAMES or int(np.clip((gb - 3.5) / 0.23, 16, 150))
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
    v1 = run_vggt(frame_paths, "pad", True)
    extrinsic, intrinsic, depth, conf, imgs = v1["extrinsic"], v1["intrinsic"], v1["depth"], v1["conf"], v1["imgs"]
    log("VGGT 第一次完成")
    MASKS = run_sam(SEL, len(frame_paths), W0, H0)
    area = MASKS.reshape(len(MASKS), -1).mean(1)
    log(f"SAM 2 完成，遮罩面積中位數 {np.median(area):.1%}")
    if np.median(area) < 0.005:
        raise RuntimeError("畫面中央找不到物體：拍攝時請讓物體一直在十字準心上")
    if np.median(area) > 0.8:
        WARN.append("遮罩幾乎佔滿畫面，可能選到背景")

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
    v2 = run_vggt(hr_paths, "crop", False)
    log("VGGT 第二次完成")

    # ---- 4. 點雲（VGGT 座標）----
    S_, H, W = depth.shape
    k3 = np.ones((3, 3), np.uint8)
    BMASK = np.zeros(depth.shape, bool); BDIL = np.zeros(depth.shape, bool)
    for i, m in enumerate(MASKS):
        m8 = m.astype(np.uint8)
        BMASK[i, padT:padT + nh, padL:padL + nw] = cv2.resize(cv2.erode(m8, k3, iterations=MASK_ERODE_PX), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        BDIL[i, padT:padT + nh, padL:padL + nw] = cv2.resize(cv2.dilate(m8, k3, iterations=12), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    u_, v_ = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    cam_pts = np.stack([(u_ - cx) * depth / fx, (v_ - cy) * depth / fy, depth], -1)
    world = np.einsum('shwj,sjk->shwk', cam_pts - extrinsic[:, None, None, :, 3], extrinsic[:, :, :3])
    del cam_pts
    valid = np.zeros(depth.shape, bool)
    valid[:, padT + 2:padT + nh - 2, padL + 2:padL + nw - 2] = True
    valid &= depth > 1e-6
    bg_px = valid & ~BDIL
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
    if hp:
        P_obj0 = np.concatenate(hp).astype(np.float64)
        COL_obj = np.concatenate(hc).astype(np.float64)
        FID_obj = np.concatenate(hf)
    else:
        WARN.append("高解析推論沒有可用的點，改用第一次推論")
        P_obj0, COL_obj, FID_obj = P_obj1, np.clip(imgs[obj_px], 0, 1).astype(np.float64), np.nonzero(obj_px)[0]
    log(f"物體點 {len(P_obj0):,}，背景點 {len(P_bg0):,}")

    # ---- 5. 用 ARKit 相機位置換算真實尺寸（公尺）與重力方向 ----
    Ccam, fwd, Rcam = cam_centers_dirs(extrinsic)
    s1, R1, t1 = umeyama(Ccam, C_ar)
    res = np.linalg.norm((s1 * Ccam @ R1.T + t1) - C_ar, axis=1)
    keep = res <= np.percentile(res, 80)
    s1, R1, t1 = umeyama(Ccam[keep], C_ar[keep])
    res = np.linalg.norm((s1 * Ccam @ R1.T + t1) - C_ar, axis=1)
    rms_cm = float(np.sqrt(np.mean(res[keep] ** 2)) * 100)
    log(f"ARKit 對齊：殘差 {rms_cm:.2f} cm")
    if rms_cm > 3:
        WARN.append(f"相機軌跡對齊誤差偏大（{rms_cm:.1f} cm），尺寸可能不準")
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
    del P_obj0, P_bg0

    # ---- 6. 網格 ----
    mesh, bp, info = build_mesh(Pc_b, COL_obj, FID_obj, Pc_g, COL_bg, Cc, U, has_table)
    is_obj = np.asarray(mesh.vertices)[:, 2] > 0.01 * U
    cols = np.asarray(mesh.vertex_colors).copy()
    cols[is_obj] = color_from_frames(mesh, np.nonzero(is_obj)[0], frame_paths, MASKS, intrinsic, Rcam, A,
                                     Cc, sx, sy, padL, padT, sc[pick], cols[is_obj])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    log("上色完成")

    # ---- 7. 匯出 ----
    export_all(mesh)
    ext = bp.max(0) - bp.min(0)
    write_meta(status="ok", size_cm=[round(float(ext[0]), 1), round(float(ext[1]), 1), round(float(bp[:, 2].max()), 1)],
               frames_used=len(frame_paths), scale_residual_cm=round(rms_cm, 2), **info)
    log("全部完成")


def build_mesh(Pc_b, COL_b, FID_b, Pc_g, COL_g, Cc, U, has_table):
    import pymeshfix, manifold3d as m3d
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
    sub_ = np.random.default_rng(0).choice(len(P_), min(len(P_), 2_000_000), replace=False)
    pcd = orient_normals_by_source_cam(pcd, P_[sub_], Cc[F_id[sub_]])
    bp = np.asarray(pcd.points)
    log(f"物體點雲 {len(bp):,} 點")

    zmin = bp[:, 2].min()
    fm = bp[:, 2] < zmin + 0.6 * U
    foot = bp[fm]
    levels = np.arange(zmin, -0.45 * U, -0.2 * U)
    foot_pts = np.vstack([np.c_[foot[:, :2], np.full(len(foot), z_)] for z_ in levels]) if len(levels) else np.zeros((0, 3))
    nb_ = np.asarray(pcd.normals)[fm].copy(); nb_[:, 2] = 0
    ln_ = np.linalg.norm(nb_, axis=1, keepdims=True)
    nb_ = np.where(ln_ > 0.2, nb_ / np.maximum(ln_, 1e-9), [0, 0, -1.0])
    pcd_p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([bp, foot_pts])))
    pcd_p.normals = o3d.utility.Vector3dVector(np.vstack([np.asarray(pcd.normals), np.tile(nb_, (len(levels), 1))]))

    lo, hi = bp.min(0) - 0.5 * U, bp.max(0) + 0.5 * U
    mesh_o = poisson_mesh(pcd_p, POISSON_DEPTH, 0.02, np.array([lo[0], lo[1], -SINK]), hi, keep_largest=True)
    if SMOOTH_ITERS > 0:
        mesh_o = mesh_o.filter_smooth_taubin(number_of_iterations=SMOOTH_ITERS)
    log("Poisson 完成，開始補洞")
    mf = pymeshfix.MeshFix(np.asarray(mesh_o.vertices), np.asarray(mesh_o.triangles))
    mf.repair(joincomp=True, remove_smallest_components=True)
    Vb, Fb = mf.points, mf.faces
    log("補洞完成")

    bpx = bp[:, :2]
    cxy = (bpx.min(0) + bpx.max(0)) / 2
    R_base = np.linalg.norm(bpx - cxy, axis=1).max() + max(1.0, 0.15 * np.ptp(bpx, axis=0).max())
    tm = (np.abs(Pc_g[:, 2]) < 0.3 * U) & (np.linalg.norm(Pc_g[:, :2] - cxy, axis=1) < R_base + 2 * U)
    to_m = lambda V, F: m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(V, np.float32), tri_verts=np.asarray(F, np.uint32)))
    obj_m = to_m(Vb, Fb)
    use_base = has_table and tm.sum() > 200
    if use_base:
        base = make_base(cxy, R_base, "circle", T_BASE, Pc_g[tm], COL_g[tm])
        base_m = to_m(base.vertices, base.triangles)
    if obj_m.status() == m3d.Error.NoError and (not use_base or base_m.status() == m3d.Error.NoError):
        solid = obj_m + base_m if use_base else obj_m.trim_by_plane((0, 0, 1), 0.0)
        out = solid.to_mesh()
        V3 = np.array(out.vert_properties)[:, :3].astype(np.float64)
        F3 = np.array(out.tri_verts).astype(np.int32)
    else:
        WARN.append("布林運算失敗，模型由兩個各自封閉的部分組成")
        V3, F3 = np.asarray(Vb, float), np.asarray(Fb, np.int32)
        if use_base:
            V3 = np.vstack([V3, np.asarray(base.vertices)]); F3 = np.vstack([F3, np.asarray(base.triangles) + len(Vb)])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V3), o3d.utility.Vector3iVector(F3))
    mesh.compute_vertex_normals()

    cols = np.zeros((len(V3), 3))
    is_obj = V3[:, 2] > 0.01 * U
    if use_base:
        top = ~is_obj & (np.abs(V3[:, 2]) < 0.01 * U)
        _, ti = cKDTree(Pc_g[tm][:, :2]).query(V3[top][:, :2], k=8)
        cols[top] = COL_g[tm][ti].mean(1)
        cols[~is_obj & ~top] = np.median(COL_g[tm], 0) * 0.85
    _, fb = cKDTree(bp).query(V3[is_obj], k=4)
    cols[is_obj] = np.asarray(pcd.colors)[fb].mean(1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))

    import trimesh
    wt = trimesh.Trimesh(V3, F3, process=False)
    info = dict(watertight=bool(wt.is_watertight), volume_cm3=round(float(abs(wt.volume)), 1) if wt.is_watertight else None,
                triangles=int(len(F3)), has_base=bool(use_base))
    return mesh, bp, info


def export_all(mesh):
    import trimesh
    V = np.asarray(mesh.vertices); F_ = np.asarray(mesh.triangles)
    VC = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    trimesh.Trimesh(V * 10, F_, vertex_colors=VC, process=False).export(f"{WORK}/model_mm.ply")
    trimesh.Trimesh(V * 10, F_, process=False).export(f"{WORK}/model_mm.stl")
    Vg = V * 0.01
    trimesh.Trimesh(np.stack([Vg[:, 0], Vg[:, 2], -Vg[:, 1]], 1), F_, vertex_colors=VC, process=False).export(f"{WORK}/model.glb")
    # App 預覽用的精簡網格（公尺、y 軸朝上）
    prev = mesh.simplify_quadric_decimation(150_000) if len(F_) > 150_000 else mesh
    prev.compute_vertex_normals()
    write_preview(prev, f"{WORK}/preview.bin")
    log("匯出完成：", sorted(os.listdir(WORK)))


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
