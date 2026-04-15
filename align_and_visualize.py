"""
Align Pi3 depth to GT depth using a single reference frame,
then visualize both point clouds + GT 3D bboxes in the GT world frame.

Alignment modes:
  - Rigid  (default): estimate R, t  (no scale)
  - Similarity (--scale): estimate s, R, t  (Umeyama)
"""

import numpy as np
import cv2
import json
import argparse
import colorsys
import open3d as o3d
from pathlib import Path


# ── loading helpers ──────────────────────────────────────────────────────────

def load_poses(traj_path):
    poses = []
    with open(traj_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(4, 4))
    return poses


def backproject(depth, K):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)


# ── colour helpers ───────────────────────────────────────────────────────────

def id_to_color(track_id, as_float=True):
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    if as_float:
        return (r, g, b)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── 3-D bbox helpers ────────────────────────────────────────────────────────

def aabb_corners(aabb):
    xn, yn, zn, xx, yx, zx = aabb
    return np.array([
        [xn, yn, zn], [xn, yn, zx], [xn, yx, zn], [xn, yx, zx],
        [xx, yn, zn], [xx, yn, zx], [xx, yx, zn], [xx, yx, zx],
    ])


BBOX_EDGES = [
    [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
    [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
]


# ── rigid / similarity transforms ───────────────────────────────────────────

def transform_points_rigid(points, pose):
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    pts_h = np.concatenate([points, ones], axis=1)
    return (pose @ pts_h.T).T[:, :3]


def transform_points_similarity(points, scale, rotation, translation):
    return (scale * (rotation @ points.T).T) + translation


def apply_similarity_to_geometry(geometry, scale, rotation, translation):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * rotation
    T[:3, 3] = translation
    geometry.transform(T)
    return geometry


# ── alignment estimators ────────────────────────────────────────────────────

def umeyama_similarity(src, dst):
    """Similarity transform  dst ≈ scale · R · src + t"""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    src_mu, dst_mu = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mu, dst - dst_mu

    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0

    R = U @ D @ Vt
    src_var = np.mean(np.sum(src_c ** 2, axis=1))
    scale = np.trace(np.diag(S) @ D) / src_var
    t = dst_mu - scale * (R @ src_mu)
    return float(scale), R, t


def rigid_alignment(src, dst):
    """Rigid transform  dst ≈ R · src + t  (scale = 1)"""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    src_mu, dst_mu = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mu, dst - dst_mu

    cov = (dst_c.T @ src_c) / n
    U, _S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0

    R = U @ D @ Vt
    t = dst_mu - R @ src_mu
    return 1.0, R, t


# ── correspondence collection from a single frame ───────────────────────────

def correspondences_from_frame(
    gt_depth_file, pi3_depth_file, gt_pose, pi3_pose, K, stride, max_points
):
    gt_depth = cv2.imread(str(gt_depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    pi3_depth = cv2.imread(str(pi3_depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    gt_cam = backproject(gt_depth, K)[::stride, ::stride].reshape(-1, 3)
    pi3_cam = backproject(pi3_depth, K)[::stride, ::stride].reshape(-1, 3)

    valid = (gt_cam[:, 2] > 0) & (pi3_cam[:, 2] > 0)
    if not np.any(valid):
        raise RuntimeError("No overlapping valid depth in the reference frame.")

    gt_world = transform_points_rigid(gt_cam[valid], gt_pose)
    pi3_world = transform_points_rigid(pi3_cam[valid], pi3_pose)

    if max_points > 0 and gt_world.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(gt_world.shape[0], size=max_points, replace=False)
        gt_world = gt_world[idx]
        pi3_world = pi3_world[idx]

    return gt_world, pi3_world


# ── point-cloud creation ────────────────────────────────────────────────────

def depth_to_world_pcd(depth_path, pose, K, rgb_path, tint, tint_strength=0.3, stride=4):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pts_cam = backproject(depth, K)[::stride, ::stride]
    rgb_sub = rgb[::stride, ::stride]

    pts_flat = pts_cam.reshape(-1, 3)
    cols_flat = rgb_sub.reshape(-1, 3)
    valid = pts_flat[:, 2] > 0
    pts_flat, cols_flat = pts_flat[valid], cols_flat[valid]

    tint_arr = np.array(tint, dtype=np.float32)
    cols_flat = cols_flat * (1.0 - tint_strength) + tint_arr * tint_strength
    cols_flat = np.clip(cols_flat, 0.0, 1.0)

    ones = np.ones((pts_flat.shape[0], 1))
    pts_world = (pose @ np.concatenate([pts_flat, ones], 1).T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(cols_flat)
    return pcd


def subsample_pcd(pcd, voxel_size, max_points):
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    n = len(pcd.points)
    if max_points > 0 and n > max_points:
        pcd = pcd.random_down_sample(float(max_points) / float(n))
    return pcd


def merge_pcd_with_cap(merged, new, max_total):
    if len(new.points) == 0:
        return merged
    merged += new
    n = len(merged.points)
    if max_total > 0 and n > max_total:
        merged = merged.random_down_sample(float(max_total) / float(n))
    return merged


# ── camera frustum ──────────────────────────────────────────────────────────

def create_camera_frustum(pose, K, color, scale=0.15):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = int(cx * 2), int(cy * 2)
    pts_cam = np.array([
        [0, 0, 0],
        [(0 - cx) / fx * scale, (0 - cy) / fy * scale, scale],
        [(w - cx) / fx * scale, (0 - cy) / fy * scale, scale],
        [(w - cx) / fx * scale, (h - cy) / fy * scale, scale],
        [(0 - cx) / fx * scale, (h - cy) / fy * scale, scale],
    ])
    ones = np.ones((pts_cam.shape[0], 1))
    pts_world = (pose @ np.concatenate([pts_cam, ones], 1).T).T[:, :3]

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    return ls


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Align Pi3 depth to GT depth (single-frame) and visualize PCDs + 3D bboxes"
    )
    parser.add_argument("--scene", type=str, required=True, help="Path to scene folder")
    parser.add_argument(
        "--align_frame", type=int, default=0,
        help="Frame index used for alignment (default: 0)"
    )
    parser.add_argument(
        "--scale", action="store_true",
        help="Estimate scale (Umeyama similarity). Without this flag, rigid (R,t) only."
    )
    parser.add_argument("--frame", type=int, default=0, help="First frame to visualize")
    parser.add_argument("--all", action="store_true", help="Visualize all frames")
    parser.add_argument(
        "--consecutive", type=int, default=1,
        help="Number of consecutive frames to overlay starting from --frame"
    )
    parser.add_argument("--stride", type=int, default=4, help="Pixel stride for subsampling")
    parser.add_argument("--every", type=int, default=10, help="When --all, take every N-th frame")
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)
    parser.add_argument("--pcd_voxel_size", type=float, default=0.03)
    parser.add_argument("--max_points_per_frame", type=int, default=25000)
    parser.add_argument("--max_points_total", type=int, default=400000)
    parser.add_argument("--camera_every", type=int, default=5)
    parser.add_argument(
        "--align_max_points", type=int, default=200000,
        help="Max correspondences for alignment estimation"
    )
    args = parser.parse_args()

    scene = Path(args.scene)
    K = np.array([[args.fx, 0, args.cx],
                   [0, args.fy, args.cy],
                   [0, 0, 1]])

    # ── load data paths ──
    gt_poses = load_poses(scene / "traj.txt")
    pi3_poses = load_poses(scene / "pi3_traj.txt")

    gt_depth_files = sorted((scene / "gt_depth").glob("*.png"))
    pi3_depth_files = sorted((scene / "pi3_depth").glob("*.png"))
    image_files = sorted((scene / "images").glob("*.jpg"))
    if not image_files:
        image_files = sorted((scene / "images").glob("*.png"))
    bbox_files = sorted((scene / "bbox").glob("bboxes*_info.json"))

    n = min(len(gt_depth_files), len(pi3_depth_files), len(image_files),
            len(gt_poses), len(pi3_poses))
    print(f"Total frames: {n}")
    assert n > 0, "No valid frames found."
    assert 0 <= args.align_frame < n, f"--align_frame {args.align_frame} out of range [0, {n})"
    assert 0 <= args.frame < n, f"--frame {args.frame} out of range [0, {n})"

    # ── estimate alignment from a single frame ──
    print(f"\nCollecting correspondences from frame {args.align_frame} …")
    gt_pts, pi3_pts = correspondences_from_frame(
        gt_depth_files[args.align_frame],
        pi3_depth_files[args.align_frame],
        gt_poses[args.align_frame],
        pi3_poses[args.align_frame],
        K, args.stride, args.align_max_points,
    )
    print(f"  {gt_pts.shape[0]} matched 3-D point pairs")

    if args.scale:
        print("Estimating similarity transform (s, R, t) …")
        s, R, t = umeyama_similarity(pi3_pts, gt_pts)
    else:
        print("Estimating rigid transform (R, t) …")
        s, R, t = rigid_alignment(pi3_pts, gt_pts)

    # report
    before = np.mean(np.linalg.norm(gt_pts - pi3_pts, axis=1))
    pi3_aligned = transform_points_similarity(pi3_pts, s, R, t)
    after = np.mean(np.linalg.norm(gt_pts - pi3_aligned, axis=1))

    np.set_printoptions(precision=6, suppress=True)
    print(f"\nAlignment (Pi3 → GT world):")
    print(f"  scale       = {s:.9f}")
    print(f"  rotation    =\n{R}")
    print(f"  translation = {t}")
    print(f"  mean dist before: {before:.6f} m")
    print(f"  mean dist after:  {after:.6f} m\n")

    # ── decide which frames to visualize ──
    if args.all:
        frames = list(range(0, n, args.every))
        print(f"Visualizing {len(frames)} frames (every {args.every}-th)")
    else:
        end = min(args.frame + args.consecutive, n)
        frames = list(range(args.frame, end))
        print(f"Visualizing frame(s) {frames[0]}–{frames[-1]}")

    RED = [1.0, 0.0, 0.0]
    GREEN = [0.0, 1.0, 0.0]
    geometries = []

    # world frame axes
    gt_world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(gt_world_frame)

    pi3_world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35, origin=[0, 0, 0])
    apply_similarity_to_geometry(pi3_world_frame, s, R, t)
    geometries.append(pi3_world_frame)

    # ── build point clouds ──
    gt_merged = o3d.geometry.PointCloud()
    pi3_merged = o3d.geometry.PointCloud()

    for idx, i in enumerate(frames):
        print(f"  Processing frame {i} …")

        # GT
        gt_pcd = depth_to_world_pcd(
            gt_depth_files[i], gt_poses[i], K, image_files[i],
            tint=RED, tint_strength=0.3, stride=args.stride,
        )
        gt_pcd = subsample_pcd(gt_pcd, args.pcd_voxel_size, args.max_points_per_frame)
        gt_merged = merge_pcd_with_cap(gt_merged, gt_pcd, args.max_points_total)

        # Pi3 → aligned to GT world
        pi3_pcd = depth_to_world_pcd(
            pi3_depth_files[i], pi3_poses[i], K, image_files[i],
            tint=GREEN, tint_strength=0.3, stride=args.stride,
        )
        apply_similarity_to_geometry(pi3_pcd, s, R, t)
        pi3_pcd = subsample_pcd(pi3_pcd, args.pcd_voxel_size, args.max_points_per_frame)
        pi3_merged = merge_pcd_with_cap(pi3_merged, pi3_pcd, args.max_points_total)

        # camera frustums
        if idx % args.camera_every == 0:
            gt_cam = create_camera_frustum(gt_poses[i], K, RED, scale=0.1)
            pi3_cam = create_camera_frustum(pi3_poses[i], K, GREEN, scale=0.1)
            apply_similarity_to_geometry(pi3_cam, s, R, t)
            geometries.append(gt_cam)
            geometries.append(pi3_cam)

    if len(gt_merged.points) > 0:
        geometries.append(gt_merged)
    if len(pi3_merged.points) > 0:
        geometries.append(pi3_merged)

    # ── 3-D bboxes (GT world frame) ─────────────────────────────────────────
    # Collect unique boxes across visualized frames
    seen_track_ids = set()
    all_boxes = []
    for i in frames:
        if i < len(bbox_files):
            with open(bbox_files[i]) as f:
                frame_boxes = json.load(f)["bboxes"]["bbox_3d"]["boxes"]
            for box in frame_boxes:
                tid = box["track_id"]
                if tid not in seen_track_ids:
                    seen_track_ids.add(tid)
                    all_boxes.append(box)

    print(f"  Drawing {len(all_boxes)} unique 3-D bboxes")
    for box in all_boxes:
        tid = box["track_id"]
        aabb = box["aabb_xyzmin_xyzmax"]
        c = id_to_color(tid)
        corners = aabb_corners(aabb)

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(BBOX_EDGES)
        ls.paint_uniform_color(list(c))
        geometries.append(ls)

    # ── launch viewer ────────────────────────────────────────────────────────
    mode = "similarity (s,R,t)" if args.scale else "rigid (R,t)"
    print(f"\nLaunching viewer with {len(geometries)} geometries …")
    print(f"  Red   = GT depth + GT poses")
    print(f"  Green = Pi3 depth + Pi3 poses  (aligned via {mode})")
    print(f"  Wireframes = GT 3-D bboxes (colored by track id)")
    print(f"  Merged GT points:  {len(gt_merged.points)}")
    print(f"  Merged Pi3 points: {len(pi3_merged.points)}")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"GT + Pi3 (aligned {mode}) + 3D BBoxes",
        width=1280, height=720,
    )


if __name__ == "__main__":
    main()
