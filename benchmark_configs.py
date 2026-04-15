"""
Benchmark different Pi3 depth configs (chunk_size / overlap) against GT depth.

For each scene, for each config (e.g. pi3_depth_12_6, pi3_traj_12_6.txt):
  1. Align Pi3 to GT world frame using frame 0 (rigid R,t and similarity s,R,t)
  2. Reconstruct full scene from ALL frames for both GT and aligned Pi3
  3. Compare the two full reconstructions with 3D metrics

Averages metrics over all scenes and produces two plots:
  1) Rigid (R,t) alignment — metrics vs config
  2) Similarity (s,R,t) alignment — metrics vs config

Usage:
  python benchmark_configs.py --dataset /path/to/scannetpp_scenes
  python benchmark_configs.py --dataset /path/to/scannetpp_scenes --configs 12_6 8_4 4_2
  python benchmark_configs.py --dataset /path/to/scannetpp_scenes --stride 4 --voxel_size 0.02
"""

import numpy as np
import cv2
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# ── helpers (same as align_and_visualize.py) ─────────────────────────────────

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


def transform_points_rigid(points, pose):
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return (pose @ np.concatenate([points, ones], 1).T).T[:, :3]


def transform_points_similarity(points, scale, rotation, translation):
    return (scale * (rotation @ points.T).T) + translation


def umeyama_similarity(src, dst):
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


# ── full-scene reconstruction + metrics ──────────────────────────────────────

def backproject_frame_to_world(depth_file, pose, K, stride):
    """Backproject a single depth frame to world-frame 3D points."""
    depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    pts_cam = backproject(depth, K)[::stride, ::stride].reshape(-1, 3)
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    if pts_cam.shape[0] == 0:
        return np.empty((0, 3))
    return transform_points_rigid(pts_cam, pose)


def reconstruct_scene(depth_files, poses, K, stride, n_frames, voxel_size):
    """
    Reconstruct full scene from all frames.
    Returns a voxel-downsampled numpy array of 3D points (N, 3).
    """
    import open3d as o3d

    merged = o3d.geometry.PointCloud()
    for i in range(n_frames):
        pts = backproject_frame_to_world(depth_files[i], poses[i], K, stride)
        if pts.shape[0] == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        merged += pcd

    # voxel downsample the merged cloud to keep memory in check
    if voxel_size > 0 and len(merged.points) > 0:
        merged = merged.voxel_down_sample(voxel_size)

    return np.asarray(merged.points)


def compute_reconstruction_metrics(gt_pts, pred_pts, threshold=0.05):
    """
    Compare two full-scene point clouds.
    Returns dict with accuracy, completion, chamfer, precision, recall, f1.
    """
    if gt_pts.shape[0] == 0 or pred_pts.shape[0] == 0:
        return None

    from scipy.spatial import cKDTree

    gt_tree = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)

    # accuracy: mean dist from each predicted point to closest GT
    dist_pred_to_gt, _ = gt_tree.query(pred_pts)
    # completion: mean dist from each GT point to closest predicted
    dist_gt_to_pred, _ = pred_tree.query(gt_pts)

    accuracy = float(np.mean(dist_pred_to_gt))
    completion = float(np.mean(dist_gt_to_pred))
    chamfer = (accuracy + completion) / 2.0
    rmse = float(np.sqrt(np.mean(dist_pred_to_gt ** 2)))

    precision = float(np.mean(dist_pred_to_gt < threshold))
    recall = float(np.mean(dist_gt_to_pred < threshold))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "completion": completion,
        "chamfer": chamfer,
        "rmse": rmse,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ── discover configs in a scene folder ───────────────────────────────────────

def discover_configs(scene_dir, explicit_configs=None):
    """
    Find Pi3 configs available in a scene directory.
    Looks for folders named pi3_depth_<X>_<Y> and matching pi3_traj_<X>_<Y>.txt.
    Returns list of config label strings like '12_6', '8_4', etc.
    """
    scene_dir = Path(scene_dir)
    if explicit_configs:
        # verify they exist
        valid = []
        for cfg in explicit_configs:
            depth_dir = scene_dir / f"pi3_depth_{cfg}"
            traj_file = scene_dir / f"pi3_traj_{cfg}.txt"
            if depth_dir.is_dir() and traj_file.is_file():
                valid.append(cfg)
        return sorted(valid)

    configs = []
    for d in sorted(scene_dir.iterdir()):
        if d.is_dir() and d.name.startswith("pi3_depth_"):
            cfg = d.name[len("pi3_depth_"):]
            traj = scene_dir / f"pi3_traj_{cfg}.txt"
            if traj.is_file():
                configs.append(cfg)
    return sorted(configs)


# ── evaluate one scene, one config ───────────────────────────────────────────

def evaluate_config(scene_dir, config, K, align_frame, stride, voxel_size, max_eval_points):
    """
    Build full GT and Pi3 reconstructions, align Pi3 → GT, compare.
    Returns dict:  { 'rigid': {metric: value}, 'similarity': {metric: value} }
    or None if data is insufficient.
    """
    scene_dir = Path(scene_dir)
    gt_depth_dir = scene_dir / "gt_depth"
    pi3_depth_dir = scene_dir / f"pi3_depth_{config}"
    gt_traj_file = scene_dir / "traj.txt"
    pi3_traj_file = scene_dir / f"pi3_traj_{config}.txt"

    if not all(p.exists() for p in [gt_depth_dir, pi3_depth_dir, gt_traj_file, pi3_traj_file]):
        return None

    gt_poses = load_poses(gt_traj_file)
    pi3_poses = load_poses(pi3_traj_file)

    gt_depth_files = sorted(gt_depth_dir.glob("*.png"))
    pi3_depth_files = sorted(pi3_depth_dir.glob("*.png"))

    n = min(len(gt_depth_files), len(pi3_depth_files), len(gt_poses), len(pi3_poses))
    if n == 0 or align_frame >= n:
        return None

    # ── 1. Estimate alignment from a single frame ──
    gt_depth = cv2.imread(str(gt_depth_files[align_frame]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    pi3_depth = cv2.imread(str(pi3_depth_files[align_frame]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    gt_cam = backproject(gt_depth, K)[::stride, ::stride].reshape(-1, 3)
    pi3_cam = backproject(pi3_depth, K)[::stride, ::stride].reshape(-1, 3)

    valid = (gt_cam[:, 2] > 0) & (pi3_cam[:, 2] > 0)
    if not np.any(valid):
        return None

    gt_corr = transform_points_rigid(gt_cam[valid], gt_poses[align_frame])
    pi3_corr = transform_points_rigid(pi3_cam[valid], pi3_poses[align_frame])

    align_cap = 200000
    if gt_corr.shape[0] > align_cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(gt_corr.shape[0], size=align_cap, replace=False)
        gt_corr, pi3_corr = gt_corr[idx], pi3_corr[idx]

    s_rigid, R_rigid, t_rigid = rigid_alignment(pi3_corr, gt_corr)
    s_sim, R_sim, t_sim = umeyama_similarity(pi3_corr, gt_corr)

    # ── 2. Reconstruct full GT scene ──
    print(f"reconstructing GT ({n} frames)…", end=" ", flush=True)
    gt_pts = reconstruct_scene(gt_depth_files, gt_poses, K, stride, n, voxel_size)
    print(f"{gt_pts.shape[0]} pts.", end=" ", flush=True)

    # ── 3. Reconstruct full Pi3 scene ──
    print(f"Pi3…", end=" ", flush=True)
    pi3_pts = reconstruct_scene(pi3_depth_files, pi3_poses, K, stride, n, voxel_size)
    print(f"{pi3_pts.shape[0]} pts.", end=" ", flush=True)

    if gt_pts.shape[0] == 0 or pi3_pts.shape[0] == 0:
        return None

    # ── 4. Subsample for KD-tree efficiency if needed ──
    rng = np.random.default_rng(0)
    gt_eval = gt_pts
    if max_eval_points > 0 and gt_eval.shape[0] > max_eval_points:
        idx = rng.choice(gt_eval.shape[0], size=max_eval_points, replace=False)
        gt_eval = gt_eval[idx]

    out = {}
    for mode, (s, R, t) in [("rigid", (s_rigid, R_rigid, t_rigid)),
                              ("similarity", (s_sim, R_sim, t_sim))]:
        # apply alignment to Pi3 cloud
        pi3_aligned = transform_points_similarity(pi3_pts, s, R, t)

        pi3_eval = pi3_aligned
        if max_eval_points > 0 and pi3_eval.shape[0] > max_eval_points:
            idx = rng.choice(pi3_eval.shape[0], size=max_eval_points, replace=False)
            pi3_eval = pi3_eval[idx]

        m = compute_reconstruction_metrics(gt_eval, pi3_eval)
        if m is None:
            return None
        out[mode] = m

    # ── 5. Save transformation JSON ──
    transform_data = {
        "scale": s_sim,
        "rotation": R_sim.tolist(),
        "translation": t_sim.tolist(),
    }
    json_path = scene_dir / f"pi3_{config}_to_world.json"
    with open(json_path, "w") as f:
        json.dump(transform_data, f, indent=2)
    print(f"saved {json_path.name}", end=" ", flush=True)

    return out


# ── plotting ─────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "accuracy": "Accuracy (m) ↓",
    "completion": "Completion (m) ↓",
    "chamfer": "Chamfer (m) ↓",
    "rmse": "RMSE (m) ↓",
    "precision": "Precision @5cm ↑",
    "recall": "Recall @5cm ↑",
    "f1": "F1 @5cm ↑",
}


def make_config_label(cfg_str):
    """Turn '12_6' into 'chunk=12 / overlap=6'."""
    parts = cfg_str.split("_")
    if len(parts) == 2:
        return f"chunk={parts[0]} / ovlp={parts[1]}"
    return cfg_str


def plot_results(all_metrics, mode_label, output_path=None):
    """
    all_metrics: dict  config_str -> {metric_name: mean_value}
    Generates a grouped bar chart with one group per metric, one bar per config.
    """
    configs = list(all_metrics.keys())
    metrics = list(METRIC_LABELS.keys())
    n_configs = len(configs)
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_configs, 1)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = [all_metrics[c].get(metric, 0) for c in configs]
        labels = [make_config_label(c) for c in configs]
        bars = ax.bar(range(n_configs), vals, color=colors[:n_configs])
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())

        # annotate bar values
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=7,
            )

    # hide unused subplot axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"3D Reconstruction Metrics — {mode_label} alignment\n(averaged over scenes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot: {output_path}")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Pi3 depth configs vs GT — 3D reconstruction metrics"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root containing scene folders")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Explicit config suffixes, e.g. 12_6 8_4 4_2. "
                             "If omitted, auto-discovers from first scene.")
    parser.add_argument("--align_frame", type=int, default=0,
                        help="Frame index used for alignment (default: 0)")
    parser.add_argument("--stride", type=int, default=4,
                        help="Pixel stride for depth backprojection")
    parser.add_argument("--voxel_size", type=float, default=0.02,
                        help="Voxel size (m) for downsampling the merged reconstructions")
    parser.add_argument("--max_eval_points", type=int, default=500000,
                        help="Max points per cloud for metric computation (<=0 disables)")
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save plot PNGs (optional)")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    K = np.array([[args.fx, 0, args.cx],
                   [0, args.fy, args.cy],
                   [0, 0, 1]])

    # ── discover scenes (folders that contain traj.txt and gt_depth/) ──
    scenes = sorted([
        d for d in dataset.iterdir()
        if d.is_dir() and (d / "traj.txt").is_file() and (d / "gt_depth").is_dir()
    ])
    if not scenes:
        raise RuntimeError(f"No valid scenes found under {dataset}")
    print(f"Found {len(scenes)} scene(s)")

    # ── discover configs ──
    configs = args.configs
    if configs is None:
        # auto-discover from the first scene
        configs = discover_configs(scenes[0])
        if not configs:
            raise RuntimeError(
                f"No pi3_depth_*/pi3_traj_*.txt configs found in {scenes[0]}. "
                "Specify --configs explicitly."
            )
    print(f"Configs to benchmark: {configs}")

    # ── evaluate ──
    # accumulate per-config, per-mode metrics across scenes
    rigid_accum = {c: defaultdict(list) for c in configs}
    sim_accum = {c: defaultdict(list) for c in configs}
    scene_counts = {c: 0 for c in configs}

    for scene in scenes:
        scene_configs = discover_configs(scene, explicit_configs=configs)
        if not scene_configs:
            print(f"  [{scene.name}] no matching configs, skipping")
            continue

        for cfg in scene_configs:
            print(f"  [{scene.name}] config {cfg} … ", end="", flush=True)
            result = evaluate_config(
                scene, cfg, K, args.align_frame,
                args.stride, args.voxel_size, args.max_eval_points,
            )
            if result is None:
                print("SKIP (insufficient data)")
                continue

            scene_counts[cfg] += 1
            for k, v in result["rigid"].items():
                rigid_accum[cfg][k].append(v)
            for k, v in result["similarity"].items():
                sim_accum[cfg][k].append(v)
            print("OK")

    # ── average over scenes ──
    rigid_avg = {}
    sim_avg = {}
    for cfg in configs:
        if scene_counts[cfg] == 0:
            print(f"  WARNING: config {cfg} had no valid scenes, skipping from plots")
            continue
        rigid_avg[cfg] = {k: float(np.mean(v)) for k, v in rigid_accum[cfg].items()}
        sim_avg[cfg] = {k: float(np.mean(v)) for k, v in sim_accum[cfg].items()}

    if not rigid_avg:
        raise RuntimeError("No valid results for any config. Check your data.")

    # ── print summary table ──
    print("\n" + "=" * 80)
    print("RESULTS (averaged over scenes)")
    print("=" * 80)
    for mode, avg in [("Rigid (R,t)", rigid_avg), ("Similarity (s,R,t)", sim_avg)]:
        print(f"\n  {mode}:")
        header = f"  {'Config':<16}"
        for m in METRIC_LABELS:
            header += f"  {m:>12}"
        print(header)
        print("  " + "-" * (16 + 14 * len(METRIC_LABELS)))
        for cfg in avg:
            row = f"  {make_config_label(cfg):<16}"
            for m in METRIC_LABELS:
                row += f"  {avg[cfg].get(m, 0):>12.5f}"
            print(row)

    # ── plot ──
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    plot_results(
        rigid_avg, "Rigid (R, t)",
        output_path=save_dir / "benchmark_rigid.png" if save_dir else None,
    )
    plot_results(
        sim_avg, "Similarity (s, R, t)",
        output_path=save_dir / "benchmark_similarity.png" if save_dir else None,
    )


if __name__ == "__main__":
    main()
