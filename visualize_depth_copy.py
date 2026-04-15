import numpy as np
import cv2
import open3d as o3d
import argparse
from pathlib import Path


def load_poses(traj_path):
    poses = []
    with open(traj_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            pose = np.array(vals).reshape(4, 4)
            poses.append(pose)
    return poses


def backproject(depth, K):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)


def depth_to_world_pcd(depth_path, pose, K, rgb_path, tint, tint_strength=0.3, stride=4):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth /= 1000.0  # mm -> m

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pts_cam = backproject(depth, K)
    # subsample for performance
    pts_cam = pts_cam[::stride, ::stride]
    rgb_sub = rgb[::stride, ::stride]

    pts_flat = pts_cam.reshape(-1, 3)
    colors_flat = rgb_sub.reshape(-1, 3)
    valid = pts_flat[:, 2] > 0
    pts_flat = pts_flat[valid]
    colors_flat = colors_flat[valid]

    # blend with tint: mix original RGB with tint color
    tint_arr = np.array(tint, dtype=np.float32)
    colors_flat = colors_flat * (1.0 - tint_strength) + tint_arr * tint_strength
    colors_flat = np.clip(colors_flat, 0.0, 1.0)

    # transform to world
    ones = np.ones((pts_flat.shape[0], 1))
    pts_h = np.concatenate([pts_flat, ones], axis=1)
    pts_world = (pose @ pts_h.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(colors_flat)
    return pcd


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


def subsample_pcd(pcd, voxel_size, max_points):
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    n = len(pcd.points)
    if max_points > 0 and n > max_points:
        ratio = float(max_points) / float(n)
        pcd = pcd.random_down_sample(ratio)
    return pcd


def merge_pcd_with_cap(merged_pcd, pcd_to_add, max_total_points):
    if len(pcd_to_add.points) == 0:
        return merged_pcd
    merged_pcd += pcd_to_add
    if max_total_points > 0:
        n = len(merged_pcd.points)
        if n > max_total_points:
            ratio = float(max_total_points) / float(n)
            merged_pcd = merged_pcd.random_down_sample(ratio)
    return merged_pcd


def umeyama_similarity(src_points, dst_points):
    """
    Estimate similarity transform that maps src -> dst:
        dst ~= scale * R * src + t
    """
    if src_points.shape != dst_points.shape:
        raise ValueError(f"Shape mismatch: {src_points.shape} vs {dst_points.shape}")
    if src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    n = src_points.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 correspondences, got {n}")

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    cov = (dst_centered.T @ src_centered) / n
    U, singular_vals, Vt = np.linalg.svd(cov)

    D = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0

    rotation = U @ D @ Vt
    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    if src_var < 1e-12:
        raise ValueError("Source point variance is too small for scale estimation.")

    scale = np.trace(np.diag(singular_vals) @ D) / src_var
    translation = dst_mean - scale * (rotation @ src_mean)
    return float(scale), rotation, translation


def collect_corresponding_world_points(gt_depth_files, pi3_depth_files, gt_poses, pi3_poses, K, frames, stride, max_points):
    rng = np.random.default_rng(42)
    num_frames = max(1, len(frames))
    per_frame_cap = max(1, max_points // num_frames)

    gt_points_all = []
    pi3_points_all = []

    for i in frames:
        gt_depth = cv2.imread(str(gt_depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        pi3_depth = cv2.imread(str(pi3_depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        gt_cam = backproject(gt_depth, K)[::stride, ::stride].reshape(-1, 3)
        pi3_cam = backproject(pi3_depth, K)[::stride, ::stride].reshape(-1, 3)

        valid = (gt_cam[:, 2] > 0) & (pi3_cam[:, 2] > 0)
        if not np.any(valid):
            continue

        gt_world = transform_points_rigid(gt_cam[valid], gt_poses[i])
        pi3_world = transform_points_rigid(pi3_cam[valid], pi3_poses[i])

        if gt_world.shape[0] > per_frame_cap:
            idx = rng.choice(gt_world.shape[0], size=per_frame_cap, replace=False)
            gt_world = gt_world[idx]
            pi3_world = pi3_world[idx]

        gt_points_all.append(gt_world)
        pi3_points_all.append(pi3_world)

    if not gt_points_all:
        raise RuntimeError("No valid GT/Pi3 depth correspondences found for Umeyama alignment.")

    gt_points = np.concatenate(gt_points_all, axis=0)
    pi3_points = np.concatenate(pi3_points_all, axis=0)
    return gt_points, pi3_points


def create_camera_frustum(pose, K, color, scale=0.15):
    """Create a small wireframe camera frustum at the given pose."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w, h = int(cx * 2), int(cy * 2)

    # corners of the image plane at depth=scale
    pts_cam = np.array([
        [0, 0, 0],  # camera center
        [(0 - cx) / fx * scale, (0 - cy) / fy * scale, scale],     # top-left
        [(w - cx) / fx * scale, (0 - cy) / fy * scale, scale],     # top-right
        [(w - cx) / fx * scale, (h - cy) / fy * scale, scale],     # bot-right
        [(0 - cx) / fx * scale, (h - cy) / fy * scale, scale],     # bot-left
    ])

    ones = np.ones((pts_cam.shape[0], 1))
    pts_h = np.concatenate([pts_cam, ones], axis=1)
    pts_world = (pose @ pts_h.T).T[:, :3]

    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    return ls


def main():
    parser = argparse.ArgumentParser(description="Visualize GT and Pi3 depth reprojection in one window")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene folder")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize (0-based)")
    parser.add_argument("--all", action="store_true", help="Visualize all frames (slower)")
    parser.add_argument(
        "--consecutive",
        type=int,
        default=1,
        help="Number of consecutive frames to overlay in one window starting from --frame",
    )
    parser.add_argument("--stride", type=int, default=4, help="Pixel stride for subsampling")
    parser.add_argument("--every", type=int, default=10, help="When --all, take every N-th frame")
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)
    parser.add_argument(
        "--pcd_voxel_size",
        type=float,
        default=0.03,
        help="Voxel size (meters) for downsampling each frame's point cloud",
    )
    parser.add_argument(
        "--max_points_per_frame",
        type=int,
        default=25000,
        help="Maximum points kept per frame after downsampling (<=0 disables)",
    )
    parser.add_argument(
        "--max_points_total",
        type=int,
        default=400000,
        help="Maximum points kept in the merged GT cloud and merged Pi3 cloud each (<=0 disables)",
    )
    parser.add_argument(
        "--camera_every",
        type=int,
        default=5,
        help="Draw every N-th camera frustum to reduce geometry count",
    )
    parser.add_argument(
        "--umeyama_max_points",
        type=int,
        default=200000,
        help="Maximum number of 3D correspondences used to estimate Umeyama alignment",
    )
    parser.add_argument(
        "--no_umeyama",
        action="store_true",
        help="Disable Umeyama alignment (show raw Pi3 frame)",
    )
    args = parser.parse_args()

    scene = Path(args.scene)
    K = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ])

    gt_poses = load_poses(scene / "traj.txt")
    pi3_poses = load_poses(scene / "pi3_traj.txt")

    gt_depth_dir = scene / "gt_depth"
    pi3_depth_dir = scene / "pi3_depth"
    image_dir = scene / "images"

    gt_depth_files = sorted(gt_depth_dir.glob("*.png"))
    pi3_depth_files = sorted(pi3_depth_dir.glob("*.png"))
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(image_dir.glob("*.png"))

    n = min(len(gt_depth_files), len(pi3_depth_files), len(image_files), len(gt_poses), len(pi3_poses))
    print(f"Total frames: {n}")
    if n == 0:
        raise RuntimeError("No valid frames found in scene data.")
    if args.frame < 0 or args.frame >= n:
        raise ValueError(f"--frame must be in [0, {n - 1}], got {args.frame}")
    if args.consecutive < 1:
        raise ValueError(f"--consecutive must be >= 1, got {args.consecutive}")
    if args.pcd_voxel_size < 0:
        raise ValueError(f"--pcd_voxel_size must be >= 0, got {args.pcd_voxel_size}")
    if args.camera_every < 1:
        raise ValueError(f"--camera_every must be >= 1, got {args.camera_every}")
    if args.umeyama_max_points < 3:
        raise ValueError(f"--umeyama_max_points must be >= 3, got {args.umeyama_max_points}")

    RED = [1.0, 0.0, 0.0]
    GREEN = [0.0, 1.0, 0.0]
    geometries = []

    # Both world-frame axes are added (each world origin is [0, 0, 0] in its own coordinates).
    gt_world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    pi3_world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35, origin=[0, 0, 0])
    geometries.append(gt_world_frame)
    geometries.append(pi3_world_frame)

    if args.all:
        frames = list(range(0, n, args.every))
        print(f"Visualizing {len(frames)} frames (every {args.every}-th)")
    else:
        end_frame = min(args.frame + args.consecutive, n)
        frames = list(range(args.frame, end_frame))
        if len(frames) == 1:
            print(f"Visualizing frame {args.frame}")
        else:
            print(
                f"Visualizing {len(frames)} consecutive frames "
                f"from {frames[0]} to {frames[-1]} (overlaid in one window)"
            )

    np.set_printoptions(precision=6, suppress=True)
    if args.no_umeyama:
        umeyama_scale = 1.0
        umeyama_rotation = np.eye(3, dtype=np.float64)
        umeyama_translation = np.zeros(3, dtype=np.float64)
        print("Umeyama alignment disabled (--no_umeyama).")
    else:
        gt_corr, pi3_corr = collect_corresponding_world_points(
            gt_depth_files,
            pi3_depth_files,
            gt_poses,
            pi3_poses,
            K,
            frames,
            args.stride,
            args.umeyama_max_points,
        )
        print(f"Estimating Umeyama using {gt_corr.shape[0]} matched 3D point pairs...")
        umeyama_scale, umeyama_rotation, umeyama_translation = umeyama_similarity(pi3_corr, gt_corr)

        before = np.mean(np.linalg.norm(gt_corr - pi3_corr, axis=1))
        pi3_corr_aligned = transform_points_similarity(
            pi3_corr, umeyama_scale, umeyama_rotation, umeyama_translation
        )
        after = np.mean(np.linalg.norm(gt_corr - pi3_corr_aligned, axis=1))

        print("\nUmeyama alignment (Pi3 -> GT):")
        print(f"scale = {umeyama_scale:.9f}")
        print("rotation =")
        print(umeyama_rotation)
        print("translation =")
        print(umeyama_translation)
        print(f"mean point distance before alignment: {before:.6f} m")
        print(f"mean point distance after alignment:  {after:.6f} m\n")

    if not args.no_umeyama:
        apply_similarity_to_geometry(
            pi3_world_frame, umeyama_scale, umeyama_rotation, umeyama_translation
        )

    gt_merged_pcd = o3d.geometry.PointCloud()
    pi3_merged_pcd = o3d.geometry.PointCloud()

    for frame_idx, i in enumerate(frames):
        print(f"  Processing frame {i}...")

        # GT depth + GT pose -> GT world frame
        gt_pcd = depth_to_world_pcd(gt_depth_files[i], gt_poses[i], K, image_files[i],
                                     tint=RED, tint_strength=0.3, stride=args.stride)
        gt_pcd = subsample_pcd(gt_pcd, args.pcd_voxel_size, args.max_points_per_frame)
        gt_merged_pcd = merge_pcd_with_cap(gt_merged_pcd, gt_pcd, args.max_points_total)

        # Pi3 depth + Pi3 pose -> Pi3 world frame
        pi3_pcd = depth_to_world_pcd(pi3_depth_files[i], pi3_poses[i], K, image_files[i],
                                      tint=GREEN, tint_strength=0.3, stride=args.stride)
        if not args.no_umeyama:
            pi3_pcd = apply_similarity_to_geometry(
                pi3_pcd, umeyama_scale, umeyama_rotation, umeyama_translation
            )
        pi3_pcd = subsample_pcd(pi3_pcd, args.pcd_voxel_size, args.max_points_per_frame)
        pi3_merged_pcd = merge_pcd_with_cap(pi3_merged_pcd, pi3_pcd, args.max_points_total)

        # camera frustums in their own world frames
        if frame_idx % args.camera_every == 0:
            gt_cam = create_camera_frustum(gt_poses[i], K, RED, scale=0.1)
            pi3_cam = create_camera_frustum(pi3_poses[i], K, GREEN, scale=0.1)
            if not args.no_umeyama:
                pi3_cam = apply_similarity_to_geometry(
                    pi3_cam, umeyama_scale, umeyama_rotation, umeyama_translation
                )
            geometries.append(gt_cam)
            geometries.append(pi3_cam)

    if len(gt_merged_pcd.points) > 0:
        geometries.append(gt_merged_pcd)
    if len(pi3_merged_pcd.points) > 0:
        geometries.append(pi3_merged_pcd)

    print(f"Launching single viewer with {len(geometries)} geometries...")
    print("  Red   = GT depth + GT camera poses")
    if args.no_umeyama:
        print("  Green = Pi3 depth + Pi3 camera poses (raw Pi3 frame)")
    else:
        print("  Green = Pi3 depth + Pi3 camera poses (Umeyama-aligned to GT)")
    print("  Two coordinate frames are included (GT and Pi3 world axes at their own origins).")
    print(f"  Merged GT points:  {len(gt_merged_pcd.points)}")
    print(f"  Merged Pi3 points: {len(pi3_merged_pcd.points)}")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="GT + Pi3 Depth Reprojection (Single Window)",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
