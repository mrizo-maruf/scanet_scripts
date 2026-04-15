"""
Align GT world frame to Pi3 world frame using Sim(3) Umeyama alignment.

Given:
  - GT trajectory (poses in GT world frame)
  - Pi3 trajectory (poses in Pi3 learned world frame)

Computes Sim(3) transform S = (s, R, t) such that:
  p_pi3 = s * R @ p_gt + t

Then transforms GT 3D bboxes into Pi3 world frame.
"""

import numpy as np
import json
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


def umeyama_alignment(x, y, with_scale=True):
    """
    Umeyama alignment: find s, R, t such that y ≈ s*R@x + t

    x: (3, N) - source points (GT positions)
    y: (3, N) - target points (Pi3 positions)

    Returns: s (scale), R (3x3 rotation), t (3x1 translation)
    """
    assert x.shape == y.shape
    n = x.shape[1]

    mu_x = x.mean(axis=1, keepdims=True)
    mu_y = y.mean(axis=1, keepdims=True)

    x_c = x - mu_x
    y_c = y - mu_y

    sigma_x_sq = np.sum(x_c ** 2) / n
    cov = (y_c @ x_c.T) / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        s = np.trace(np.diag(D) @ S) / sigma_x_sq
    else:
        s = 1.0

    t = mu_y - s * R @ mu_x

    return s, R, t.flatten()


def compute_alignment(gt_poses, pi3_poses):
    """Compute Sim(3) alignment from GT world to Pi3 world."""
    gt_positions = np.array([p[:3, 3] for p in gt_poses])   # (N, 3)
    pi3_positions = np.array([p[:3, 3] for p in pi3_poses]) # (N, 3)

    # umeyama expects (3, N)
    s, R, t = umeyama_alignment(gt_positions.T, pi3_positions.T, with_scale=True)
    return s, R, t


def transform_bbox_gt_to_pi3(bbox_min_max, s, R, t):
    """
    Transform a GT AABB [xmin, ymin, zmin, xmax, ymax, zmax]
    into Pi3 world frame using Sim(3).

    Since Sim(3) can rotate the box, we transform all 8 corners
    and recompute the AABB in the new frame.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_min_max

    # 8 corners of the AABB
    corners = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ])

    # apply Sim(3): p_pi3 = s * R @ p_gt + t
    corners_pi3 = (s * (R @ corners.T)).T + t

    new_min = corners_pi3.min(axis=0)
    new_max = corners_pi3.max(axis=0)

    return [float(new_min[0]), float(new_min[1]), float(new_min[2]),
            float(new_max[0]), float(new_max[1]), float(new_max[2])]


def transform_pose_gt_to_pi3(gt_pose, s, R, t):
    """Transform a GT pose into Pi3 world frame."""
    # S * T_gt where S is Sim(3)
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t
    return S @ gt_pose


def main():
    parser = argparse.ArgumentParser(description="Align GT and Pi3 world frames")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--save", action="store_true", help="Save transformed bboxes")
    parser.add_argument("--visualize", action="store_true", help="Visualize alignment in Open3D")
    parser.add_argument("--frame", type=int, default=0, help="Frame to visualize bboxes for")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)
    args = parser.parse_args()

    scene = Path(args.scene)

    gt_poses = load_poses(scene / "traj.txt")
    pi3_poses = load_poses(scene / "pi3_traj.txt")

    n = min(len(gt_poses), len(pi3_poses))
    print(f"Frames: {n}")

    # --- Compute Sim(3) alignment ---
    s, R, t = compute_alignment(gt_poses[:n], pi3_poses[:n])

    print(f"\n=== Sim(3) alignment: GT world -> Pi3 world ===")
    print(f"  Scale:       {s:.6f}")
    print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"  Rotation:\n    {R[0]}\n    {R[1]}\n    {R[2]}")

    # --- Evaluate alignment quality ---
    gt_pos = np.array([p[:3, 3] for p in gt_poses[:n]])
    pi3_pos = np.array([p[:3, 3] for p in pi3_poses[:n]])
    aligned_gt = (s * (R @ gt_pos.T)).T + t
    errors = np.linalg.norm(aligned_gt - pi3_pos, axis=1)
    print(f"\n=== Alignment residuals (position) ===")
    print(f"  Mean:   {errors.mean():.4f} m")
    print(f"  Median: {np.median(errors):.4f} m")
    print(f"  Max:    {errors.max():.4f} m")
    print(f"  Std:    {errors.std():.4f} m")

    # --- Transform GT bboxes ---
    bbox_dir = scene / "bbox"
    if bbox_dir.exists():
        bbox_files = sorted(bbox_dir.glob("bboxes*_info.json"))
        print(f"\nFound {len(bbox_files)} bbox files")

        # Show example for the specified frame
        if args.frame < len(bbox_files):
            with open(bbox_files[args.frame]) as f:
                data = json.load(f)

            boxes_gt = data["bboxes"]["bbox_3d"]["boxes"]
            print(f"\nFrame {args.frame}: {len(boxes_gt)} objects")

            boxes_pi3 = []
            for box in boxes_gt:
                aabb = box["aabb_xyzmin_xyzmax"]
                aabb_pi3 = transform_bbox_gt_to_pi3(aabb, s, R, t)
                box_pi3 = dict(box)
                box_pi3["aabb_xyzmin_xyzmax"] = aabb_pi3
                boxes_pi3.append(box_pi3)

                print(f"  obj {box['track_id']:3d}  GT: [{aabb[0]:+.2f},{aabb[1]:+.2f},{aabb[2]:+.2f}]-"
                      f"[{aabb[3]:+.2f},{aabb[4]:+.2f},{aabb[5]:+.2f}]")
                print(f"          Pi3: [{aabb_pi3[0]:+.2f},{aabb_pi3[1]:+.2f},{aabb_pi3[2]:+.2f}]-"
                      f"[{aabb_pi3[3]:+.2f},{aabb_pi3[4]:+.2f},{aabb_pi3[5]:+.2f}]")

        # Optionally save all transformed bboxes
        if args.save:
            out_dir = scene / "bbox_pi3"
            out_dir.mkdir(exist_ok=True)

            for bf in bbox_files:
                with open(bf) as f:
                    data = json.load(f)
                for box in data["bboxes"]["bbox_3d"]["boxes"]:
                    box["aabb_xyzmin_xyzmax"] = transform_bbox_gt_to_pi3(
                        box["aabb_xyzmin_xyzmax"], s, R, t)
                out_path = out_dir / bf.name
                with open(out_path, "w") as f:
                    json.dump(data, f)

            # Also transform global bboxes
            global_path = bbox_dir / "global_bboxes.json"
            if global_path.exists():
                with open(global_path) as f:
                    gdata = json.load(f)
                for box in gdata["bboxes"]["bbox_3d"]["boxes"]:
                    box["aabb_xyzmin_xyzmax"] = transform_bbox_gt_to_pi3(
                        box["aabb_xyzmin_xyzmax"], s, R, t)
                with open(out_dir / "global_bboxes.json", "w") as f:
                    json.dump(gdata, f)

            print(f"\nSaved transformed bboxes to {out_dir}")

    # --- Save alignment for reuse ---
    alignment = {
        "scale": float(s),
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "description": "Sim(3): p_pi3 = s * R @ p_gt + t"
    }
    align_path = scene / "gt_to_pi3_alignment.json"
    with open(align_path, "w") as f:
        json.dump(alignment, f, indent=2)
    print(f"\nAlignment saved to {align_path}")

    # --- Visualization ---
    if args.visualize:
        visualize(args, scene, gt_poses[:n], pi3_poses[:n], s, R, t)


def visualize(args, scene, gt_poses, pi3_poses, s, R, t):
    import open3d as o3d
    import cv2

    K = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ])

    n = len(gt_poses)
    geometries = []

    # world frame (Pi3 frame since that's where we're visualizing)
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0]))

    RED = [1.0, 0.0, 0.0]
    GREEN = [0.0, 1.0, 0.0]
    CYAN = [0.0, 1.0, 1.0]

    # --- trajectories ---
    # Pi3 trajectory (native, green)
    pi3_pos = np.array([p[:3, 3] for p in pi3_poses])
    pi3_traj = o3d.geometry.LineSet()
    pi3_traj.points = o3d.utility.Vector3dVector(pi3_pos)
    pi3_traj.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(n-1)])
    pi3_traj.paint_uniform_color(GREEN)
    geometries.append(pi3_traj)

    # GT trajectory BEFORE alignment (red) - will be in wrong place
    gt_pos = np.array([p[:3, 3] for p in gt_poses])
    gt_traj_raw = o3d.geometry.LineSet()
    gt_traj_raw.points = o3d.utility.Vector3dVector(gt_pos)
    gt_traj_raw.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(n-1)])
    gt_traj_raw.paint_uniform_color(RED)
    geometries.append(gt_traj_raw)

    # GT trajectory AFTER alignment (cyan) - should overlap with green
    gt_pos_aligned = (s * (R @ gt_pos.T)).T + t
    gt_traj_aligned = o3d.geometry.LineSet()
    gt_traj_aligned.points = o3d.utility.Vector3dVector(gt_pos_aligned)
    gt_traj_aligned.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(n-1)])
    gt_traj_aligned.paint_uniform_color(CYAN)
    geometries.append(gt_traj_aligned)

    # --- Pi3 depth point clouds (every N frames, greenish) ---
    pi3_depth_dir = scene / "pi3_depth"
    pi3_depth_files = sorted(pi3_depth_dir.glob("*.png"))
    image_dir = scene / "images"
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(image_dir.glob("*.png"))

    frames = list(range(0, min(n, len(pi3_depth_files), len(image_files)), args.every))

    for i in frames:
        # Pi3 depth + Pi3 pose
        depth = cv2.imread(str(pi3_depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        rgb = cv2.cvtColor(cv2.imread(str(image_files[i])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w = depth.shape
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        Z = depth[::args.stride, ::args.stride]
        X = ((xs[::args.stride, ::args.stride] - K[0,2]) * Z / K[0,0])
        Y = ((ys[::args.stride, ::args.stride] - K[1,2]) * Z / K[1,1])

        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        colors = rgb[::args.stride, ::args.stride].reshape(-1, 3)
        valid = pts[:, 2] > 0
        pts, colors = pts[valid], colors[valid]

        # tint greenish
        colors = colors * 0.7 + np.array([0, 1, 0]) * 0.3
        colors = np.clip(colors, 0, 1)

        ones = np.ones((pts.shape[0], 1))
        pts_w = (pi3_poses[i] @ np.concatenate([pts, ones], axis=1).T).T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_w)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

    # --- GT 3D bboxes transformed to Pi3 frame ---
    bbox_dir = scene / "bbox"
    if bbox_dir.exists():
        bbox_files = sorted(bbox_dir.glob("bboxes*_info.json"))
        # show bboxes for a few frames
        for fi in frames:
            if fi >= len(bbox_files):
                continue
            with open(bbox_files[fi]) as f:
                data = json.load(f)
            for box in data["bboxes"]["bbox_3d"]["boxes"]:
                aabb = box["aabb_xyzmin_xyzmax"]
                aabb_pi3 = transform_bbox_gt_to_pi3(aabb, s, R, t)
                xmin, ymin, zmin, xmax, ymax, zmax = aabb_pi3

                corners = np.array([
                    [xmin, ymin, zmin], [xmin, ymin, zmax],
                    [xmin, ymax, zmin], [xmin, ymax, zmax],
                    [xmax, ymin, zmin], [xmax, ymin, zmax],
                    [xmax, ymax, zmin], [xmax, ymax, zmax],
                ])
                lines = [
                    [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],
                    [2,6],[3,7],[4,5],[4,6],[5,7],[6,7]
                ]
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(corners)
                ls.lines = o3d.utility.Vector2iVector(lines)
                ls.paint_uniform_color([1.0, 1.0, 0.0])  # yellow bboxes
                geometries.append(ls)

    print(f"\nVisualization:")
    print(f"  GREEN line  = Pi3 trajectory (native)")
    print(f"  RED line    = GT trajectory (before alignment, misaligned)")
    print(f"  CYAN line   = GT trajectory (after Sim(3) alignment, should overlap green)")
    print(f"  Greenish cloud = Pi3 depth reprojected in Pi3 world")
    print(f"  YELLOW boxes   = GT 3D bboxes transformed to Pi3 world")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Trajectory Alignment: GT -> Pi3 world",
        width=1280, height=720,
    )


if __name__ == "__main__":
    main()
