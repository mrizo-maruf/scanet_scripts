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
    parser = argparse.ArgumentParser(description="Visualize GT vs Pi3 depth reprojection")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene folder")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize (0-based)")
    parser.add_argument("--all", action="store_true", help="Visualize all frames (slower)")
    parser.add_argument("--stride", type=int, default=4, help="Pixel stride for subsampling")
    parser.add_argument("--every", type=int, default=80, help="When --all, take every N-th frame")
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)
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

    RED = [1.0, 0.0, 0.0]
    GREEN = [0.0, 1.0, 0.0]
    BLUE = [0.0, 0.0, 1.0]
    YELLOW = [1.0, 1.0, 0.0]

    geometries = []

    # world frame axes
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(world_frame)

    if args.all:
        frames = list(range(0, n, args.every))
        print(f"Visualizing {len(frames)} frames (every {args.every}-th)")
    else:
        frames = [args.frame]
        print(f"Visualizing frame {args.frame}")

    for i in frames:
        print(f"  Processing frame {i}...")

        # GT depth + GT pose -> reddish RGB
        gt_pcd = depth_to_world_pcd(gt_depth_files[i], gt_poses[i], K, image_files[i],
                                     tint=RED, tint_strength=0.3, stride=args.stride)
        geometries.append(gt_pcd)

        # Pi3 depth + Pi3 pose -> greenish RGB
        pi3_pcd = depth_to_world_pcd(pi3_depth_files[i], pi3_poses[i], K, image_files[i],
                                      tint=GREEN, tint_strength=0.3, stride=args.stride)
        geometries.append(pi3_pcd)

        # camera frustums
        gt_cam = create_camera_frustum(gt_poses[i], K, RED, scale=0.1)
        pi3_cam = create_camera_frustum(pi3_poses[i], K, GREEN, scale=0.1)
        geometries.append(gt_cam)
        geometries.append(pi3_cam)

    # also draw full camera trajectories as lines
    gt_positions = [gt_poses[i][:3, 3] for i in range(n)]
    pi3_positions = [pi3_poses[i][:3, 3] for i in range(n)]

    gt_traj_ls = o3d.geometry.LineSet()
    gt_traj_ls.points = o3d.utility.Vector3dVector(np.array(gt_positions))
    gt_traj_ls.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(n-1)])
    gt_traj_ls.paint_uniform_color(RED)
    geometries.append(gt_traj_ls)

    pi3_traj_ls = o3d.geometry.LineSet()
    pi3_traj_ls.points = o3d.utility.Vector3dVector(np.array(pi3_positions))
    pi3_traj_ls.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(n-1)])
    pi3_traj_ls.paint_uniform_color(GREEN)
    geometries.append(pi3_traj_ls)

    print(f"Launching Open3D viewer with {len(geometries)} geometries...")
    print("  Reddish tint  = GT depth + GT pose")
    print("  Greenish tint = Pi3 depth + Pi3 pose")
    print("  Coordinate frame = world origin")
    print("  Lines = camera trajectories (red=GT, green=Pi3)")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="GT (red) vs Pi3 (green) Depth Reprojection",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
