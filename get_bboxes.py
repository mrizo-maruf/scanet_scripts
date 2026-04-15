import numpy as np
import json
from pathlib import Path
import cv2
import argparse
import open3d as o3d
import numpy as np

def clean_pointcloud(obj_pts):
    if len(obj_pts) < 50:
        return obj_pts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_pts)

    # --- основной фильтр ---
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    cleaned = np.asarray(pcd.points)

    return cleaned



def load_poses(traj_path):
    poses = []
    with open(traj_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            pose = np.array(vals).reshape(4, 4)
            poses.append(pose)
    return poses


def backproject_fast(depth, K):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    return np.stack([X, Y, Z], axis=-1)


def transform_points(pts, pose):
    pts_flat = pts.reshape(-1, 3)
    ones = np.ones((pts_flat.shape[0], 1))
    pts_h = np.concatenate([pts_flat, ones], axis=1)

    pts_w = (pose @ pts_h.T).T[:, :3]
    return pts_w.reshape(pts.shape)

def compute_bbox(pts, mask, min_points=100):
    obj_pts = pts[mask]

    if obj_pts.shape[0] < min_points:
        return None

    obj_pts = clean_pointcloud(obj_pts)

    if obj_pts.shape[0] < min_points:
        return None

    # --- 8. bbox ---
    mins = obj_pts.min(axis=0)
    maxs = obj_pts.max(axis=0)

    return [
        float(mins[0]), float(mins[1]), float(mins[2]),
        float(maxs[0]), float(maxs[1]), float(maxs[2])
    ]

def update_global(global_boxes, obj_id, bbox):
    if obj_id not in global_boxes:
        global_boxes[obj_id] = bbox
        return

    g = global_boxes[obj_id]
    global_boxes[obj_id] = [
        min(g[0], bbox[0]),
        min(g[1], bbox[1]),
        min(g[2], bbox[2]),
        max(g[3], bbox[3]),
        max(g[4], bbox[4]),
        max(g[5], bbox[5]),
    ]


def save_frame(output_dir, frame_id, boxes):
    data = {
        "bboxes": {
            "bbox_3d": {
                "boxes": boxes
            }
        }
    }

    fname = output_dir / f"bboxes{frame_id:06d}_info.json"
    with open(fname, "w") as f:
        json.dump(data, f)


def generate_gt(scene_path, K, percentile=1.0, min_points=100):

    scene_path = Path(scene_path)

    depth_dir = scene_path / "gt_depth"
    mask_dir = scene_path / "masks"
    traj_path = scene_path / "traj.txt"

    out_dir = scene_path / "bbox"
    out_dir.mkdir(exist_ok=True)

    poses = load_poses(traj_path)

    depth_files = sorted(depth_dir.glob("*.png"))
    mask_files = sorted(mask_dir.glob("*.npy"))

    assert len(depth_files) == len(mask_files), "Depth и mask не совпадают"
    assert len(depth_files) == len(poses), "Depth и poses не совпадают"

    global_boxes = {}

    for i, (d_path, m_path) in enumerate(zip(depth_files, mask_files)):
        print(f"[Frame {i}]")

        depth = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth /= 1000.0  # мм → м

        mask = np.load(m_path)
        pose = poses[i]

        pts_cam = backproject_fast(depth, K)
        pts_world = transform_points(pts_cam, pose)

        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]

        frame_boxes = []

        for obj_id in object_ids:
            bbox = compute_bbox(
                pts_world,
                mask == obj_id,
                min_points=min_points
            )

            if bbox is None:
                continue

            box = {
                "track_id": int(obj_id),
                "bbox_id": int(obj_id),
                "prim_path": f"/Object/{obj_id}",
                "aabb_xyzmin_xyzmax": bbox
            }

            frame_boxes.append(box)
            update_global(global_boxes, obj_id, bbox)

        save_frame(out_dir, i, frame_boxes)

    # global bbox
    global_data = {
        "bboxes": {
            "bbox_3d": {
                "boxes": [
                    {
                        "track_id": int(obj_id),
                        "bbox_id": int(obj_id),
                        "prim_path": f"/Object/{obj_id}",
                        "aabb_xyzmin_xyzmax": bbox
                    }
                    for obj_id, bbox in global_boxes.items()
                ]
            }
        }
    }

    with open(out_dir / "global_bboxes.json", "w") as f:
        json.dump(global_data, f)

    print("✅ Done")


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="/Users/macbook/clear_yolo_sgg/0cf2e9402d", help="Path to scene folder")

    # можно не задавать — будут дефолты
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)

    parser.add_argument("--percentile", type=float, default=1.0)
    parser.add_argument("--min_points", type=int, default=100)

    args = parser.parse_args()

    K = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ])

    generate_gt(
        scene_path=args.scene,
        K=K,
        percentile=args.percentile,
        min_points=args.min_points
    )


if __name__ == "__main__":
    main()