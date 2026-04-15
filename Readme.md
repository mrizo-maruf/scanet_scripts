```
python visualize_depth_copy.py --scene /home/yehia/rizo/scanetpp/scene0a76e06478 --frame 90

python visualize_depth_copy.py --scene /home/yehia/rizo/scanetpp/scene0a76e06478 --all

python visualize_depth_copy.py --scene /home/yehia/rizo/scanetpp/scene0a76e06478 --frame 50

python visualize_depth.py --scene /home/yehia/rizo/scanetpp/scene0a76e06478 --frame 50

python visualize_gt_labels.py --scene /home/yehia/rizo/scanetpp/scene0a76e06478 --frame 50
```

### Aligning
aligning based on 1st frame, and visualizing on-top of each other and showing 3d bboxes

How it works:
* Collects pixel-wise depth correspondences from the single `--align_frame`
* Back-projects both GT and Pi3 depth into their respective world frames
* Estimates the transform (rigid or similarity) mapping Pi3 world → GT world
* Applies that transform to all Pi3 point clouds, camera frustums, and the Pi3 coordinate frame

```
# Rigid alignment (R, t only):

python align_and_visualize.py --scene /path/to/scene --align_frame 0

# Similarity alignment (s, R, t):
python align_and_visualize.py --scene /path/to/scene --align_frame 0 --scale
```

Flag	Purpose
`--align_frame`	Which frame to estimate alignment from (default: 0)
`--scale`	Enable scale estimation (Umeyama similarity). Without it, rigid only (scale=1)
`--frame / --all / --consecutive`	Which frames to visualize (same as visualize_depth_copy.py)
`--stride, --pcd_voxel_size`, etc.	Subsampling controls

---

### Benchmarking Configs

`benchmark_configs.py` compares different Pi3 depth prediction configs (varying chunk size and overlap) against GT depth across all scenes in a dataset.

#### How it works

1. **Alignment (frame 0)** — For each config, collects pixel-wise depth correspondences from frame 0, estimates a rigid (R, t) and a similarity (s, R, t) transform mapping Pi3 world → GT world.
2. **Full-scene reconstruction** — Reconstructs the complete scene from ALL frames for both GT and Pi3. Each frame's depth is backprojected to 3D, transformed to world coordinates, and merged into a single point cloud. The merged cloud is voxel-downsampled.
3. **Alignment applied** — The estimated transform is applied to the full Pi3 reconstruction.
4. **Metric computation** — The aligned Pi3 cloud is compared against the GT cloud using nearest-neighbour distances (KD-tree).
5. **Averaging** — Metrics are averaged over all scenes.
6. **Plotting** — Two plots are generated: one for rigid alignment, one for similarity alignment, each showing all metrics as bar charts across configs.

#### Expected folder structure

```
dataset_root/
  scene_A/
    traj.txt               # GT trajectory
    gt_depth/              # GT depth PNGs
    pi3_depth_12_6/        # predicted depth PNGs (chunk=12, overlap=6)
    pi3_traj_12_6.txt      # predicted trajectory
    pi3_depth_8_4/
    pi3_traj_8_4.txt
    pi3_depth_4_2/
    pi3_traj_4_2.txt
  scene_B/
    ...
```

#### Usage

```bash
# Auto-discover configs from scene folders
python benchmark_configs.py --dataset /path/to/scannetpp_scenes

# Explicit configs
python benchmark_configs.py --dataset /path/to/scannetpp_scenes --configs 12_6 8_4 4_2

# Save plots to disk
python benchmark_configs.py --dataset /path/to/scannetpp_scenes --save ./plots

# Custom alignment frame and subsampling
python benchmark_configs.py --dataset /path/to/scannetpp_scenes --align_frame 0 --stride 4 --voxel_size 0.02
```

#### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Path to dataset root containing scene folders |
| `--configs` | auto-discover | Config suffixes to benchmark, e.g. `12_6 8_4 4_2` |
| `--align_frame` | `0` | Frame index used for alignment |
| `--stride` | `4` | Pixel stride for depth backprojection (higher = faster, coarser) |
| `--voxel_size` | `0.02` | Voxel size in meters for downsampling merged reconstructions |
| `--max_eval_points` | `500000` | Max points per cloud for metric computation (caps KD-tree cost) |
| `--save` | None | Directory to save plot PNGs |
| `--fx/fy/cx/cy` | ScanNet++ defaults | Camera intrinsics |

#### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** (m) ↓ | Mean distance from each predicted point to its nearest GT point | How close the prediction is to GT |
| **Completion** (m) ↓ | Mean distance from each GT point to its nearest predicted point | How much of the GT surface is covered |
| **Chamfer** (m) ↓ | (Accuracy + Completion) / 2 | Symmetric overall reconstruction quality |
| **RMSE** (m) ↓ | √(mean of squared pred→GT distances) | Like accuracy but penalizes large errors more |
| **Precision @5cm** ↑ | Fraction of predicted points within 5 cm of GT | What fraction of the prediction is correct |
| **Recall @5cm** ↑ | Fraction of GT points within 5 cm of predicted | What fraction of GT is reconstructed |
| **F1 @5cm** ↑ | Harmonic mean of Precision and Recall | Balanced quality score |

↓ = lower is better, ↑ = higher is better
