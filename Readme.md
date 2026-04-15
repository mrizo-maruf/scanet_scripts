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
