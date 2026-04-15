import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class GTObject:
    track_id: int
    mask: Optional[np.ndarray] = None
    class_name: str = "unknown"


@dataclass
class FrameData:
    frame_idx: int
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    mask_map: Optional[np.ndarray] = None
    gt_objects: List[GTObject] = field(default_factory=list)


class IGGTDataLoader:

    def __init__(self, scene_path):

        self.scene_path = Path(scene_path)

        self.rgb_dir = self.scene_path / "images"
        self.depth_dir = self.scene_path / "pi3_depth"
        self.mask_dir = self.scene_path / "masks"
        self.benchmark_dir = self.scene_path / f"3D Tracking Benchmark/scannetpp/{self.scene_path.stem}/images"

        self.rgb_files = sorted(self.rgb_dir.glob("*"))
        self.depth_files = sorted(self.depth_dir.glob("*"))
        self.mask_files = sorted(self.mask_dir.glob("*.npy"))

        self.frame_count = min(len(self.rgb_files), len(self.mask_files))

    def __len__(self):
        return self.frame_count

    def load_rgb(self, frame_idx):
        path = self.rgb_files[frame_idx]
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    def load_depth(self, frame_idx):
        path = self.depth_files[frame_idx]
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return depth.astype(np.float32) / 1000

    def load_mask(self, frame_idx):
        path = self.mask_files[frame_idx]
        mask = np.load(path)
        return mask

    def get_bench_gt_objects(self, frame_name):
        mask = np.load(self.benchmark_dir / f"{frame_name}_label.npy")
        if mask is None:
            return []
        gt_objects = []
        bench_ids = set()
        with open(self.benchmark_dir / f"{frame_name}.json", "r") as f:
            data = json.load(f)
            for object in data["objects"]:
                obj_id = object["group"]
                if obj_id in bench_ids or obj_id == 0:
                    continue
                obj_label = object["category"]
                obj_mask = (mask == obj_id).astype(np.uint8) * 255
                gt_objects.append(
                    GTObject(
                        track_id=int(obj_id),
                        mask=obj_mask,
                        class_name=obj_label,
                    )
                )
                bench_ids.add(obj_id)
        return gt_objects
    
    def get_gt_objects(self, frame_idx):
        mask = self.load_mask(frame_idx)
        if mask is None:
            return []
        gt_objects = []
        ids = np.unique(mask)
        for obj_id in ids:
            if obj_id == 0:
                continue
            obj_mask = (mask == obj_id).astype(np.uint8) * 255
            gt_objects.append(
                GTObject(
                    track_id=int(obj_id),
                    mask=obj_mask
                )
            )
        return gt_objects

    def get_frame_data(self, frame_idx):
        rgb = self.load_rgb(frame_idx)
        depth = self.load_depth(frame_idx)
        mask = self.load_mask(frame_idx)
        objects = self.get_gt_objects(frame_idx)

        return FrameData(
            frame_idx=frame_idx,
            rgb=rgb,
            depth=depth,
            mask_map=mask,
            gt_objects=objects
        )