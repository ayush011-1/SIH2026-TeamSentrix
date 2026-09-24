"""
Lightweight Multi-Object Tracker (IoU & Motion Centroid Tracking)
Used for tracking detected incidents across consecutive video frames.
Team Sentrix - SIH 2026
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class TrackedObject:
    """Represents a single tracked bounding box across frames."""
    def __init__(self, track_id: int, box: List[int], confidence: float, class_name: str):
        self.track_id = track_id
        self.box = list(box)  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_name = class_name
        self.missed_frames = 0
        self.history: List[Tuple[int, int]] = [self.get_center()]

    def get_center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, box: List[int], confidence: float) -> None:
        self.box = list(box)
        self.confidence = confidence
        self.missed_frames = 0
        self.history.append(self.get_center())
        if len(self.history) > 30:
            self.history.pop(0)


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


class IoUTracker:
    """Lightweight bounding box tracker assigning persistent track IDs across frames."""
    def __init__(self, iou_threshold: float = 0.25, max_lost: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: List[TrackedObject] = []

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Input: list of detections, each with 'box': [x1, y1, x2, y2], 'confidence': float, 'class_name': str
        Output: list of detections with 'track_id' and 'history' added.
        """
        updated_detections: List[Dict[str, Any]] = []
        matched_track_indices = set()
        matched_det_indices = set()

        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = compute_iou(track.box, det["box"])

            # Match greedily
            while True:
                max_val = float(np.max(iou_matrix))
                if max_val < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                t_idx, d_idx = int(t_idx), int(d_idx)
                if t_idx in matched_track_indices or d_idx in matched_det_indices:
                    iou_matrix[t_idx, d_idx] = -1.0
                    continue

                track = self.tracks[t_idx]
                det = detections[d_idx]
                track.update(det["box"], det["confidence"])

                det_res = dict(det)
                det_res["track_id"] = track.track_id
                det_res["history"] = list(track.history)
                updated_detections.append(det_res)

                matched_track_indices.add(t_idx)
                matched_det_indices.add(d_idx)
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Increment missed frame counter for unmatched tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track.missed_frames += 1

        # Drop stale tracks
        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_lost]

        # Register new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_track = TrackedObject(
                    self.next_id,
                    det["box"],
                    det["confidence"],
                    det.get("class_name", "incident"),
                )
                self.next_id += 1
                self.tracks.append(new_track)

                det_res = dict(det)
                det_res["track_id"] = new_track.track_id
                det_res["history"] = list(new_track.history)
                updated_detections.append(det_res)

        return updated_detections
