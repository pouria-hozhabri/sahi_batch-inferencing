import math
import os
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.nms import TorchNMS



# Type aliases
Point = Tuple[int, int]
Box = Tuple[Point, Point]  # ((x1, y1), (x2, y2))


def generate_slices(
    image_h: int,
    image_w: int,
    slice_h: int,
    slice_w: int,
    overlap_h_pct: float,
    overlap_w_pct: float,
    keep_inside: bool = True,
) -> List[Box]:
    """
    Generate overlapping slice boxes for tiled inference.
    
    Returns list of ((x1, y1), (x2, y2)) boxes where (x2, y2) are exclusive
    coordinates (compatible with img[y1:y2, x1:x2] slicing).
    """
    if not (0 <= overlap_h_pct < 1) or not (0 <= overlap_w_pct < 1):
        raise ValueError("Overlap percentages must be in [0, 1).")
    if slice_h <= 0 or slice_w <= 0:
        raise ValueError("Slice sizes must be positive.")
    if image_h <= 0 or image_w <= 0:
        raise ValueError("Image size must be positive.")

    # Calculate step sizes (slice size minus overlap)
    step_y = max(1, int(round(slice_h * (1.0 - overlap_h_pct))))
    step_x = max(1, int(round(slice_w * (1.0 - overlap_w_pct))))

    # Generate top-left positions
    y_positions = list(range(0, max(1, image_h - slice_h + 1), step_y))
    x_positions = list(range(0, max(1, image_w - slice_w + 1), step_x))

    # Ensure coverage of bottom/right edges
    if keep_inside:
        if (image_h - slice_h) not in y_positions:
            y_positions.append(max(0, image_h - slice_h))
        if (image_w - slice_w) not in x_positions:
            x_positions.append(max(0, image_w - slice_w))

    # Create boxes
    boxes = []
    for y1 in y_positions:
        for x1 in x_positions:
            x2 = x1 + slice_w
            y2 = y1 + slice_h

            if keep_inside:
                # Force boxes to stay fully inside image boundaries
                x2 = min(x2, image_w)
                y2 = min(y2, image_h)
                x1 = max(0, x2 - slice_w)
                y1 = max(0, y2 - slice_h)

            boxes.append(((x1, y1), (x2, y2)))

    # Remove duplicates
    return list(dict.fromkeys(boxes))


def draw_polygons_cv2(
    image: np.ndarray,
    final_results: List[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    draw_label: bool = True,
) -> np.ndarray:
    """
    Draw rotated bounding box polygons on an image.
    
    Args:
        image: BGR image (OpenCV format)
        final_results: Post-NMS detection results
        color: BGR color tuple
        thickness: Line thickness
        draw_label: Whether to draw class name and confidence
        
    Returns:
        Image with polygons drawn
    """
    img = image.copy()

    for result in final_results:
        bbox = np.array(result["bbox_xyxy"], dtype=np.int32)

        # Draw polygon
        cv2.polylines(img, [bbox], isClosed=True, color=color, thickness=thickness)

        # Draw label
        if draw_label:
            x, y = bbox[0]
            label = result["class_name"]
            conf = result.get("conf")
            if conf is not None:
                label = f"{label} {conf:.2f}"
            cv2.putText(
                img, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

    return img

def is_edge_stitched(pts: torch.Tensor, slice_box: Box, eps: float = 5.0) -> bool:
    """
    Return True if any polygon point is within eps of the slice border.
    pts are in slice-local coordinates (before offset is applied).
    """
    (x1, y1), (x2, y2) = slice_box
    slice_w = max(1.0, float(x2 - x1))
    slice_h = max(1.0, float(y2 - y1))

    max_x = slice_w - 1.0 - eps
    max_y = slice_h - 1.0 - eps
    edge_mask = (
        (pts[:, 0] <= eps)
        | (pts[:, 0] >= max_x)
        | (pts[:, 1] <= eps)
        | (pts[:, 1] >= max_y)
    )
    return bool(edge_mask.any().item())

def process_obb_results(results, slices, batch_start_idx):
    """Process oriented bounding box detection results."""
    detections = []

    for j, result in enumerate(results):
        obb = result.obb
        if obb is None or obb.xyxyxyxy.numel() == 0:
            continue

        slice_box = slices[batch_start_idx + j]
        (x1, y1), (x2, y2) = slice_box
        slice_w = max(1.0, float(x2 - x1))
        slice_h = max(1.0, float(y2 - y1))

        eps = 5.0
        max_x = slice_w - 1.0 - eps
        max_y = slice_h - 1.0 - eps

        conf_colibr = 0.05

        device = obb.xyxyxyxy.device
        dtype = obb.xyxyxyxy.dtype
        offset_xy = torch.tensor(slice_box[0], device=device, dtype=dtype)
        offset_xyxyxyxy = offset_xy.repeat(4)
        offset_xywhr = torch.tensor(
            [offset_xy[0], offset_xy[1], 0.0, 0.0, 0.0],
            device=device,
            dtype=obb.xywhr.dtype,
        )

        pts = obb.xyxyxyxy.view(-1, 4, 2)
        edge_mask = (
            (pts[..., 0] <= eps)
            | (pts[..., 0] >= max_x)
            | (pts[..., 1] <= eps)
            | (pts[..., 1] >= max_y)
        )
        edge_any = edge_mask.any(dim=1)
        conf_adj = obb.conf - edge_any.to(obb.conf.dtype) * conf_colibr

        xyxyxyxy = obb.xyxyxyxy
        if xyxyxyxy.dim() == 3 and xyxyxyxy.size(-1) == 2:
            xyxyxyxy = xyxyxyxy + offset_xyxyxyxy.view(1, 4, 2)
        else:
            xyxyxyxy = xyxyxyxy + offset_xyxyxyxy
        xywhr = obb.xywhr + offset_xywhr
        cls_int = obb.cls.to(torch.int64).tolist()
        conf_list = conf_adj.tolist()

        for idx in range(len(cls_int)):
            cls_id = cls_int[idx]
            detections.append({
                "class_name": result.names[cls_id],
                "class_id": cls_id,
                "bbox_xyxy": xyxyxyxy[idx],
                "bbox_xywhr": xywhr[idx],
                "conf": float(conf_list[idx]),
            })

    return detections
def process_hbb_results(results, slices, batch_start_idx):
    """Process horizontal bounding box detection results."""
    detections = []
    
    for j, result in enumerate(results):
        if result.boxes is None:
            continue

        boxes = result.boxes
        box_xyxy = boxes.xyxy
        box_xywh = boxes.xywh
        if box_xyxy.numel() == 0:
            continue

        slice_box = slices[batch_start_idx + j]
        (x1, y1), (x2, y2) = slice_box
        slice_w = max(1.0, float(x2 - x1))
        slice_h = max(1.0, float(y2 - y1))

        eps = 5.0
        max_x = slice_w - 1.0 - eps
        max_y = slice_h - 1.0 - eps

        conf_colibr = 0.05

        slice_offset = slice_box[0]
        device = box_xyxy.device
        offset_xyxy = torch.tensor(
            [slice_offset[0], slice_offset[1], slice_offset[0], slice_offset[1]],
            device=device,
            dtype=box_xyxy.dtype,
        )
        offset_xywh = torch.tensor(
            [slice_offset[0], slice_offset[1], 0.0, 0.0],
            device=device,
            dtype=box_xywh.dtype,
        )

        edge_mask = (
            (box_xyxy[:, 0] <= eps)
            | (box_xyxy[:, 2] >= max_x)
            | (box_xyxy[:, 1] <= eps)
            | (box_xyxy[:, 3] >= max_y)
        )
        conf_adj = boxes.conf - edge_mask.to(boxes.conf.dtype) * conf_colibr

        xyxy = box_xyxy + offset_xyxy
        xywh = box_xywh + offset_xywh
        cls_int = boxes.cls.to(torch.int64).tolist()
        conf_list = conf_adj.tolist()

        for idx in range(len(cls_int)):
            cls_id = cls_int[idx]
            detections.append({
                "class_name": result.names[cls_id],
                "class_id": cls_id,
                "bbox_xyxy": xyxy[idx],
                "bbox_xywhr": xywh[idx],
                "conf": float(conf_list[idx]),
            })
    
    return detections


def apply_nms_obb(detections, slices, slice_h, slice_w, iou_thresh):
    """Apply NMS to oriented bounding box detections."""

    if not detections:
        return []

    device = detections[0]["bbox_xywhr"].device
    
    boxes = torch.stack([d["bbox_xywhr"] for d in detections])
    scores = torch.tensor([d["conf"] for d in detections], device=device)
    classes = torch.tensor([d["class_id"] for d in detections], device=device)
    
    # Apply class-specific offsets for class-agnostic NMS
    max_wh = math.ceil(math.sqrt(len(slices))) * max(slice_h, slice_w) + 10000
    offsets = classes[:, None] * max_wh
    
    boxes_for_nms = torch.cat([
        boxes[:, :2] + offsets,  # cx, cy + class offset
        boxes[:, 2:4],           # w, h
        boxes[:, 4:5]            # rotation
    ], dim=1)
    
    keep = TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold=iou_thresh, iou_func=batch_probiou)
    
    # Convert to final format (only bbox_xyxy and class_name)
    final_results = []
    for idx in keep.tolist():
        item = detections[idx]
        final_results.append({
            "bbox_xyxy": [[int(x), int(y)] for x, y in item["bbox_xyxy"].tolist()],
            "bbox_xywhr": [int(v) for v in item["bbox_xywhr"].tolist()],
            "class_name": item["class_name"],
            "conf": item["conf"]
        })
    
    return final_results


def apply_nms_hbb(detections, slices, slice_h, slice_w, iou_thresh):
    """Apply NMS to horizontal bounding box detections."""
    if not detections:
        return []
    device = detections[0]["bbox_xyxy"].device
    
    boxes = torch.stack([d["bbox_xyxy"] for d in detections]).to(device)
    scores = torch.tensor([d["conf"] for d in detections], device=device, dtype=boxes.dtype)
    classes = torch.tensor([d["class_id"] for d in detections], device=device)
    
    # Apply class-specific offsets for class-agnostic NMS
    max_wh = math.ceil(math.sqrt(len(slices))) * max(slice_h, slice_w) + 10000
    offsets = classes[:, None] * max_wh
    
    boxes_for_nms = torch.cat([
        boxes[:, 0:2] + offsets,  # x1, y1 + class offset
        boxes[:, 2:4] + offsets,  # x2, y2 + class offset
    ], dim=1)
    
    keep = TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold=iou_thresh, iou_func=box_iou)
    
    # Convert to final format (only bbox_xyxy and class_name)
    final_results = []
    for idx in keep.tolist():
        item = detections[idx]
        x1, y1, x2, y2 = [int(v) for v in item["bbox_xyxy"].tolist()]
        final_results.append({
            "bbox_xyxy": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "bbox_xywhr": [int(v) for v in item["bbox_xywhr"].tolist()],
            "class_name": item["class_name"],
            "conf": item["conf"]
        })
    
    return final_results






def polygon_overlap_ratio(poly1_pts, poly2_pts) -> float:
    """
    Compute overlap ratio between two polygons as:
        intersection_area / area(poly1)
    """
    poly1 = Polygon(poly1_pts)
    poly2 = Polygon(poly2_pts)

    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)

    if poly1.is_empty or poly1.area == 0:
        return 0.0

    inter_area = poly1.intersection(poly2).area
    return float(inter_area / poly1.area)



def delete_overlapping_det(final_results, overlap_thresh: float = 0.5):
    """
    Remove detections that are mostly covered by another detection.

    A detection is removed if:
        intersection_area / area(candidate) >= overlap_thresh
    """
    if not final_results:
        return []
    class_groups = {}
    for res in final_results:
        class_name = res["class_name"]
        bucket = class_groups.get(class_name)
        if bucket is None:
            bucket = []
            class_groups[class_name] = bucket
        bucket.append(res)

    keep = []
    for group in class_groups.values():
        if len(group) == 1:
            keep.append(group[0])
            continue
        group.sort(key=lambda r: r["conf"], reverse=True)
        kept_group = []
        for res in group:
            candidate = res["bbox_xyxy"]
            drop = False
            for kept in kept_group:
                if polygon_overlap_ratio(candidate, kept["bbox_xyxy"]) >= overlap_thresh:
                    drop = True
                    break
            if not drop:
                kept_group.append(res)
        keep.extend(kept_group)

    return keep






def SliceDetect(
    image: str = "sample.avi",
    model = "best.pt",
    threshold: float = 0.5,
    slice_h: int = 640,
    slice_w: int = 640,
    overlap_h: float = 0.2,
    overlap_w: float = 0.2,
    batch: int = 5,
    iou_thresh: float = 0.5,
    device: int = 0,
    tta: bool = False,
    vis: bool = False,
    overlap_thresh = 0.5,
    **kwargs
) -> List[dict]:
    """
    Perform SAHI (Slicing Aided Hyper Inference) on large images.
    
    Args:
        image: Input image (numpy array or path to image file)
        model: YOLO model path or a preloaded YOLO model instance
        threshold: Confidence threshold
        slice_h: Slice height
        slice_w: Slice width
        overlap_h: Vertical overlap ratio
        overlap_w: Horizontal overlap ratio
        batch: Batch size for inference
        iou_thresh: IoU threshold for NMS
        device: GPU device ID
        tta: Test-time augmentation
        vis: Visualize results
        
    Returns:
        List of detection dictionaries with bbox_xyxy and class_name
    """
    t0 = time.time()
    image_name = None
    # Load image
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = cv2.imread(image)
        image_name = os.path.basename(image)
    if img is None:
        raise FileNotFoundError(f"Could not load image from: {image}")
    h, w, _ = img.shape
    
    # Generate slices
    slices = generate_slices(h, w, slice_h, slice_w, overlap_h, overlap_w)


    # ## draw gridlines of tiling
    # if vis:
    #     img_slices = img.copy()
    #     color = (255, 0, 0)
    #     thickness = 1
    #     for ((x1, y1), (x2, y2)) in slices:
    #         cv2.rectangle(img_slices, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
    #     cv2.imwrite("tiling.png",img_slices)
    print(slices)
    

    t1 = time.time()
    

    if model.task == "obb":
        is_obb = True
    elif model.task == "detect":
        is_obb = False
    else:
        raise ValueError(f"Unsupported model task: {model.task}. Expected 'obb' or 'detect'.")
    
    # Run inference on slices
    all_detections = []
    for i in range(0, len(slices), batch):
        batch_slices = slices[i:i + batch]
        batch_imgs = [img[y1:y2, x1:x2] for ((x1, y1), (x2, y2)) in batch_slices]
        # Allow callers to forward extra YOLO inference args via SliceDetect(**kwargs).
        infer_kwargs = {
            "source": batch_imgs,
            "device": device,
            "conf": threshold,
            "iou": 1,
            "agnostic_nms": False,
            "augment": tta,
        }
        infer_kwargs.update(kwargs)
        results = model(**infer_kwargs)
        
        # Process results based on model type
        if is_obb:
            all_detections.extend(process_obb_results(results, slices, i))
        else:
            all_detections.extend(process_hbb_results(results, slices, i))
    
    t2 = time.time()
    
    # Apply NMS
    if not all_detections:
        print("No detections found")
        final_results = []
    else:
        if is_obb:
            final_results = apply_nms_obb(all_detections, slices, slice_h, slice_w, iou_thresh)
        else:
            final_results = apply_nms_hbb(all_detections, slices, slice_h, slice_w, iou_thresh)

        final_results = delete_overlapping_det(final_results = final_results,overlap_thresh= overlap_thresh )
    
    t3 = time.time()


    # Visualization
    if vis:
        root = "runs/image_inference_results"
        os.makedirs(root, exist_ok=True)
        if final_results :
            img_drawn = draw_polygons_cv2(img, final_results)
            if image_name is not None:
                output_path = os.path.join(root, os.path.splitext(image_name)[0] + ".jpg")
            else:
                output_path = os.path.join(root, "result.jpg")
            cv2.imwrite(output_path, img_drawn)
            print(f"Saved visualization with {len(final_results)} detections")
        else:
            print("Nothing has been detected")
    
    t4 = time.time()
    
    # Print timing statistics
    print(f"Slice time: {t1 - t0:.3f}s")
    print(f"Inference time: {t2 - t1:.3f}s")
    print(f"Postprocess time: {t3 - t2:.3f}s")
    print(f"Visualization time: {t4 - t3:.3f}s")
    print(f"Total time: {t4 - t0:.3f}s")


    return final_results


def video_inference(**kwargs):
    video_path = kwargs.pop("image", None)
    vis = bool(kwargs.pop("vis", False))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    out_path = None
    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if vis:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        runs_dir = "runs"
        os.makedirs(runs_dir, exist_ok=True)
        max_inference_idx = 0
        with os.scandir(runs_dir) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name.startswith("inference_"):
                    suffix = entry.name[10:]
                    if suffix.isdigit():
                        idx = int(suffix)
                        if idx > max_inference_idx:
                            max_inference_idx = idx

        next_inference_idx = max_inference_idx + 1
        while True:
            run_dir = os.path.join(runs_dir, f"inference_{next_inference_idx}")
            try:
                os.makedirs(run_dir)
                break
            except FileExistsError:
                next_inference_idx += 1

        out_path = os.path.join(run_dir, f"{base_name}.mp4")

    frame_results = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx}")
        results = SliceDetect(image=frame, **kwargs)
        frame_results.append((frame_idx, results))

        if vis:
            frame_to_write = draw_polygons_cv2(frame, results) if results else frame
            if writer is None:
                h, w = frame_to_write.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    cap.release()
                    raise IOError(f"Could not create output video writer: {out_path}")
            writer.write(frame_to_write)

        frame_idx += 1

    if writer is not None:
        writer.release()
        print(f"Saved inferenced video: {out_path}")
    cap.release()
    return frame_results



def inference_sliced(**kwargs):
    #read model if its a path, else use the provided model instance
    model = kwargs.get("model")
    if model is None:
        raise ValueError("model is required")
    if isinstance(model, (str, os.PathLike)):
        kwargs["model"] = YOLO(str(model))
    elif not hasattr(model, "task"):
        raise TypeError("model must be a YOLO model instance or a valid model path")
    
    #check if input is video or image based on extension
    video_exts = ["mp4", "avi", "mov", "mkv", "m4v", "wmv", "flv", "webm", "mpeg", "mpg", "ts", "m2ts", "mts", "3gp", "ogv"]
    image_path = kwargs.get("image")
    if image_path is None:
        raise ValueError("image_path is required")
    if image_path.split(".")[-1].lower()  in video_exts:
        return video_inference(**kwargs)
    elif image_path.split(".")[-1].lower() in ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "gif"]:
        return SliceDetect(**kwargs)
    elif isinstance(image_path, (str, os.PathLike)) and os.path.isdir(image_path):
        images = os.listdir(image_path)
        images = [os.path.join(image_path, img) for img in images if img.split(".")[-1].lower() in ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "gif"]]
        results = {}
        for img in images:
            kwargs["image"] = img
            results[os.path.basename(img)] = SliceDetect(**kwargs)
        return results
    else:
        raise ValueError(f"Unsupported file type or path: {image_path}")



if __name__ == "__main__":
    t0 = time.time()
    model = YOLO("airplane_det_11m.pt")
    results = inference_sliced(
        image= "sample.mp4",
        model= model,
        threshold= 0.5,
        slice_h= 1024,
        slice_w= 1024,
        overlap_h= 0.2,
        overlap_w= 0.2,
        batch= 4,
        iou_thresh= 0.9,
        device= 0,
        tta= False,
        vis= True,
        overlap_thresh = 0.9
    )
    t1 = time.time()
    print(f"\nTotal detections: {len(results)}")
    print(results)
    print(f"Inference completed in {t1 - t0:.3f} seconds")
