#!/usr/bin/env python3
"""
Evaluate FP32 vs INT8 mAP for a Hailo-optimized HAR file on COCO val2017
(13-keypoint upper-body subset).

Streams predictions to JSONL files on disk to avoid OOM on large datasets.
If a run crashes, the FP32 JSONL on disk can be reused on retry.
"""
import sys, types
sys.modules['hailo_platform'] = types.ModuleType('hailo_platform')
sys.modules['hailo_platform'].HEF = None

import argparse, json, os, gc
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from hailo_sdk_client import ClientRunner, InferenceContext
from custom_pose_estimation_utils import PoseEstPostProcessing

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Indices to KEEP from the 17 COCO keypoints (drop 13-16: knees + ankles)
KEEP_KPT_INDICES = list(range(13))

# The 12 skeleton edges for the 13-keypoint upper-body subset
UPPER_BODY_SKELETON = [
    [1, 2], [2, 4], [1, 3], [3, 5],
    [6, 7], [6, 8], [8, 10], [7, 9], [9, 11],
    [6, 12], [7, 13], [12, 13]
]

# Standard COCO keypoint names, first 13 only
UPPER_BODY_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip"
]


def get_filtered_gt(original_gt_json):
    """
    Create a 13-keypoint version of the COCO GT JSON if it doesn't already exist.
    Strips keypoints at indices 13-16 (knees + ankles) from every annotation.
    Returns the path to the filtered file.
    """
    filtered_path = original_gt_json.replace(".json", "_13kpt.json")

    if os.path.exists(filtered_path):
        print(f"  Using cached 13-keypoint GT: {filtered_path}")
        return filtered_path

    print(f"  Creating 13-keypoint GT from {original_gt_json}...")
    import copy

    with open(original_gt_json) as f:
        gt = json.load(f)

    gt_filtered = copy.deepcopy(gt)

    # Update category metadata
    for cat in gt_filtered['categories']:
        if cat['name'] == 'person':
            cat['keypoints'] = UPPER_BODY_KPT_NAMES
            cat['skeleton'] = UPPER_BODY_SKELETON
            cat['num_keypoints'] = 13  # not standard but informational

    # Strip keypoints from every annotation
    for ann in gt_filtered['annotations']:
        if 'keypoints' not in ann:
            continue
        kpts_17 = ann['keypoints']  # flat list: [x0,y0,v0, x1,y1,v1, ..., x16,y16,v16]
        kpts_13 = []
        n_visible = 0
        for idx in KEEP_KPT_INDICES:
            x = kpts_17[idx * 3]
            y = kpts_17[idx * 3 + 1]
            v = kpts_17[idx * 3 + 2]
            kpts_13.extend([x, y, v])
            if v > 0:
                n_visible += 1
        ann['keypoints'] = kpts_13
        ann['num_keypoints'] = n_visible

    with open(filtered_path, 'w') as f:
        json.dump(gt_filtered, f)

    n_anns = len(gt_filtered['annotations'])
    print(f"  Saved {filtered_path} ({n_anns} annotations, 13 keypoints each)")
    return filtered_path


def letterbox(img, target=640):
    """Resize keeping aspect ratio + pad. Returns img, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (target - nw) // 2
    pad_y = (target - nh) // 2
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def run_one(runner, context_type, gt_json, img_dir, out_jsonl, batch_size=8):
    """Run inference and STREAM COCO-format predictions to a JSONL file."""
    coco = COCO(gt_json)
    img_ids = coco.getImgIds()
    img_ids = img_ids[:500]
    img_metas = {i: coco.loadImgs(i)[0] for i in img_ids}

    pp = PoseEstPostProcessing(
        max_detections=20,          # was 100 — match COCO submission norm
        score_threshold=0.05,       # was 0.01 — drop low-confidence noise
        nms_iou_thresh=0.6,         # was 0.7  — slightly more aggressive NMS
        regression_length=15,
        strides=[8, 16, 32],
    )

    n_preds = 0
    with open(out_jsonl, "w") as fout, runner.infer_context(context_type) as ctx:
        for batch_start in tqdm(range(0, len(img_ids), batch_size),
                                desc=f"Inference ({context_type.name})"):
            batch_ids = img_ids[batch_start:batch_start + batch_size]
            batch_imgs, batch_meta = [], []

            for iid in batch_ids:
                meta = img_metas[iid]
                path = os.path.join(img_dir, meta['file_name'])
                img = cv2.imread(path)
                if img is None:
                    continue
                rgb = img[:, :, ::-1]
                lb, scale, (pad_x, pad_y) = letterbox(rgb, 640)
                batch_imgs.append(lb)
                batch_meta.append((iid, meta['width'], meta['height'], scale, pad_x, pad_y))

            if not batch_imgs:
                continue

            batch_arr = np.stack(batch_imgs).astype(np.float32)
            raw = runner.infer(ctx, batch_arr)
            # raw is a list of 9 tensors, each shape (batch, H, W, C)

            for i, (iid, ow, oh, scale, pad_x, pad_y) in enumerate(batch_meta):
                # Build per-sample dict; post-processor resolves outputs by shape, not key name
                raw_single = {f"out_{j}": tensor[i:i+1] for j, tensor in enumerate(raw)}

                results = pp.post_process(raw_single, height=640, width=640, class_num=1)

                bboxes       = results['bboxes'][0]
                keypoints    = results['keypoints'][0]
                joint_scores = results['joint_scores'][0]
                scores       = results['scores'][0, :, 0]

                valid = scores > 0.05
                if not valid.any():
                    continue

                bboxes       = bboxes[valid]
                keypoints    = keypoints[valid]
                joint_scores = joint_scores[valid]
                scores       = scores[valid]

                # Hard cap to top-20 by score per image (COCO submission norm)
                if len(scores) > 20:
                    top = np.argsort(scores)[-20:][::-1]
                    bboxes       = bboxes[top]
                    keypoints    = keypoints[top]
                    joint_scores = joint_scores[top]
                    scores       = scores[top]

                # Map from 640x640 letterbox space back to original image coords
                for det_idx in range(len(scores)):
                    bx, by, bw, bh = bboxes[det_idx]
                    x = (bx - pad_x) / scale
                    y = (by - pad_y) / scale
                    w = bw / scale
                    h = bh / scale

                    kpts_orig = []
                    for k in range(13):
                        kx, ky = keypoints[det_idx, k]
                        kx = (kx - pad_x) / scale
                        ky = (ky - pad_y) / scale
                        v = float(joint_scores[det_idx, k, 0])
                        kpts_orig.extend([float(kx), float(ky), v])

                    fout.write(json.dumps({
                        "image_id": int(iid),
                        "category_id": 1,
                        "keypoints": kpts_orig,
                        "score": float(scores[det_idx]),
                        "bbox": [float(x), float(y), float(w), float(h)],
                    }) + "\n")
                    n_preds += 1

    print(f"  wrote {n_preds} predictions to {out_jsonl}")
    return out_jsonl


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate(gt_json, predictions, label):
    """Run COCO bbox + keypoint mAP. Returns dict of metrics."""
    coco_gt = COCO(gt_json)

    if not predictions:
        print(f"[{label}] no predictions, skipping")
        return None

    coco_dt = coco_gt.loadRes(predictions)

    print(f"\n=== {label}: BBOX ===")
    bbox_eval = COCOeval(coco_gt, coco_dt, "bbox")
    bbox_eval.evaluate(); bbox_eval.accumulate(); bbox_eval.summarize()

    print(f"\n=== {label}: KEYPOINTS ===")
    kpt_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    # 13-keypoint OKS sigmas (first 13 of standard COCO 17)
    coco_default_sigmas = np.array([.26,.25,.25,.35,.35,.79,.79,.72,.72,.62,.62,1.07,1.07,
                                    .87,.87,.89,.89]) / 10.0
    kpt_eval.params.kpt_oks_sigmas = coco_default_sigmas[:13]
    kpt_eval.evaluate(); kpt_eval.accumulate(); kpt_eval.summarize()

    return {
        "bbox_mAP_50_95": bbox_eval.stats[0],
        "bbox_mAP_50":    bbox_eval.stats[1],
        "kpt_mAP_50_95":  kpt_eval.stats[0],
        "kpt_mAP_50":     kpt_eval.stats[1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har",   required=True, help="Path to *_optimized.har")
    ap.add_argument("--gt",    required=True, help="Path to 13-keypoint val2017 GT JSON")
    ap.add_argument("--imgs",  required=True, help="Path to val2017 image directory")
    ap.add_argument("--out",   required=True, help="Output summary JSON file")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reuse-fp32", action="store_true",
                    help="If FP32 jsonl already exists, skip FP32 inference and reuse it")
    ap.add_argument("--reuse-int8", action="store_true",
                    help="If INT8 jsonl already exists, skip INT8 inference and reuse it")
    args = ap.parse_args()

    out_stem = args.out.rsplit(".", 1)[0]
    fp32_jsonl = f"{out_stem}_fp32_preds.jsonl"
    int8_jsonl = f"{out_stem}_int8_preds.jsonl"

    # Filter GT from 17 to 13 keypoints (cached on disk after first run)
    gt_13 = get_filtered_gt(args.gt)

    # ── Phase 1: Inference (Hailo SDK in memory) ──
    runner = ClientRunner(har=args.har)

    if args.reuse_fp32 and os.path.exists(fp32_jsonl):
        print(f"\n>>> FP32 — reusing existing {fp32_jsonl}")
    else:
        print("\n>>> FP32 (SDK_NATIVE)")
        run_one(runner, InferenceContext.SDK_NATIVE,
                gt_13, args.imgs, fp32_jsonl, batch_size=args.batch)

    if args.reuse_int8 and os.path.exists(int8_jsonl):
        print(f"\n>>> INT8 — reusing existing {int8_jsonl}")
    else:
        print("\n>>> INT8 (SDK_QUANTIZED)")
        run_one(runner, InferenceContext.SDK_QUANTIZED,
                gt_13, args.imgs, int8_jsonl, batch_size=args.batch)

    # ── Free Hailo SDK memory before evaluation ──
    del runner
    gc.collect()
    print("\n>>> Runner released, starting evaluation...")

    # ── Phase 2: Evaluation (pycocotools, no Hailo SDK) ──
    print("\n>>> Evaluating FP32...")
    fp32_preds = load_jsonl(fp32_jsonl)
    fp32_metrics = evaluate(gt_13, fp32_preds, "FP32")
    del fp32_preds; gc.collect()

    print("\n>>> Evaluating INT8...")
    int8_preds = load_jsonl(int8_jsonl)
    int8_metrics = evaluate(gt_13, int8_preds, "INT8")
    del int8_preds; gc.collect()

    summary = {
        "har":      args.har,
        "n_images": len(COCO(gt_13).getImgIds()),
        "fp32":     fp32_metrics,
        "int8":     int8_metrics,
    }
    if fp32_metrics and int8_metrics:
        summary["delta"] = {k: int8_metrics[k] - fp32_metrics[k] for k in fp32_metrics}

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
