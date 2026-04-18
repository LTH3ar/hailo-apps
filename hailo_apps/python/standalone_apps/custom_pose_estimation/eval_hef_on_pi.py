#!/usr/bin/env python3
"""
Evaluate INT8 mAP for a compiled HEF on the Hailo-8 device (Raspberry Pi 5).
Runs inference via HailoRT, applies the custom 13-keypoint post-processor,
and computes COCO keypoint + bbox mAP using pycocotools.

Usage:
    python eval_hef_on_pi.py \
        --hef /path/to/best.hef \
        --gt  /path/to/person_keypoints_val2017.json \
        --imgs /path/to/val2017/ \
        --out  hef_eval_n.json

Requires: hailo_platform (HailoRT), pycocotools, opencv-python, tqdm, numpy
Place this script next to custom_pose_estimation_utils.py.
"""
import argparse, json, os, gc
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

from custom_pose_estimation_utils import PoseEstPostProcessing

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams,
    FormatType
)

# ── 13-keypoint GT filtering (same as training-machine script) ──

KEEP_KPT_INDICES = list(range(13))

UPPER_BODY_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip"
]

UPPER_BODY_SKELETON = [
    [1, 2], [2, 4], [1, 3], [3, 5],
    [6, 7], [6, 8], [8, 10], [7, 9], [9, 11],
    [6, 12], [7, 13], [12, 13]
]


def get_filtered_gt(original_gt_json):
    """Create a 13-keypoint GT JSON if it doesn't already exist."""
    filtered_path = original_gt_json.replace(".json", "_13kpt.json")
    if os.path.exists(filtered_path):
        print(f"  Using cached 13-keypoint GT: {filtered_path}")
        return filtered_path

    print(f"  Creating 13-keypoint GT from {original_gt_json}...")
    import copy
    with open(original_gt_json) as f:
        gt = json.load(f)
    gt_filtered = copy.deepcopy(gt)
    for cat in gt_filtered['categories']:
        if cat['name'] == 'person':
            cat['keypoints'] = UPPER_BODY_KPT_NAMES
            cat['skeleton'] = UPPER_BODY_SKELETON
    for ann in gt_filtered['annotations']:
        if 'keypoints' not in ann:
            continue
        kpts_17 = ann['keypoints']
        kpts_13 = []
        n_vis = 0
        for idx in KEEP_KPT_INDICES:
            x, y, v = kpts_17[idx*3], kpts_17[idx*3+1], kpts_17[idx*3+2]
            kpts_13.extend([x, y, v])
            if v > 0:
                n_vis += 1
        ann['keypoints'] = kpts_13
        ann['num_keypoints'] = n_vis
    with open(filtered_path, 'w') as f:
        json.dump(gt_filtered, f)
    print(f"  Saved {filtered_path} ({len(gt_filtered['annotations'])} annotations)")
    return filtered_path


# ── Preprocessing ──

def letterbox(img, target=640):
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (target - nw) // 2
    pad_y = (target - nh) // 2
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


# ── Inference ──

def run_hef_eval(hef_path, gt_json, img_dir, out_jsonl):
    """Run HEF inference on Hailo-8 and stream predictions to JSONL."""
    coco = COCO(gt_json)
    img_ids = coco.getImgIds()
    img_metas = {i: coco.loadImgs(i)[0] for i in img_ids}

    pp = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32],
    )

    hef = HEF(hef_path)
    devices = VDevice()
    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )
    network_group = devices.configure(hef, configure_params)[0]

    input_vstream_info = hef.get_input_vstream_infos()
    output_vstream_info = hef.get_output_vstream_infos()

    input_params = InputVStreamParams.make(
        network_group, format_type=FormatType.FLOAT32
    )
    output_params = OutputVStreamParams.make(
        network_group, format_type=FormatType.FLOAT32
    )

    input_name = input_vstream_info[0].name

    n_preds = 0
    with open(out_jsonl, "w") as fout:
        with network_group.activate():
            with InferVStreams(network_group, input_params, output_params) as pipeline:
                for iid in tqdm(img_ids, desc="HEF inference"):
                    meta = img_metas[iid]
                    path = os.path.join(img_dir, meta['file_name'])
                    img = cv2.imread(path)
                    if img is None:
                        continue

                    rgb = img[:, :, ::-1]
                    lb, scale, (pad_x, pad_y) = letterbox(rgb, 640)

                    # HailoRT expects (batch, H, W, C) uint8 or float32
                    input_data = {input_name: np.expand_dims(lb.astype(np.float32), axis=0)}
                    raw_results = pipeline.infer(input_data)

                    # raw_results is a dict: {output_name: ndarray(1, H, W, C)}
                    raw_single = {k: v for k, v in raw_results.items()}

                    results = pp.post_process(raw_single, height=640, width=640, class_num=1)

                    bboxes       = results['bboxes'][0]
                    keypoints    = results['keypoints'][0]
                    joint_scores = results['joint_scores'][0]
                    scores       = results['scores'][0, :, 0]

                    valid = scores > 0.001
                    if not valid.any():
                        continue

                    bboxes       = bboxes[valid]
                    keypoints    = keypoints[valid]
                    joint_scores = joint_scores[valid]
                    scores       = scores[valid]

                    ow, oh = meta['width'], meta['height']
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

    devices.release()
    print(f"  Wrote {n_preds} predictions to {out_jsonl}")
    return out_jsonl


# ── Evaluation ──

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate(gt_json, predictions, label):
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
    coco_sigmas = np.array([.26,.25,.25,.35,.35,.79,.79,.72,.72,.62,.62,1.07,1.07,
                            .87,.87,.89,.89]) / 10.0
    kpt_eval.params.kpt_oks_sigmas = coco_sigmas[:13]
    kpt_eval.evaluate(); kpt_eval.accumulate(); kpt_eval.summarize()

    return {
        "bbox_mAP_50_95": bbox_eval.stats[0],
        "bbox_mAP_50":    bbox_eval.stats[1],
        "kpt_mAP_50_95":  kpt_eval.stats[0],
        "kpt_mAP_50":     kpt_eval.stats[1],
    }


# ── Main ──

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate HEF on Hailo-8 (INT8 mAP on COCO val2017, 13 keypoints)"
    )
    ap.add_argument("--hef",  required=True, help="Path to compiled .hef file")
    ap.add_argument("--gt",   required=True, help="Path to COCO person_keypoints_val2017.json")
    ap.add_argument("--imgs", required=True, help="Path to val2017 image directory")
    ap.add_argument("--out",  required=True, help="Output summary JSON file")
    ap.add_argument("--reuse", action="store_true",
                    help="If prediction JSONL already exists, skip inference and reuse it")
    args = ap.parse_args()

    out_stem = args.out.rsplit(".", 1)[0]
    pred_jsonl = f"{out_stem}_preds.jsonl"

    gt_13 = get_filtered_gt(args.gt)

    if args.reuse and os.path.exists(pred_jsonl):
        print(f"\n>>> Reusing existing predictions: {pred_jsonl}")
    else:
        print(f"\n>>> Running HEF inference on Hailo-8: {args.hef}")
        run_hef_eval(args.hef, gt_13, args.imgs, pred_jsonl)

    gc.collect()

    print("\n>>> Evaluating INT8 predictions...")
    preds = load_jsonl(pred_jsonl)
    metrics = evaluate(gt_13, preds, "INT8 (HEF on Hailo-8)")
    del preds; gc.collect()

    summary = {
        "hef":      args.hef,
        "n_images": len(COCO(gt_13).getImgIds()),
        "int8":     metrics,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
