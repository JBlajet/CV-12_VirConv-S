import argparse
import glob
import os
import pickle
import re
import subprocess
import sys
import struct
from pathlib import Path

import numpy as np

from pcdet.models.model_utils import model_nms_utils
from pcdet.utils import box_utils, calibration_kitti


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate pseudo labels by fusing predictions from multiple checkpoints.'
    )

    parser.add_argument(
        '--cfg_file',
        type=str,
        default='tools/cfgs/models/kitti/VirConv-T_c3.yaml',
        help='Teacher config file used for inference.'
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        required=True,
        help='Directory containing teacher checkpoints (checkpoint_epoch_*.pth).'
    )
    parser.add_argument(
        '--selection_mode',
        choices=['paper_last_k', 'around_best', 'explicit'],
        default='paper_last_k',
        help='Checkpoint selection policy.'
    )
    parser.add_argument('--k', type=int, default=10, help='Number of checkpoints for paper_last_k.')
    parser.add_argument('--best_epoch', type=int, default=None, help='Best epoch for around_best mode.')
    parser.add_argument('--window', type=int, default=2, help='Half window for around_best mode.')
    parser.add_argument(
        '--ckpt_list',
        type=str,
        nargs='*',
        default=None,
        help='Explicit checkpoint paths for explicit mode.'
    )

    parser.add_argument('--score_thresh', type=float, default=0.9, help='Final pseudo-label score threshold.')
    parser.add_argument('--wbf_iou', type=float, default=0.85, help='IoU threshold for WBF clustering.')
    parser.add_argument('--retain_low', action='store_true', help='Enable WBF retain_low mode.')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extra_tag', type=str, default='pseudo_gen_c3')

    parser.add_argument('--data_root', type=str, default='data/kitti')
    parser.add_argument('--semi_info_name', type=str, default='kitti_infos_semi.pkl')
    parser.add_argument('--semi_split_name', type=str, default='semi.txt')
    parser.add_argument('--semi_label_dir', type=str, default='data/kitti/semi/label_2')

    parser.add_argument(
        '--teacher_split',
        type=str,
        default='semi',
        help='DATA_CONFIG.DATA_SPLIT.test override used for pseudo generation.'
    )
    parser.add_argument('--dry_run', action='store_true', help='Only print actions, do not execute inference/fusion.')

    return parser.parse_args()


def _extract_epoch(ckpt_path: Path):
    match = re.search(r'checkpoint_epoch_(\d+)\.pth$', ckpt_path.name)
    return int(match.group(1)) if match else None


def _read_png_shape(image_path: Path):
    with open(image_path, 'rb') as f:
        signature = f.read(8)
        if signature != b'\x89PNG\r\n\x1a\n':
            raise ValueError(f'Unsupported image format (expected PNG): {image_path}')
        length = struct.unpack('>I', f.read(4))[0]
        chunk_type = f.read(4)
        if chunk_type != b'IHDR':
            raise ValueError(f'Invalid PNG header: {image_path}')
        width, height = struct.unpack('>II', f.read(8))
        return np.array([height, width], dtype=np.int32)


def list_checkpoints(ckpt_dir: Path):
    candidates = []
    for p in ckpt_dir.glob('checkpoint_epoch_*.pth'):
        if 'optim' in p.name:
            continue
        epoch = _extract_epoch(p)
        if epoch is not None:
            candidates.append((epoch, p))
    candidates.sort(key=lambda x: x[0])
    return candidates


def select_checkpoints(args, all_ckpts):
    if args.selection_mode == 'paper_last_k':
        selected = all_ckpts[-args.k:]
    elif args.selection_mode == 'around_best':
        if args.best_epoch is None:
            raise ValueError('--best_epoch is required when selection_mode=around_best')
        lo = args.best_epoch - args.window
        hi = args.best_epoch + args.window
        selected = [x for x in all_ckpts if lo <= x[0] <= hi]
    else:
        if not args.ckpt_list:
            raise ValueError('--ckpt_list is required when selection_mode=explicit')
        selected = []
        for ckpt_str in args.ckpt_list:
            p = Path(ckpt_str)
            epoch = _extract_epoch(p)
            if epoch is None:
                raise ValueError(f'Invalid checkpoint name format: {p}')
            selected.append((epoch, p))
        selected.sort(key=lambda x: x[0])

    if len(selected) == 0:
        raise RuntimeError('No checkpoints selected. Check ckpt_dir and selection args.')

    return selected


def ensure_semi_info_file(repo_root: Path, data_root: Path, split_name: str, info_name: str):
    info_path = data_root / info_name
    if info_path.exists():
        return info_path

    split_path = data_root / 'ImageSets' / split_name
    if not split_path.exists():
        raise FileNotFoundError(f'Missing split file: {split_path}')

    semi_root = data_root / 'semi'
    image_root = semi_root / 'image_2'
    calib_root = semi_root / 'calib'

    sample_ids = [x.strip() for x in split_path.read_text().splitlines() if x.strip()]
    infos = []
    for sid in sample_ids:
        img_file = image_root / f'{sid}.png'
        calib_file = calib_root / f'{sid}.txt'
        if not img_file.exists() or not calib_file.exists():
            continue

        image_shape = _read_png_shape(img_file)
        calib = calibration_kitti.Calibration(calib_file)

        p2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
        r0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
        r0_4x4[3, 3] = 1.
        r0_4x4[:3, :3] = calib.R0
        v2c_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)

        infos.append({
            'point_cloud': {'num_features': 4, 'lidar_idx': sid},
            'image': {'image_idx': sid, 'image_shape': image_shape},
            'calib': {'P2': p2, 'R0_rect': r0_4x4, 'Tr_velo_to_cam': v2c_4x4}
        })

    if len(infos) == 0:
        raise RuntimeError(f'No valid semi infos generated from {split_path}')

    with open(info_path, 'wb') as f:
        pickle.dump(infos, f)

    return info_path


def run_inference_for_ckpt(repo_root: Path, args, ckpt_path: Path, semi_info_name: str):
    # Make cfg_file relative to tools directory so base config paths resolve correctly
    cfg_file = str((repo_root / args.cfg_file).relative_to(repo_root / 'tools'))
    epoch = _extract_epoch(ckpt_path)
    eval_tag = f'pseudo_ckpt_{epoch}'

    cmd = [
        sys.executable,
        'test.py',
        '--cfg_file', cfg_file,
        '--ckpt', str(ckpt_path),
        '--batch_size', str(args.batch_size),
        '--workers', str(args.workers),
        '--extra_tag', args.extra_tag,
        '--eval_tag', eval_tag,
        '--save_to_file',
        '--set',
        'DATA_CONFIG.DATASET', 'KittiDatasetSemi',
        'DATA_CONFIG.DATA_SPLIT.test', args.teacher_split,
        'DATA_CONFIG.INFO_PATH.test', f"['{semi_info_name}']"
    ]

    print('Running:', ' '.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=str(repo_root / 'tools'), check=True)

    cfg_stem = Path(args.cfg_file).stem
    cfg_parts = args.cfg_file.replace('\\', '/').split('/')[1:-1]
    exp_group = '/'.join(cfg_parts)
    result_dir = repo_root / 'output' / exp_group / cfg_stem / args.extra_tag / 'eval' / f'epoch_{epoch}' / args.teacher_split / eval_tag

    result_candidates = sorted(result_dir.glob('result*.pkl'), key=os.path.getmtime)
    if len(result_candidates) == 0:
        raise FileNotFoundError(f'No result pkl found in {result_dir}')

    return result_candidates[-1]


def load_annos(result_pkl: Path):
    with open(result_pkl, 'rb') as f:
        return pickle.load(f)


def build_frame_prediction_pool(result_pkls, class_to_id):
    pooled = {}
    for pkl_path in result_pkls:
        annos = load_annos(pkl_path)
        for anno in annos:
            frame_id = str(anno['frame_id'])
            names = anno.get('name', np.array([]))
            scores = anno.get('score', np.array([]))
            boxes = anno.get('boxes_lidar', np.zeros((0, 7)))

            for i in range(len(scores)):
                cls_name = str(names[i])
                if cls_name not in class_to_id:
                    continue
                label_id = class_to_id[cls_name]
                pooled.setdefault(frame_id, {}).setdefault(label_id, []).append((label_id, float(scores[i]), boxes[i]))
    return pooled


def fuse_frame_predictions(pred_list, wbf_iou, retain_low, score_thresh):
    if len(pred_list) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.zeros((0, 7), dtype=np.float32)

    labels = np.array([x[0] for x in pred_list], dtype=np.int32)
    scores = np.array([x[1] for x in pred_list], dtype=np.float32)
    boxes = np.stack([x[2] for x in pred_list], axis=0).astype(np.float32)

    fused_labels, fused_scores, fused_boxes = model_nms_utils.compute_WBF(
        det_names=labels,
        det_scores=scores,
        det_boxes=boxes,
        iou_thresh=wbf_iou,
        retain_low=retain_low,
        score_thresh=score_thresh
    )

    fused_labels = np.array(fused_labels)
    fused_scores = np.array(fused_scores)
    fused_boxes = np.array(fused_boxes)

    keep = fused_scores >= score_thresh
    return fused_labels[keep], fused_scores[keep], fused_boxes[keep]


def write_kitti_label_file(label_path: Path, calib, image_shape, labels, scores, boxes_lidar, class_names):
    if boxes_lidar.shape[0] == 0:
        label_path.write_text('')
        return

    boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(boxes_lidar, calib)
    boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes_camera, calib, image_shape=image_shape)
    alphas = -np.arctan2(-boxes_lidar[:, 1], boxes_lidar[:, 0]) + boxes_camera[:, 6]

    order = np.argsort(-scores)

    with open(label_path, 'w') as f:
        for idx in order:
            cls_name = class_names[int(labels[idx]) - 1]
            bbox = boxes_img[idx]
            loc = boxes_camera[idx, 0:3]
            dims = boxes_camera[idx, 3:6]  # l, h, w
            ry = boxes_camera[idx, 6]
            score = scores[idx]

            print(
                '%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                % (
                    cls_name, alphas[idx],
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    dims[1], dims[2], dims[0],
                    loc[0], loc[1], loc[2],
                    ry, score
                ),
                file=f
            )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / args.data_root).resolve()
    semi_label_dir = (repo_root / args.semi_label_dir).resolve()
    semi_label_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir).resolve()
    all_ckpts = list_checkpoints(ckpt_dir)
    selected = select_checkpoints(args, all_ckpts)

    print('Selected checkpoints:')
    for epoch, ckpt in selected:
        print(f'  epoch={epoch} ckpt={ckpt}')

    semi_info_path = ensure_semi_info_file(
        repo_root=repo_root,
        data_root=data_root,
        split_name=args.semi_split_name,
        info_name=args.semi_info_name
    )
    print(f'Using semi info file: {semi_info_path}')

    if args.dry_run:
        print('Dry run finished before inference/fusion.')
        return

    result_pkls = []
    for _, ckpt in selected:
        result_pkl = run_inference_for_ckpt(
            repo_root=repo_root,
            args=args,
            ckpt_path=ckpt,
            semi_info_name=semi_info_path.name
        )
        result_pkls.append(result_pkl)
        print(f'Collected: {result_pkl}')

    class_names = ['Car', 'Pedestrian', 'Cyclist']
    class_to_id = {name: i + 1 for i, name in enumerate(class_names)}

    pooled = build_frame_prediction_pool(result_pkls, class_to_id)
    print(f'Frames with pooled detections: {len(pooled)}')

    semi_root = data_root / 'semi'
    calib_root = semi_root / 'calib'
    image_root = semi_root / 'image_2'

    written = 0
    for frame_id, class_pred_map in pooled.items():
        fused_labels_list = []
        fused_scores_list = []
        fused_boxes_list = []

        for label_id, pred_list in class_pred_map.items():
            labels, scores, boxes = fuse_frame_predictions(
                pred_list,
                wbf_iou=args.wbf_iou,
                retain_low=args.retain_low,
                score_thresh=args.score_thresh
            )
            if len(scores) == 0:
                continue
            fused_labels_list.append(labels)
            fused_scores_list.append(scores)
            fused_boxes_list.append(boxes)

        if len(fused_scores_list) == 0:
            continue

        labels = np.concatenate(fused_labels_list, axis=0)
        scores = np.concatenate(fused_scores_list, axis=0)
        boxes = np.concatenate(fused_boxes_list, axis=0)

        calib_file = calib_root / f'{frame_id}.txt'
        image_file = image_root / f'{frame_id}.png'
        if not calib_file.exists() or not image_file.exists():
            continue

        calib = calibration_kitti.Calibration(calib_file)
        image_shape = _read_png_shape(image_file)

        out_file = semi_label_dir / f'{frame_id}.txt'
        write_kitti_label_file(out_file, calib, image_shape, labels, scores, boxes, class_names)
        written += 1

    print(f'Pseudo labels written: {written} files -> {semi_label_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
