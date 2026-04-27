import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
PENET_DIR = THIS_DIR.parent / "PENet"

# Add DMD3C extensions build directory to path
EXTS_BUILD_DIR = THIS_DIR / "exts" / "build"
for lib_dir in EXTS_BUILD_DIR.glob("lib.*"):
    if lib_dir.is_dir():
        sys.path.insert(0, str(lib_dir))

if str(PENET_DIR) not in sys.path:
    sys.path.insert(0, str(PENET_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataloaders import calibration_kitti  # noqa: E402
from dataloaders.my_loader import depth2pointsrgbp, load_depth_input  # noqa: E402
from models.BPNet import Pre_MF_Post  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PENet-compatible velodyne_depth points using DMD3C"
    )
    parser.add_argument(
        "--detpath",
        type=str,
        required=True,
        help="KITTI root that contains image_2, velodyne, calib",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        type=str,
        required=True,
        help="DMD3C checkpoint path (.pth/.pt) or folder containing result_ema.pth",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="velodyne_depth_dmd3c",
        help="Output folder name under detpath. Use velodyne_depth for in-place replacement.",
    )
    parser.add_argument(
        "--on-exist",
        type=str,
        choices=["skip", "overwrite", "backup", "error"],
        default="skip",
        help="How to handle existing output files",
    )
    parser.add_argument(
        "--backup-subdir",
        type=str,
        default="velodyne_depth_backup",
        help="Backup folder name under detpath when on-exist=backup",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Run on CPU",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=352,
        help="Network input/output height",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1216,
        help="Network input/output width",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[90.9950, 96.2278, 94.3213],
        help="RGB normalization mean",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[79.2382, 80.5267, 82.1483],
        help="RGB normalization std",
    )
    return parser.parse_args()


def list_frame_ids(detpath):
    velodyne_dir = Path(detpath) / "velodyne"
    if not velodyne_dir.exists():
        raise FileNotFoundError(f"Missing velodyne directory: {velodyne_dir}")
    frame_ids = sorted([p.stem for p in velodyne_dir.glob("*.bin")])
    if not frame_ids:
        raise RuntimeError(f"No .bin files found in {velodyne_dir}")
    return frame_ids


def get_fov_flag(pts_rect, img_shape, calib):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag


def resolve_checkpoint(path_str):
    ckpt_path = Path(path_str)
    if ckpt_path.is_dir():
        candidate = ckpt_path / "result_ema.pth"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Checkpoint folder has no result_ema.pth: {ckpt_path}")
    if ckpt_path.exists():
        return ckpt_path
    candidate = Path("checkpoints") / ckpt_path / "result_ema.pth"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {path_str}")


def load_model(checkpoint_path, device):
    model = Pre_MF_Post()
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and "net" in ckpt:
        state_dict = ckpt["net"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}")

    clean_state = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        clean_state[new_key] = value

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def prepare_output_paths(detpath, output_subdir, on_exist, backup_subdir):
    output_dir = Path(detpath) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = None
    if on_exist == "backup":
        backup_dir = Path(detpath) / backup_subdir
        backup_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, backup_dir


def handle_existing(path, policy, backup_dir=None):
    if not path.exists():
        return "write"
    if policy == "skip":
        return "skip"
    if policy == "overwrite":
        return "write"
    if policy == "error":
        raise FileExistsError(f"Output exists: {path}")
    if policy == "backup":
        assert backup_dir is not None
        backup_target = backup_dir / path.name
        if backup_target.exists():
            stamp = int(time.time())
            backup_target = backup_dir / f"{path.stem}.{stamp}.npy"
        shutil.move(str(path), str(backup_target))
        return "write"
    raise ValueError(f"Unsupported on-exist policy: {policy}")


def run(args):
    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    print(f"[info] device: {device}")

    detpath = Path(args.detpath)
    frame_ids = list_frame_ids(detpath)
    output_dir, backup_dir = prepare_output_paths(
        detpath, args.output_subdir, args.on_exist, args.backup_subdir
    )

    ckpt_path = resolve_checkpoint(args.evaluate)
    print(f"[info] checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device)

    image_mean = np.array(args.mean, dtype=np.float32)
    image_std = np.array(args.std, dtype=np.float32)

    processed = 0
    skipped = 0

    with torch.no_grad():
        for idx, frame_id in enumerate(frame_ids):
            output_path = output_dir / f"{frame_id}.npy"
            action = handle_existing(output_path, args.on_exist, backup_dir)
            if action == "skip":
                skipped += 1
                continue

            image_path = detpath / "image_2" / f"{frame_id}.png"
            velo_path = detpath / "velodyne" / f"{frame_id}.bin"
            calib_path = detpath / "calib" / f"{frame_id}.txt"

            if not (image_path.exists() and velo_path.exists() and calib_path.exists()):
                print(f"[warn] missing files for frame {frame_id}, skipping")
                skipped += 1
                continue

            calib = calibration_kitti.Calibration(str(calib_path))
            lidar = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)
            image = np.array(Image.open(str(image_path)).convert("RGB"), dtype=np.int32)
            image = image[: args.image_height, : args.image_width]

            pts_rect = calib.lidar_to_rect(lidar[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, image.shape, calib)
            lidar_fov = lidar[fov_flag]

            rgb, sparse = load_depth_input(calib, image, lidar)
            rgb = rgb[: args.image_height, : args.image_width].astype(np.float32)
            sparse = sparse[: args.image_height, : args.image_width].astype(np.float32)

            rgb = (rgb - image_mean) / image_std

            image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)[None]).to(device)
            sparse_tensor = torch.from_numpy(sparse.transpose(2, 0, 1)[None]).to(device)
            k_cam = torch.from_numpy(calib.P2[:3, :3].astype(np.float32))[None].to(device)

            output = model(image_tensor, None, sparse_tensor, k_cam)
            if isinstance(output, (tuple, list)):
                output = output[-1]

            depth = output.squeeze().detach().cpu().numpy().reshape(
                args.image_height, args.image_width, 1
            )
            final_points = depth2pointsrgbp(depth, image, calib, lidar_fov).astype(np.float16)
            np.save(str(output_path), final_points)

            processed += 1
            if (idx + 1) % 100 == 0:
                print(f"[info] processed {idx + 1}/{len(frame_ids)}")

    print(
        f"[done] total={len(frame_ids)} processed={processed} skipped={skipped} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    run(parse_args())