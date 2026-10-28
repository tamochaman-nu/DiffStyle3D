"""Make every view of an undistorted COLMAP scene the same resolution.

train_style.py stacks the views of a batch with torch.stack, so all training
images must share one size. COLMAP's image_undistorter usually produces a
slightly different size per camera, which makes train_style.py fail.

This writes a copy of the scene in which every registered image is resized to
WIDTH x HEIGHT and each camera's intrinsics are scaled to match
(fx, cx by WIDTH / w and fy, cy by HEIGHT / h). Poses and points are copied
unchanged, so the scene geometry stays the same.

The source scene is only read; the result goes to a new directory.
"""
import os
import shutil
import struct
import sys
from argparse import ArgumentParser

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary  # noqa: E402

PINHOLE_MODEL_ID = 1


def scaled_pinhole_params(cam, sx, sy):
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params
    elif cam.model == "SIMPLE_PINHOLE":
        fx, cx, cy = cam.params
        fy = fx
    else:
        sys.exit(f"[prepare-scene] Unsupported camera model {cam.model}: "
                 "only undistorted scenes (PINHOLE / SIMPLE_PINHOLE) are supported.")
    return fx * sx, fy * sy, cx * sx, cy * sy


def write_cameras_binary(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, (width, height, params) in sorted(cameras.items()):
            f.write(struct.pack("<iiQQ", cam_id, PINHOLE_MODEL_ID, width, height))
            f.write(struct.pack("<4d", *params))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source scene with images/ and sparse/0/")
    parser.add_argument("--dst", required=True, help="output scene directory (must not exist)")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--images", default="images", help="image folder name inside --src")
    args = parser.parse_args()

    src_sparse = os.path.join(args.src, "sparse", "0")
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        sys.exit("[prepare-scene] --dst must differ from --src.")
    if os.path.exists(args.dst):
        sys.exit(f"[prepare-scene] Output already exists: {args.dst} (remove it to regenerate).")

    extrinsics = read_extrinsics_binary(os.path.join(src_sparse, "images.bin"))
    intrinsics = read_intrinsics_binary(os.path.join(src_sparse, "cameras.bin"))

    dst_images = os.path.join(args.dst, "images")
    dst_sparse = os.path.join(args.dst, "sparse", "0")
    os.makedirs(dst_images)
    os.makedirs(dst_sparse)

    new_cameras = {}
    for i, image in enumerate(extrinsics.values(), 1):
        cam = intrinsics[image.camera_id]
        src_image = os.path.join(args.src, args.images, image.name)
        with Image.open(src_image) as img:
            if img.size != (cam.width, cam.height):
                sys.exit(f"[prepare-scene] {image.name} is {img.size}, but camera {cam.id} "
                         f"is {cam.width}x{cam.height}. Use the undistorted images.")
            resized = img.resize((args.width, args.height), Image.LANCZOS)
            dst_image = os.path.join(dst_images, image.name)
            os.makedirs(os.path.dirname(dst_image), exist_ok=True)
            resized.save(dst_image, quality=95)
        params = scaled_pinhole_params(cam, args.width / cam.width, args.height / cam.height)
        prev = new_cameras.setdefault(cam.id, (args.width, args.height, params))
        if prev[2] != params:
            sys.exit(f"[prepare-scene] Camera {cam.id} is shared by images of different sizes.")
        print(f"[prepare-scene] {i}/{len(extrinsics)} {image.name}: "
              f"{cam.width}x{cam.height} -> {args.width}x{args.height}")

    write_cameras_binary(new_cameras, os.path.join(dst_sparse, "cameras.bin"))
    for name in ("images.bin", "points3D.bin", "points3D.ply"):
        if os.path.exists(os.path.join(src_sparse, name)):
            shutil.copy2(os.path.join(src_sparse, name), os.path.join(dst_sparse, name))
    print(f"[prepare-scene] Wrote {args.dst}")


if __name__ == "__main__":
    main()
