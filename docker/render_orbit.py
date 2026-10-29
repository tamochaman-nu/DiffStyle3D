"""Render a video that orbits the subject of a trained 3DGS / DiffStyle3D model.

The orbit follows the training cameras: it circles a target point at their
mean distance and height, sweeping their azimuth range (times --span_scale).
The target is the median of the opaque Gaussians (--center gaussians) or the
point the cameras look at, i.e. the least-squares intersection of their optical
axes (--center cameras), which tends to lie on the front surface rather than
inside the head. With --pingpong the sweep goes there and back, so the video
loops seamlessly. Intrinsics and image size are taken from the first training
camera. --max_scale hides oversized Gaussians (streaks that only show up from
novel views) in the video; the model itself is not changed.

Reads <model_path>/cameras.json and <model_path>/point_cloud/iteration_<N>/point_cloud.ply
(both written by train-3dgs and train-style) and writes

    <model_path>/orbit/iteration_<N>/frames/00000.png ...
    <model_path>/orbit/iteration_<N>/orbit.mp4
"""
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from types import SimpleNamespace

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaussian_renderer import render  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2  # noqa: E402

ZNEAR, ZFAR = 0.01, 100.0
PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)


def look_at_point(centers, forwards):
    """Point closest (least squares) to every camera's optical axis."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, forwards):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def orbit_poses(cams, target, n_frames, span_scale, pingpong):
    """Camera-to-world rotations (columns: right, down, forward) and centers."""
    centers = np.array([c["position"] for c in cams])
    rots = np.array([c["rotation"] for c in cams])  # camera-to-world, OpenCV axes
    up = -rots[:, :, 1].mean(0)
    up /= np.linalg.norm(up)

    rel = centers - target
    height = float((rel @ up).mean())
    flat = rel - np.outer(rel @ up, up)
    radius = float(np.linalg.norm(flat, axis=1).mean())
    a = flat.mean(0)
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    az = np.arctan2(flat @ b, flat @ a)
    mid, half = (az.min() + az.max()) / 2, (az.max() - az.min()) / 2 * span_scale

    thetas = np.linspace(mid - half, mid + half, n_frames)
    if pingpong:
        thetas = np.concatenate([thetas, thetas[-2:0:-1]])
    poses = []
    for th in thetas:
        eye = target + height * up + radius * (np.cos(th) * a + np.sin(th) * b)
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        poses.append((np.stack([right, down, fwd], axis=1), eye))
    print(f"[render-orbit] target {np.round(target, 3).tolist()}, radius {radius:.3f}, "
          f"sweep {math.degrees(2 * half):.1f} deg, {len(poses)} frames")
    return poses


def minicam(c2w_rot, eye, fx, fy, width, height):
    w2c_rot = c2w_rot.T
    t = -w2c_rot @ eye
    fovx, fovy = focal2fov(fx, width), focal2fov(fy, height)
    world_view = torch.tensor(getWorld2View2(c2w_rot, t)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(ZNEAR, ZFAR, fovx, fovy).transpose(0, 1).cuda()
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, ZNEAR, ZFAR, world_view, full)


def latest_iteration(model_path):
    dirs = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*"))
    if not dirs:
        sys.exit(f"[render-orbit] No point_cloud/iteration_* in {model_path}")
    return max(int(re.search(r"iteration_(\d+)$", d).group(1)) for d in dirs)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1, help="-1: latest point_cloud/iteration_*")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--n_frames", type=int, default=120, help="frames for one sweep")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--span_scale", type=float, default=1.0,
                        help="sweep range as a multiple of the training cameras' azimuth range")
    parser.add_argument("--pingpong", action=BooleanOptionalAction, default=True,
                        help="sweep there and back so the video loops")
    parser.add_argument("--center", choices=["gaussians", "cameras"], default="gaussians",
                        help="orbit target: median of the opaque Gaussians, or where the training cameras look")
    parser.add_argument("--max_scale", type=float, default=0.0,
                        help="hide Gaussians whose largest scale exceeds this (0: keep all)")
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    iteration = args.iteration if args.iteration >= 0 else latest_iteration(args.model_path)
    ply = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    with open(os.path.join(args.model_path, "cameras.json")) as f:
        cams = json.load(f)

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(ply)
    print(f"[render-orbit] {ply}: {gaussians.get_xyz.shape[0]} Gaussians")
    if args.max_scale > 0:
        keep = gaussians.get_scaling.max(dim=1).values <= args.max_scale
        # Rendering only reads these tensors, so masking them is enough.
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            setattr(gaussians, name, getattr(gaussians, name)[keep])
        print(f"[render-orbit] --max_scale {args.max_scale}: hid {int((~keep).sum())} Gaussians")

    if args.center == "gaussians":
        xyz = gaussians.get_xyz.detach()
        target = xyz[gaussians.get_opacity[:, 0] > 0.5].median(dim=0).values.cpu().numpy()
    else:
        target = look_at_point(np.array([c["position"] for c in cams]),
                               np.array([c["rotation"] for c in cams])[:, :, 2])
    bg = torch.tensor([1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0], device="cuda")

    out_dir = os.path.join(args.model_path, "orbit", f"iteration_{iteration}")
    frames_dir = os.path.join(out_dir, "frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)

    ref = cams[0]
    with torch.no_grad():
        for i, (rot, eye) in enumerate(orbit_poses(cams, target, args.n_frames, args.span_scale, args.pingpong)):
            cam = minicam(rot, eye, ref["fx"], ref["fy"], ref["width"], ref["height"])
            image = render(cam, gaussians, PIPE, bg)["render"]
            torchvision.utils.save_image(image, os.path.join(frames_dir, f"{i:05d}.png"))

    video = os.path.join(out_dir, "orbit.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
        "-i", os.path.join(frames_dir, "%05d.png"),
        # libx264 + yuv420p needs even sizes; this is what most players accept.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", video,
    ], check=True)
    print(f"[render-orbit] Wrote {video}")


if __name__ == "__main__":
    main()
