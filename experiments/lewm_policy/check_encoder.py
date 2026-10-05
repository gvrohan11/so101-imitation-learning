"""Integration check: render a simulator frame and encode it with the LeWM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .env import BallCupVisionEnv
from .features import make_lewm_backbone


DEFAULT_LEWM_REPO = Path(
    "/Users/rohan/Documents/Python-Projects/machine-learning/Le-World-Model-Implementation"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-repo", type=Path, default=DEFAULT_LEWM_REPO)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Use a saved MuJoCo camera frame when this shell has no OpenGL context.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or args.lewm_repo / "lewm_seq_projectors.pt"

    if args.image is None:
        env = BallCupVisionEnv(stage=1, seed=7)
        try:
            observation, _ = env.reset(seed=7)
            image_array = observation["image"]
            camera_source = "MuJoCo front camera"
            state_shape = list(observation["proprio"].shape)
        finally:
            env.close()
    else:
        with Image.open(args.image) as source:
            rgb = source.convert("RGB")
            tensor = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float()[None]
        tensor = F.interpolate(
            tensor, size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0).round().clamp(0, 255).byte()
        image_array = tensor.numpy()
        camera_source = f"saved frame: {args.image.resolve()}"
        state_shape = None

    backbone, _ = make_lewm_backbone(args.lewm_repo, checkpoint)
    backbone = backbone.to(args.device).eval()
    image = torch.as_tensor(image_array, device=args.device).unsqueeze(0)
    image = image.float().div(255.0)
    with torch.inference_mode():
        embedding = backbone(image)

    values = embedding.detach().float().cpu().numpy()
    if values.shape != (1, 192) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid LeWM output: shape={values.shape}, finite={np.isfinite(values).all()}")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "input_shape_chw": list(image_array.shape),
                "embedding_shape": list(values.shape),
                "finite": bool(np.isfinite(values).all()),
                "embedding_mean": float(values.mean()),
                "embedding_std": float(values.std()),
                "robot_state_shape": state_shape,
                "camera_source": camera_source,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
