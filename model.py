"""Single-file inference entrypoint for the original LTX-2 one-stage pipeline.

This script is intentionally self-contained at the repository level:
- it adds local package sources to ``sys.path`` so it can run without pip-installing
  ``ltx-core`` / ``ltx-pipelines`` as separate modules,
- it defaults to CPU when CUDA is unavailable,
- it keeps FP8 quantization opt-in and CUDA-only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_SRCS = [
    REPO_ROOT / "packages" / "ltx-core" / "src",
    REPO_ROOT / "packages" / "ltx-pipelines" / "src",
]
for src in LOCAL_SRCS:
    src_str = str(src)
    if src.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.types import Audio
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.utils.media_io import encode_video


class LTX2Model:
    """Unified wrapper for the original LTX-2 one-stage pipeline."""

    def __init__(
        self,
        checkpoint: str,
        gemma_root: str,
        *,
        use_fp8_cast: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        quantization = None
        if use_fp8_cast and self.device.type == "cuda":
            from ltx_core.quantization.policy import QuantizationPolicy

            quantization = QuantizationPolicy.fp8_cast()
        elif use_fp8_cast:
            logging.warning("FP8 cast requested but CUDA is unavailable; running without FP8 quantization.")

        self.pipeline = TI2VidOneStagePipeline(
            checkpoint_path=checkpoint,
            gemma_root=gemma_root,
            loras=[],
            device=self.device,
            quantization=quantization,
        )

    @torch.inference_mode()
    def infer(
        self,
        prompt: str,
        output_path: str,
        *,
        negative_prompt: str = "",
        seed: int = 42,
        height: int = 512,
        width: int = 704,
        num_frames: int = 97,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 3.0,
        enhance_prompt: bool = False,
    ) -> tuple[object, Audio]:
        video, audio = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=MultiModalGuiderParams(cfg_scale=video_cfg_scale),
            audio_guider_params=MultiModalGuiderParams(cfg_scale=audio_cfg_scale),
            images=[],
            enhance_prompt=enhance_prompt,
        )

        encode_video(video=video, fps=frame_rate, audio=audio, output_path=output_path, video_chunks_number=1)
        return video, audio


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Original LTX-2 one-stage inference")
    parser.add_argument("--checkpoint", required=True, help="Path to ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--gemma-root", required=True, help="Path to Gemma-3 model directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt text")
    parser.add_argument("--output", default="output.mp4", help="Output mp4 path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--video-cfg-scale", type=float, default=3.0)
    parser.add_argument("--audio-cfg-scale", type=float, default=3.0)
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--fp8-cast", action="store_true", help="Enable fp8-cast quantization (CUDA only)")
    return parser


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = _parser().parse_args()

    model = LTX2Model(
        checkpoint=args.checkpoint,
        gemma_root=args.gemma_root,
        use_fp8_cast=args.fp8_cast,
    )
    model.infer(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=args.output,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_cfg_scale=args.video_cfg_scale,
        audio_cfg_scale=args.audio_cfg_scale,
        enhance_prompt=args.enhance_prompt,
    )


if __name__ == "__main__":
    main()
