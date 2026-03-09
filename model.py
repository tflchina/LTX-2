"""Single-file high-performance inference entrypoint for LTX-2.

This wrapper intentionally prioritizes throughput over exact parity with all
pipelines/configurations. It uses the distilled two-stage pipeline, optional
FP8 quantization, and low-step defaults for faster generation.

Inference example:
    python model.py \
      --distilled-checkpoint /models/ltx-2.3-22b-distilled.safetensors \
      --spatial-upscaler /models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
      --gemma-root /models/gemma-3-12b-it-qat-q4_0-unquantized \
      --prompt "A cinematic drone shot over snowy mountains at sunrise" \
      --output out.mp4
"""

from __future__ import annotations

import argparse
import logging

import torch

from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video


class FastLTX2Model:
    """Unified model wrapper tuned for fast inference."""

    def __init__(
        self,
        distilled_checkpoint: str,
        spatial_upscaler: str,
        gemma_root: str,
        *,
        use_fp8_cast: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quantization = QuantizationPolicy.fp8_cast() if use_fp8_cast else None

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=distilled_checkpoint,
            spatial_upsampler_path=spatial_upscaler,
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
        seed: int = 42,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 24.0,
        enhance_prompt: bool = False,
    ) -> tuple[object, Audio]:
        """Run a fast distilled inference and write MP4 output."""
        video, audio = self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[],
            tiling_config=None,
            enhance_prompt=enhance_prompt,
        )

        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=1,
        )
        return video, audio


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast single-file LTX-2 inference")
    parser.add_argument("--distilled-checkpoint", required=True, help="Path to ltx-2.3-22b-distilled.safetensors")
    parser.add_argument("--spatial-upscaler", required=True, help="Path to spatial upscaler safetensors")
    parser.add_argument("--gemma-root", required=True, help="Path to Gemma-3 model directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--output", default="output.mp4", help="Output mp4 path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--no-fp8-cast", action="store_true", help="Disable fp8-cast quantization")
    return parser


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = _parser().parse_args()

    model = FastLTX2Model(
        distilled_checkpoint=args.distilled_checkpoint,
        spatial_upscaler=args.spatial_upscaler,
        gemma_root=args.gemma_root,
        use_fp8_cast=not args.no_fp8_cast,
    )
    model.infer(
        prompt=args.prompt,
        output_path=args.output,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        enhance_prompt=args.enhance_prompt,
    )


if __name__ == "__main__":
    main()
