"""Single-file inference entrypoint for the original LTX-2 one-stage pipeline.

This script is intentionally self-contained at the repository level:
- it adds local package sources to ``sys.path`` so it can run without pip-installing
  ``ltx-core`` / ``ltx-pipelines`` as separate modules,
- it defaults to CPU when CUDA is unavailable,
- it keeps FP8 quantization opt-in and CUDA-only.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from ltx_core.types import Audio


def _build_simulated_sample(
    *,
    num_frames: int,
    height: int,
    width: int,
    frame_rate: float,
    seed: int,
) -> tuple[torch.Tensor, "Audio"]:
    """Return a synthetic video+audio pair for smoke-testing without model weights."""
    from ltx_core.types import Audio

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    video = torch.randint(
        low=0,
        high=255,
        size=(num_frames, height, width, 3),
        dtype=torch.uint8,
        generator=generator,
    )

    duration_seconds = max(num_frames / max(frame_rate, 1e-6), 0.1)
    sampling_rate = 16_000
    num_audio_samples = int(duration_seconds * sampling_rate)
    audio_waveform = torch.zeros((1, num_audio_samples), dtype=torch.float32)
    audio = Audio(waveform=audio_waveform, sampling_rate=sampling_rate)
    return video, audio


class LTX2Model:
    """Unified wrapper for the original LTX-2 one-stage pipeline."""

    def __init__(
        self,
        checkpoint: str | None,
        gemma_root: str | None,
        *,
        use_fp8_cast: bool = False,
        simulate: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.simulate = simulate

        if self.simulate:
            self.pipeline = None
            return

        if checkpoint is None or gemma_root is None:
            raise ValueError("checkpoint and gemma_root are required unless --simulate is enabled")

        quantization = None
        if use_fp8_cast and self.device.type == "cuda":
            from ltx_core.quantization.policy import QuantizationPolicy

            quantization = QuantizationPolicy.fp8_cast()
        elif use_fp8_cast:
            logging.warning("FP8 cast requested but CUDA is unavailable; running without FP8 quantization.")

        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

        self.pipeline = TI2VidOneStagePipeline(
            checkpoint_path=checkpoint,
            gemma_root=gemma_root,
            loras=[],
            device=self.device,
            quantization=quantization,
        )

    @torch.inference_mode()
    def list_root_modules(self) -> list[tuple[str, torch.nn.Module]]:
        """Return the root ``torch.nn.Module`` instances used by this model.

        Modules are discovered dynamically from zero-argument ``ModelLedger``
        methods and returned as ``(name, module)`` tuples.
        """
        if self.simulate or self.pipeline is None:
            return []

        model_ledger = self.pipeline.model_ledger
        modules: list[tuple[str, torch.nn.Module]] = []

        for name in sorted(dir(model_ledger)):
            if name.startswith("_"):
                continue

            candidate = getattr(model_ledger, name)
            if not callable(candidate):
                continue

            signature = inspect.signature(candidate)
            requires_arguments = any(
                parameter.default is inspect.Signature.empty
                and parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                for parameter in signature.parameters.values()
            )
            if requires_arguments:
                continue

            try:
                value = candidate()
            except (TypeError, ValueError):
                continue

            if isinstance(value, torch.nn.Module):
                modules.append((name, value))

        return modules

    @torch.inference_mode()
    def infer(
        self,
        prompt: str,
        output_path: str,
        *,
        negative_prompt: str = "",
        seed: int = 42,
        height: int = 1080,
        width: int = 1920,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 1,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 3.0,
        enhance_prompt: bool = False,
    ) -> tuple[Any, "Audio"]:
        if self.simulate:
            video, audio = _build_simulated_sample(
                num_frames=num_frames,
                height=height,
                width=width,
                frame_rate=frame_rate,
                seed=seed,
            )
            torch.save({"video": video, "audio": audio.waveform, "sampling_rate": audio.sampling_rate}, output_path)
            logging.info("Saved simulated sample tensor bundle to %s", output_path)
            return video, audio

        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_pipelines.utils.media_io import encode_video

        video, audio = self.pipeline(  # type: ignore[misc]
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
    parser.add_argument("--checkpoint", help="Path to ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--gemma-root", help="Path to Gemma-3 model directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt text")
    parser.add_argument("--output", default="output.mp4", help="Output path (mp4 for real inference, .pt for --simulate)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--video-cfg-scale", type=float, default=3.0)
    parser.add_argument("--audio-cfg-scale", type=float, default=3.0)
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--fp8-cast", action="store_true", help="Enable fp8-cast quantization (CUDA only)")
    parser.add_argument("--simulate", action="store_true", help="Run with synthetic outputs and skip loading checkpoints")
    return parser


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = _parser().parse_args()

    if not args.simulate and (not args.checkpoint or not args.gemma_root):
        raise SystemExit("--checkpoint and --gemma-root are required unless --simulate is enabled")

    model = LTX2Model(
        checkpoint=args.checkpoint,
        gemma_root=args.gemma_root,
        use_fp8_cast=args.fp8_cast,
        simulate=args.simulate,
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
