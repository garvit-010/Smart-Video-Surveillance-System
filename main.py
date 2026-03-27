"""Entry point for the Smart Video Surveillance pipeline."""

import argparse
import os
import sys

from src.pipeline.pipeline import run_pipeline
from src.utils.utils       import log


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input and output paths."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Smart Video Surveillance — Multi-Stage Computer Vision Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input data/sample_video.mp4\n"
            "  python main.py --input data/sample_video.mp4 --output outputs/result.mp4\n"
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        metavar="VIDEO_PATH",
        help="Path to the input video file (e.g. data/sample_video.mp4)",
    )

    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "processed_video.mp4"),
        metavar="OUTPUT_PATH",
        help="Destination path for the annotated output video "
             "(default: outputs/processed_video.mp4)",
    )

    return parser.parse_args()


def main() -> None:
    """Run the video processing pipeline."""
    args = parse_args()

    log.info("=" * 52)
    log.info("  Smart Video Surveillance Pipeline  ")
    log.info("=" * 52)
    log.info(f"Input  : {args.input}")
    log.info(f"Output : {args.output}")

    try:
        run_pipeline(input_path=args.input, output_path=args.output)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        log.warn("Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
