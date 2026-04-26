#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Run the SAM3 agentic pipeline locally using Gemma 4 via Ollama.

Ollama manages model downloads, quantization, and an OpenAI-compatible REST
server on your laptop — no GPU server or cloud API key required.

Prerequisites
-------------
1. Install Ollama:        https://ollama.com
2. Pull a Gemma 4 model:  ollama pull gemma4:12b
3. Start Ollama server:   ollama serve   (usually starts automatically)
4. Install SAM3 deps:     pip install -e ".[notebooks]"

Recommended Gemma 4 variants
------------------------------
  gemma4:4b   ~4 GB RAM   fastest, less accurate
  gemma4:12b  ~8 GB RAM   good balance for most laptops  (default)
  gemma4:27b  ~20 GB RAM  best quality, full multimodal support

Usage
-----
  python scripts/run_agent_gemma4_local.py \\
      --image  assets/images/test_image.jpg \\
      --prompt "the leftmost child wearing blue vest"

  # Choose a different model variant
  python scripts/run_agent_gemma4_local.py \\
      --image  assets/images/test_image.jpg \\
      --prompt "red bicycle" \\
      --model  gemma4:27b

  # CPU-only fallback (slow but works without a GPU)
  python scripts/run_agent_gemma4_local.py \\
      --image  assets/images/test_image.jpg \\
      --prompt "red bicycle" \\
      --device cpu
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from contextlib import nullcontext
from functools import partial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_tags(host: str):
    """Return the parsed JSON from /api/tags, or None on any error."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def check_ollama_running(host: str) -> bool:
    return _ollama_tags(host) is not None


def check_model_available(host: str, model: str) -> bool:
    data = _ollama_tags(host)
    if data is None:
        return False
    available = [m["name"] for m in data.get("models", [])]
    model_base = model.split(":")[0]
    return any(model == a or a.startswith(model_base + ":") for a in available)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 agent with Gemma 4 running locally via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended Gemma 4 variants:
  gemma4:4b   – ~4 GB RAM, fastest
  gemma4:12b  – ~8 GB RAM, recommended  (default)
  gemma4:27b  – ~20 GB RAM, best multimodal quality

Pull a model:    ollama pull gemma4:12b
List models:     ollama list
        """,
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--prompt", required=True, help="Text description of the object to segment"
    )
    parser.add_argument(
        "--model",
        default="gemma4:12b",
        help="Ollama model tag (default: gemma4:12b)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output-dir",
        default="agent_output_gemma4",
        help="Directory for output files (default: agent_output_gemma4)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for LLM response (default: 4096)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Torch device override. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate agent messages for debugging",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate image path
    # ------------------------------------------------------------------
    if not os.path.isfile(args.image):
        print(f"❌  Image not found: {args.image}")
        sys.exit(1)
    image_path = os.path.abspath(args.image)

    # ------------------------------------------------------------------
    # Check Ollama availability and model presence
    # ------------------------------------------------------------------
    print(f"🔍 Checking Ollama at {args.ollama_host} ...")
    if not check_ollama_running(args.ollama_host):
        print(
            f"\n❌  Ollama is not reachable at {args.ollama_host}.\n"
            "    Start it with:  ollama serve\n"
            "    Install from:   https://ollama.com\n"
        )
        sys.exit(1)
    print("✅ Ollama is running.")

    if not check_model_available(args.ollama_host, args.model):
        print(
            f"\n❌  Model '{args.model}' is not available locally.\n"
            f"    Pull it with:  ollama pull {args.model}\n"
        )
        sys.exit(1)
    print(f"✅ Model '{args.model}' is available.")

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    import torch

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🖥️  Using device: {device}")

    # ------------------------------------------------------------------
    # Load SAM3 model
    # ------------------------------------------------------------------
    print("\n📦 Loading SAM3 model (this may take a moment on first run) ...")

    # Make sure the repo root is on sys.path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import sam3 as sam3_pkg
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    pkg_root = os.path.join(os.path.dirname(sam3_pkg.__file__), "..")
    bpe_path = os.path.abspath(os.path.join(pkg_root, "assets/bpe_simple_vocab_16e6.txt.gz"))

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Use bfloat16 on GPU/MPS to reduce memory usage; float32 on CPU
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=dtype)
        if device in ("cuda", "mps")
        else nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        sam3_model = build_sam3_image_model(bpe_path=bpe_path)
        if device != "cpu":
            sam3_model = sam3_model.to(device)
        processor = Sam3Processor(sam3_model, confidence_threshold=args.confidence)
        print("✅ SAM3 model loaded.")

        # --------------------------------------------------------------
        # Wire up the LLM and SAM3 callables
        # --------------------------------------------------------------
        from sam3.agent.client_llm import send_generate_request_ollama
        from sam3.agent.client_sam3 import call_sam_service as _call_sam_service
        from sam3.agent.inference import run_single_image_inference

        send_generate_request = partial(
            send_generate_request_ollama,
            model=args.model,
            ollama_host=args.ollama_host,
            max_tokens=args.max_tokens,
        )
        call_sam_service = partial(_call_sam_service, sam3_processor=processor)

        llm_config = {"name": args.model.replace(":", "_")}

        # --------------------------------------------------------------
        # Run inference
        # --------------------------------------------------------------
        print(f"\n🚀 Starting SAM3 Agent")
        print(f"   Image:  {image_path}")
        print(f"   Prompt: {args.prompt}")
        print(f"   LLM:    {args.model} @ {args.ollama_host}")
        print()

        output_image_path = run_single_image_inference(
            image_path=image_path,
            text_prompt=args.prompt,
            llm_config=llm_config,
            send_generate_request=send_generate_request,
            call_sam_service=call_sam_service,
            output_dir=args.output_dir,
            debug=args.debug,
        )

    if output_image_path:
        print(f"\n✅ Done!  Output image: {output_image_path}")
    else:
        print("\n⚠️  No output produced (output may already exist in the output dir).")


if __name__ == "__main__":
    main()
