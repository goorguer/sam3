#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
SAM3 Gradio Interface — segmentación visual con IA local.

Modos:
  • SAM3 directo  — rápido, sin LLM, ideal para bajo recursos / tiempo real.
  • Agente Gemma 4 — razonamiento multi-turno via Ollama (requiere ollama pull gemma4:12b).

Instalación de dependencias UI:
    pip install -e ".[ui]"

Uso:
    python app.py                    # abre en http://localhost:7860
    python app.py --preload          # carga SAM3 al arrancar
    python app.py --device cpu       # fuerza CPU
    python app.py --share            # enlace público Gradio
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import traceback
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_processor = None          # Sam3Processor instance
_proc_lock = threading.Lock()
_device: str = "cpu"
_torch_ctx = nullcontext   # autocast context factory (set after torch import)

# Colour palette for mask overlays (BGR for cv2, RGB for PIL)
_PALETTE_RGB = [
    (220,  50,  50), ( 50, 200,  50), ( 50,  50, 220),
    (220, 200,  50), (200,  50, 220), ( 50, 200, 200),
    (240, 140,  50), (140,  50, 240), ( 50, 240, 140),
    (200, 100,  80), ( 80, 200, 100), (100,  80, 200),
]

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_torch_ctx(device: str):
    """Return a no-op or autocast context factory depending on device."""
    import torch
    if device in ("cuda", "mps"):
        def ctx():
            return torch.autocast(device_type=device, dtype=torch.bfloat16)
        return ctx
    return nullcontext

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device_choice: str = "auto", confidence: float = 0.5) -> str:
    """Load SAM3 model. Thread-safe; returns status string."""
    global _processor, _device, _torch_ctx

    device = _detect_device() if device_choice == "auto" else device_choice

    with _proc_lock:
        if _processor is not None:
            return f"✅ SAM3 ya cargado  ({_device})"
        try:
            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import sam3 as _sam3_pkg

            pkg_root = Path(_sam3_pkg.__file__).parent.parent
            bpe_path = str(pkg_root / "assets" / "bpe_simple_vocab_16e6.txt.gz")

            if device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            model = build_sam3_image_model(bpe_path=bpe_path)
            if device != "cpu":
                model = model.to(device)

            _processor = Sam3Processor(model, confidence_threshold=confidence)
            _device = device
            _torch_ctx = _make_torch_ctx(device)
            return f"✅ SAM3 cargado  ({device})"

        except Exception as exc:
            return f"❌ Error cargando SAM3: {exc}\n{traceback.format_exc()}"

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_pil(img) -> Optional[Image.Image]:
    """Normalise Gradio image input → PIL RGB."""
    if img is None:
        return None
    if isinstance(img, dict):          # ImageEditor returns a dict
        raw = img.get("composite") or img.get("background")
        if raw is None:
            return None
        img = raw
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return None


def _resize(img: Image.Image, max_px: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _overlay(pil_img: Image.Image, state: dict) -> Image.Image:
    """Draw masks, boxes and score labels on a PIL image."""
    import torch

    scores = state.get("scores")
    masks  = state.get("masks")
    boxes  = state.get("boxes")

    if scores is None or (hasattr(scores, '__len__') and len(scores) == 0):
        return pil_img

    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8).copy()
    overlay = arr.copy()

    # normalise tensors → numpy
    def _np(t):
        if t is None:
            return None
        return t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)

    scores_np = _np(scores)
    masks_np  = _np(masks)   # (N, H, W) float
    boxes_np  = _np(boxes)   # (N, 4)  [x1 y1 x2 y2] pixels

    n = len(scores_np)
    for i in range(n):
        color = _PALETTE_RGB[i % len(_PALETTE_RGB)]

        # --- mask fill ---
        if masks_np is not None and i < len(masks_np):
            mask = masks_np[i] > 0.5          # (H, W) bool
            if mask.any():
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        mask,
                        np.clip(0.55 * arr[:, :, c] + 0.45 * color[c], 0, 255).astype(np.uint8),
                        overlay[:, :, c],
                    )

        # --- box + label ---
        if boxes_np is not None and i < len(boxes_np):
            x1, y1, x2, y2 = (int(v) for v in boxes_np[i])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color[::-1], 2)  # BGR
            label = f"#{i+1}  {scores_np[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color[::-1], -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(overlay)


def _results_md(state: dict, prompt: str) -> str:
    """Build a markdown summary of detected objects."""
    scores = state.get("scores")
    boxes  = state.get("boxes")

    def _np(t):
        if t is None:
            return None
        return t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)

    scores_np = _np(scores)
    boxes_np  = _np(boxes)

    if scores_np is None or len(scores_np) == 0:
        return f"### ❌ No se encontró «{prompt}»"

    n = len(scores_np)
    lines = [
        f"### ✅ {n} instancia(s) de «{prompt}»\n",
        "| # | Confianza | Posición (x₁ y₁ → x₂ y₂) |",
        "|---|-----------|--------------------------|",
    ]
    for i in range(n):
        sc = scores_np[i]
        if boxes_np is not None and i < len(boxes_np):
            x1, y1, x2, y2 = (int(v) for v in boxes_np[i])
            pos = f"{x1}, {y1} → {x2}, {y2}"
        else:
            pos = "—"
        lines.append(f"| {i+1} | **{sc:.3f}** | {pos} |")

    avg = float(np.mean(scores_np))
    lines.append(f"\n*Confianza media: {avg:.3f}*")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Direct SAM3 inference  (image)
# ---------------------------------------------------------------------------

def _run_sam3_on_pil(pil_img: Image.Image, prompt: str,
                     confidence: float) -> tuple[Image.Image, str]:
    """Core SAM3 text-prompt inference. Returns (result_img, markdown_info)."""
    import torch

    _processor.confidence_threshold = confidence
    with torch.inference_mode(), _torch_ctx():
        state = _processor.set_image(pil_img)
        state = _processor.set_text_prompt(prompt=prompt.strip(), state=state)

    result = _overlay(pil_img, state)
    info   = _results_md(state, prompt.strip())
    return result, info


def analyze_image_direct(img_input, prompt: str, confidence: float,
                         max_res: int, progress=gr.Progress()):
    if _processor is None:
        return None, "❌ Modelo no cargado — pulsa **⚡ Cargar SAM3**."
    if img_input is None:
        return None, "❌ Sin imagen de entrada."
    if not prompt.strip():
        return None, "❌ Escribe qué objeto quieres segmentar."

    try:
        progress(0.1, desc="Preparando imagen…")
        pil = _to_pil(img_input)
        if pil is None:
            return None, "❌ No se pudo leer la imagen."
        pil = _resize(pil, max_res)

        progress(0.4, desc="Codificando imagen…")
        result, info = _run_sam3_on_pil(pil, prompt, confidence)
        progress(1.0, desc="Listo.")
        return result, info

    except Exception as exc:
        return None, f"❌ Error: {exc}\n```\n{traceback.format_exc()}\n```"

# ---------------------------------------------------------------------------
# Gemma 4 agent inference  (image)
# ---------------------------------------------------------------------------

def analyze_image_agent(img_input, prompt: str, confidence: float, max_res: int,
                        ollama_model: str, ollama_host: str,
                        progress=gr.Progress()):
    if _processor is None:
        return None, "❌ Modelo no cargado — pulsa **⚡ Cargar SAM3**."
    if img_input is None:
        return None, "❌ Sin imagen de entrada."
    if not prompt.strip():
        return None, "❌ Escribe qué objeto quieres segmentar."

    try:
        import torch
        from sam3.agent.client_llm import send_generate_request_ollama
        from sam3.agent.client_sam3 import call_sam_service as _call_sam
        from sam3.agent.inference import run_single_image_inference

        progress(0.05, desc="Preparando imagen…")
        pil = _to_pil(img_input)
        if pil is None:
            return None, "❌ No se pudo leer la imagen."
        pil = _resize(pil, max_res)

        # Agent needs a file path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil.save(tmp.name, quality=92)
            tmp_path = tmp.name

        progress(0.15, desc="Iniciando agente Gemma 4…")
        _processor.confidence_threshold = confidence

        send_fn  = partial(send_generate_request_ollama,
                           model=ollama_model, ollama_host=ollama_host)
        call_sam = partial(_call_sam, sam3_processor=_processor)
        out_dir  = tempfile.mkdtemp(prefix="sam3_agent_")
        llm_cfg  = {"name": ollama_model.replace(":", "_")}

        progress(0.30, desc="Agente razonando… (puede tardar)")
        with torch.inference_mode(), _torch_ctx():
            out_img_path = run_single_image_inference(
                image_path=tmp_path,
                text_prompt=prompt.strip(),
                llm_config=llm_cfg,
                send_generate_request=send_fn,
                call_sam_service=call_sam,
                output_dir=out_dir,
            )

        os.unlink(tmp_path)
        progress(1.0, desc="Listo.")

        if out_img_path and os.path.exists(out_img_path):
            result = Image.open(out_img_path).convert("RGB")
            info = (
                f"### ✅ Agente Gemma 4 completado\n"
                f"Prompt: *«{prompt.strip()}»*\n\n"
                f"Resultado guardado en:\n`{out_img_path}`"
            )
        else:
            result = pil
            info = f"### ⚠️ El agente no encontró «{prompt.strip()}»"

        return result, info

    except Exception as exc:
        return None, f"❌ Error: {exc}\n```\n{traceback.format_exc()}\n```"

# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

def analyze_video(vid_path: str, prompt: str, confidence: float,
                  frame_skip: int, max_res: int,
                  progress=gr.Progress()):
    if _processor is None:
        return None, "❌ Modelo no cargado — pulsa **⚡ Cargar SAM3**."
    if not vid_path:
        return None, "❌ Sin vídeo de entrada."
    if not prompt.strip():
        return None, "❌ Escribe qué objeto quieres segmentar."

    try:
        import torch

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return None, f"❌ No se pudo abrir el vídeo: {vid_path}"

        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = min(1.0, max_res / max(w_src, h_src, 1))
        w_out = max(1, int(w_src * scale))
        h_out = max(1, int(h_src * scale))

        # Output video: same FPS as source (we duplicate last frame for skipped ones)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as _f:
            out_path = _f.name
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, src_fps, (w_out, h_out))

        _processor.confidence_threshold = confidence

        last_overlay: Optional[np.ndarray] = None
        fi = 0
        processed = 0
        total_det = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize
            if scale < 1.0:
                frame = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_AREA)
            else:
                frame = frame.copy()

            if fi % frame_skip == 0:
                progress(min(fi / total, 0.99),
                         desc=f"Fotograma {fi}/{total} — procesando…")

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                with torch.inference_mode(), _torch_ctx():
                    state = _processor.set_image(pil_frame)
                    state = _processor.set_text_prompt(prompt=prompt.strip(), state=state)

                n_det     = len(state.get("scores", []) or [])
                total_det += n_det
                result_pil = _overlay(pil_frame, state)
                last_overlay = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                processed += 1
            else:
                # Re-use last processed overlay for smooth output
                if last_overlay is None:
                    last_overlay = frame

            writer.write(last_overlay)
            fi += 1

        cap.release()
        writer.release()
        progress(1.0, desc="Vídeo listo.")

        dur_s  = total / src_fps
        p_fps  = processed / max(dur_s, 0.001)
        avg_d  = total_det / max(processed, 1)

        info = (
            f"### 📊 Resumen del vídeo\n"
            f"| Métrica | Valor |\n"
            f"|---------|-------|\n"
            f"| Prompt | *{prompt.strip()}* |\n"
            f"| Duración | {dur_s:.1f} s |\n"
            f"| Fotogramas totales | {fi} |\n"
            f"| Fotogramas procesados | {processed} (1 de cada {frame_skip}) |\n"
            f"| Velocidad efectiva | {p_fps:.1f} fps |\n"
            f"| Detecciones totales | {total_det} |\n"
            f"| Detecciones medias/fotograma | {avg_d:.2f} |\n"
        )
        return out_path, info

    except Exception as exc:
        return None, f"❌ Error: {exc}\n```\n{traceback.format_exc()}\n```"

# ---------------------------------------------------------------------------
# Ollama connectivity check
# ---------------------------------------------------------------------------

def check_ollama(host: str, model: str) -> str:
    import json, urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=6) as r:
            data = json.loads(r.read())
    except urllib.error.URLError as exc:
        return (
            f"❌ No se puede conectar a `{host}`: {exc.reason}\n\n"
            f"Asegúrate de que Ollama está corriendo: `ollama serve`"
        )
    except Exception as exc:
        return f"❌ Error inesperado: {exc}"

    available = [m["name"] for m in data.get("models", [])]
    gemma = [m for m in available if "gemma" in m.lower()]
    wanted_base = model.split(":")[0]
    has_model   = any(m == model or m.startswith(wanted_base + ":") for m in available)

    lines = ["✅ Ollama activo.\n"]
    if has_model:
        lines.append(f"✅ Modelo `{model}` disponible.")
    else:
        lines.append(
            f"⚠️ Modelo `{model}` **no encontrado**.\n"
            f"Descárgalo con:\n```bash\nollama pull {model}\n```"
        )
    if gemma:
        lines.append(f"\nModelos Gemma disponibles: `{'`, `'.join(gemma)}`")
    elif not has_model:
        lines.append(f"\nNo hay ningún modelo Gemma. Modelos instalados: `{'`, `'.join(available) or 'ninguno'}`")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.title { text-align:center; margin-bottom:4px; }
.subtitle { text-align:center; color:#666; font-size:0.9em; margin-bottom:16px; }
.result-panel { min-height:420px; }
footer { display:none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)


def build_ui() -> gr.Blocks:

    with gr.Blocks(title="SAM3 — Segmentación con IA", theme=THEME, css=CSS) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown("# 🎯 SAM3 — Segmentación Visual con IA", elem_classes="title")
        gr.Markdown(
            "Segmenta objetos en imágenes y vídeos con **SAM3** "
            "(rápido) o con razonamiento inteligente mediante **Gemma 4** local (Ollama).",
            elem_classes="subtitle",
        )

        # ── Model loader ────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            device_sel = gr.Radio(
                choices=["auto", "cuda", "mps", "cpu"],
                value="auto",
                label="Dispositivo",
                scale=2,
            )
            load_btn = gr.Button("⚡ Cargar SAM3", variant="primary", scale=1, size="lg")
            load_status = gr.Textbox(
                value="⚠️ Pulsa «Cargar SAM3» para inicializar el modelo.",
                label="Estado",
                interactive=False,
                scale=4,
            )

        gr.Divider()

        # ── Shared settings state ────────────────────────────────────────────
        # (these are read in the tab callbacks; declared here so they're in scope)
        ollama_model_state = gr.State("gemma4:12b")
        ollama_host_state  = gr.State("http://localhost:11434")

        # ================================================================
        # TABS
        # ================================================================
        with gr.Tabs():

            # ── TAB 1: Image ─────────────────────────────────────────────
            with gr.Tab("📷 Imagen"):
                with gr.Row():

                    # Left column: controls
                    with gr.Column(scale=1):
                        img_in = gr.Image(
                            label="Imagen — sube un archivo, pega del portapapeles o captura desde la cámara",
                            sources=["upload", "webcam", "clipboard"],
                            type="numpy",
                            height=380,
                        )
                        img_prompt = gr.Textbox(
                            label="¿Qué quieres segmentar?",
                            placeholder="Ej: persona, bicicleta roja, perro tumbado…",
                            lines=1,
                        )
                        with gr.Row():
                            img_conf = gr.Slider(
                                0.05, 0.95, value=0.5, step=0.05,
                                label="Umbral de confianza",
                            )
                            img_res = gr.Slider(
                                256, 1024, value=640, step=64,
                                label="Resolución máx. (px)",
                                info="Reduce para mayor velocidad",
                            )
                        img_mode = gr.Radio(
                            choices=["SAM3 directo  (rápido)", "Agente Gemma 4  (inteligente)"],
                            value="SAM3 directo  (rápido)",
                            label="Modo",
                        )
                        with gr.Row():
                            img_btn   = gr.Button("🔍 Analizar", variant="primary", scale=3)
                            img_clear = gr.Button("🗑️ Limpiar",  variant="secondary", scale=1)

                    # Right column: results
                    with gr.Column(scale=1, elem_classes="result-panel"):
                        img_out  = gr.Image(label="Resultado", height=380, interactive=False)
                        img_info = gr.Markdown("*El resultado aparecerá aquí.*")

            # ── TAB 2: Video ─────────────────────────────────────────────
            with gr.Tab("🎬 Vídeo"):
                with gr.Row():

                    with gr.Column(scale=1):
                        vid_in = gr.Video(
                            label="Vídeo — sube un archivo o graba con la cámara",
                            sources=["upload", "webcam"],
                            height=340,
                        )
                        vid_prompt = gr.Textbox(
                            label="¿Qué quieres segmentar?",
                            placeholder="Ej: coche rojo, persona caminando…",
                            lines=1,
                        )
                        with gr.Row():
                            vid_conf = gr.Slider(
                                0.05, 0.95, value=0.5, step=0.05,
                                label="Umbral de confianza",
                            )
                            vid_res = gr.Slider(
                                256, 1024, value=480, step=64,
                                label="Resolución máx. (px)",
                                info="Reduce para mayor velocidad",
                            )
                        vid_skip = gr.Slider(
                            1, 12, value=3, step=1,
                            label="Procesar 1 de cada N fotogramas",
                            info="Mayor N = más rápido, menos densidad temporal",
                        )
                        with gr.Row():
                            vid_btn  = gr.Button("▶️ Procesar vídeo", variant="primary", scale=3)
                            vid_stop = gr.Button("⏹️ Detener",        variant="stop",    scale=1)

                    with gr.Column(scale=1, elem_classes="result-panel"):
                        vid_out  = gr.Video(label="Vídeo segmentado", height=340)
                        vid_info = gr.Markdown("*El resumen aparecerá aquí.*")

            # ── TAB 3: Settings (Gemma 4) ────────────────────────────────
            with gr.Tab("⚙️ Gemma 4 / Ollama"):
                with gr.Row():
                    with gr.Column(scale=1):
                        cfg_model = gr.Dropdown(
                            choices=["gemma4:4b", "gemma4:12b", "gemma4:27b"],
                            value="gemma4:12b",
                            label="Variante de Gemma 4",
                            info="gemma4:12b recomendado para portátiles (~8 GB RAM)",
                            allow_custom_value=True,
                        )
                        cfg_host = gr.Textbox(
                            value="http://localhost:11434",
                            label="URL del servidor Ollama",
                        )
                        with gr.Row():
                            cfg_check = gr.Button("🔗 Verificar conexión", variant="secondary")
                            cfg_apply = gr.Button("💾 Aplicar configuración", variant="primary")
                        cfg_status = gr.Markdown("*Pulsa «Verificar conexión» para comprobar Ollama.*")

                    with gr.Column(scale=1):
                        gr.Markdown("""
### Cómo configurar Ollama con Gemma 4

**1. Instala Ollama**
```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh
```
O descarga desde [ollama.com](https://ollama.com)

**2. Descarga un modelo Gemma 4**
```bash
ollama pull gemma4:12b   # recomendado
# ollama pull gemma4:4b  # más ligero, menos preciso
# ollama pull gemma4:27b # mayor calidad, +20 GB RAM
```

**3. Inicia el servidor** *(suele iniciarse automáticamente)*
```bash
ollama serve
```

**4. En la pestaña Imagen**, selecciona **Agente Gemma 4**.

---

| Modelo | RAM mínima | Velocidad |
|--------|-----------|-----------|
| gemma4:4b  | ~4 GB  | ⚡ Rápida |
| gemma4:12b | ~8 GB  | ✅ Media  |
| gemma4:27b | ~20 GB | 🎯 Óptima |
                        """)

        # ================================================================
        # EVENT HANDLERS
        # ================================================================

        # -- Load model --------------------------------------------------
        def _handle_load(device_choice):
            return load_model(device_choice)

        load_btn.click(_handle_load, inputs=[device_sel], outputs=[load_status])

        # -- Image analysis ----------------------------------------------
        def _handle_image(img, prompt, conf, res, mode,
                          o_model, o_host, progress=gr.Progress()):
            if "Agente" in mode:
                return analyze_image_agent(
                    img, prompt, conf, res, o_model, o_host, progress)
            return analyze_image_direct(img, prompt, conf, res, progress)

        img_btn.click(
            _handle_image,
            inputs=[img_in, img_prompt, img_conf, img_res, img_mode,
                    ollama_model_state, ollama_host_state],
            outputs=[img_out, img_info],
        )

        def _clear_image():
            return None, None, "*El resultado aparecerá aquí.*"

        img_clear.click(_clear_image, outputs=[img_in, img_out, img_info])

        # -- Video analysis ----------------------------------------------
        vid_event = vid_btn.click(
            analyze_video,
            inputs=[vid_in, vid_prompt, vid_conf, vid_skip, vid_res],
            outputs=[vid_out, vid_info],
        )
        vid_stop.click(fn=None, cancels=[vid_event])

        # -- Ollama check & apply ----------------------------------------
        def _handle_check(host, model):
            return check_ollama(host, model)

        def _handle_apply(model, host):
            return (
                model, host,
                f"✅ Configuración guardada: **{model}** en `{host}`"
            )

        cfg_check.click(
            _handle_check,
            inputs=[cfg_host, cfg_model],
            outputs=[cfg_status],
        )

        cfg_apply.click(
            _handle_apply,
            inputs=[cfg_model, cfg_host],
            outputs=[ollama_model_state, ollama_host_state, cfg_status],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SAM3 Gradio UI")
    ap.add_argument("--host",    default="127.0.0.1",
                    help="Dirección del servidor (default: 127.0.0.1)")
    ap.add_argument("--port",    default=7860, type=int,
                    help="Puerto (default: 7860)")
    ap.add_argument("--share",   action="store_true",
                    help="Crear enlace público Gradio")
    ap.add_argument("--device",  default=None, choices=["cuda", "mps", "cpu"],
                    help="Dispositivo (default: detección automática)")
    ap.add_argument("--preload", action="store_true",
                    help="Cargar SAM3 al arrancar (no esperar al botón)")
    args = ap.parse_args()

    if args.preload:
        print("Precargando SAM3…")
        status = load_model(device_choice=args.device or "auto")
        print(status)

    demo = build_ui()
    demo.queue()   # needed for Progress() and long-running tasks
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=None,
    )
