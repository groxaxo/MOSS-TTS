"""OpenAI-compatible FastAPI server for MOSS-TTS-Realtime."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

try:
    from .app import BackendPaths, GenerationConfig, SAMPLE_RATE, StreamingConfig, StreamingRequest, StreamingTTSDemo
except ImportError:
    from app import BackendPaths, GenerationConfig, SAMPLE_RATE, StreamingConfig, StreamingRequest, StreamingTTSDemo

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = os.getenv("MOSS_TTS_MODEL_PATH", "OpenMOSS-Team/MOSS-TTS-Realtime")
DEFAULT_TOKENIZER_PATH = os.getenv("MOSS_TTS_TOKENIZER_PATH", DEFAULT_MODEL_PATH)
DEFAULT_CODEC_MODEL_PATH = os.getenv("MOSS_TTS_CODEC_MODEL_PATH", "OpenMOSS-Team/MOSS-Audio-Tokenizer")
DEFAULT_DEVICE = os.getenv("MOSS_TTS_DEVICE", "cuda:0")
DEFAULT_USER_TEXT = os.getenv(
    "MOSS_TTS_API_USER_TEXT",
    "Please read the assistant response naturally and conversationally.",
)
DEFAULT_MAX_LENGTH = int(os.getenv("MOSS_TTS_MAX_LENGTH", "384"))
DEFAULT_TEXT_CHUNK_TOKENS = int(os.getenv("MOSS_TTS_TEXT_CHUNK_TOKENS", "8"))
DEFAULT_DECODE_CHUNK_FRAMES = int(os.getenv("MOSS_TTS_DECODE_CHUNK_FRAMES", "24"))
DEFAULT_DECODE_OVERLAP_FRAMES = int(os.getenv("MOSS_TTS_DECODE_OVERLAP_FRAMES", "4"))
DEFAULT_CHUNK_DURATION = float(os.getenv("MOSS_TTS_CHUNK_DURATION", "0.24"))
DEFAULT_TEMPERATURE = float(os.getenv("MOSS_TTS_TEMPERATURE", "0.8"))
DEFAULT_TOP_P = float(os.getenv("MOSS_TTS_TOP_P", "0.6"))
DEFAULT_TOP_K = int(os.getenv("MOSS_TTS_TOP_K", "30"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("MOSS_TTS_REPETITION_PENALTY", "1.1"))
DEFAULT_REPETITION_WINDOW = int(os.getenv("MOSS_TTS_REPETITION_WINDOW", "50"))
WARMUP_ON_START = os.getenv("MOSS_TTS_WARMUP_ON_START", "true").lower() in ("true", "1", "yes")
MAX_CONCURRENT = max(1, int(os.getenv("MOSS_TTS_MAX_CONCURRENT", "1")))

_generation_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT)
_demo = StreamingTTSDemo()

_VOICE_PRESETS = {
    "alloy": APP_DIR / "audio" / "prompt_audio.mp3",
    "echo": APP_DIR / "audio" / "prompt_audio1.mp3",
    "fable": APP_DIR / "audio" / "prompt_audio.mp3",
    "nova": APP_DIR / "audio" / "prompt_audio1.mp3",
    "onyx": APP_DIR / "audio" / "prompt_audio.mp3",
    "shimmer": APP_DIR / "audio" / "prompt_audio1.mp3",
    "default": None,
}

_SUPPORTED_MODELS = {
    "tts-1": DEFAULT_MODEL_PATH,
    "tts-1-hd": DEFAULT_MODEL_PATH,
    "moss-tts-realtime": DEFAULT_MODEL_PATH,
}


class OpenAISpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="tts-1")
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = Field(default="alloy")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = Field(default=False)


class VoiceInfo(BaseModel):
    id: str
    name: str
    description: str | None = None


def _resolve_attn_implementation() -> str:
    requested = os.getenv("MOSS_TTS_ATTN_IMPLEMENTATION", "auto").strip().lower()
    if requested in {"sdpa", "flash_attention_2", "eager"}:
        return requested
    if requested not in {"", "auto"}:
        raise RuntimeError(f"Unsupported MOSS_TTS_ATTN_IMPLEMENTATION value: {requested}")

    try:
        import torch

        if torch.cuda.is_available():
            # Prefer SDPA over flash_attention_2 when torch.compile is active:
            # Triton-compiled SDPA kernels fuse better on Ampere GPUs (SM 8.x)
            # and outperform flash_attention_2 for typical TTS sequence lengths.
            return "sdpa"
    except Exception:
        pass
    return "eager"


def _backend_paths() -> BackendPaths:
    return BackendPaths(
        model_path=DEFAULT_MODEL_PATH,
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        codec_model_path=DEFAULT_CODEC_MODEL_PATH,
        device_str=DEFAULT_DEVICE,
        attn_impl=_resolve_attn_implementation(),
    )


def _content_type(audio_format: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }[audio_format]


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    import wave

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def _pcm_bytes(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def _encode_audio(audio: np.ndarray, sample_rate: int, response_format: str) -> bytes:
    if response_format == "wav":
        return _wav_bytes(audio, sample_rate)
    if response_format == "pcm":
        return _pcm_bytes(audio)

    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(f"Compressed output requires pydub: {exc}") from exc

    wav_bytes = _wav_bytes(audio, sample_rate)
    segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    output = io.BytesIO()
    export_kwargs = {
        "mp3": {"format": "mp3", "bitrate": "192k"},
        "opus": {"format": "opus", "bitrate": "128k"},
        "aac": {"format": "adts", "bitrate": "192k"},
        "flac": {"format": "flac"},
    }[response_format]
    export_format = export_kwargs.pop("format")
    segment.export(output, format=export_format, **export_kwargs)
    return output.getvalue()


def _voice_prompt_path(voice: str) -> tuple[str | None, str]:
    normalized = voice.strip().lower()
    if not normalized:
        raise HTTPException(status_code=400, detail="voice is required")

    if normalized in _VOICE_PRESETS:
        prompt_path = _VOICE_PRESETS[normalized]
        if prompt_path is not None and not prompt_path.exists():
            raise HTTPException(status_code=500, detail=f"Bundled voice prompt is missing: {prompt_path}")
        return (str(prompt_path.resolve()) if prompt_path is not None else None, normalized)

    candidate = Path(voice).expanduser()
    if candidate.is_file():
        return str(candidate.resolve()), candidate.stem

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported voice '{voice}'. Available voices: {', '.join(sorted(_VOICE_PRESETS))}",
    )


def _build_streaming_request(payload: OpenAISpeechRequest, prompt_audio: str | None) -> StreamingRequest:
    if payload.model not in _SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{payload.model}'. Supported models: {', '.join(sorted(_SUPPORTED_MODELS))}",
        )
    if abs(payload.speed - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail="speed != 1.0 is not yet supported by the MOSS realtime OpenAI-compatible server",
        )

    return StreamingRequest(
        user_text=DEFAULT_USER_TEXT,
        assistant_text=payload.input.strip(),
        prompt_audio=prompt_audio,
        user_audio=None,
        use_default_prompt=False,
        use_default_user=False,
        generation=GenerationConfig(
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            repetition_window=DEFAULT_REPETITION_WINDOW,
            do_sample=True,
            max_length=DEFAULT_MAX_LENGTH,
            seed=None,
        ),
        streaming=StreamingConfig(
            text_chunk_tokens=DEFAULT_TEXT_CHUNK_TOKENS,
            input_delay=0.0,
            decode_chunk_frames=DEFAULT_DECODE_CHUNK_FRAMES,
            decode_overlap_frames=DEFAULT_DECODE_OVERLAP_FRAMES,
            chunk_duration=DEFAULT_CHUNK_DURATION,
            prebuffer_seconds=0.0,
            buffer_threshold_seconds=0.0,
        ),
        backend=_backend_paths(),
    )


def _render_audio(request: StreamingRequest) -> tuple[np.ndarray, int, dict[str, float]]:
    started_at = time.monotonic()
    first_audio_at = None
    sample_rate = SAMPLE_RATE
    chunks: list[np.ndarray] = []

    with _generation_semaphore:
        for event in _demo.run_stream(request):
            if event.audio is None:
                continue
            sample_rate, chunk = event.audio
            chunk_array = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if chunk_array.size == 0:
                continue
            if first_audio_at is None:
                first_audio_at = time.monotonic()
            chunks.append(chunk_array)

    if not chunks:
        raise RuntimeError("No audio waveform chunks were generated.")

    audio = np.concatenate(chunks, axis=0)
    elapsed = time.monotonic() - started_at
    audio_seconds = float(audio.size) / float(sample_rate)
    ttfb_ms = (first_audio_at - started_at) * 1000.0 if first_audio_at is not None else float("inf")
    rtf = elapsed / audio_seconds if audio_seconds > 0 else float("inf")
    metrics = {
        "elapsed_seconds": elapsed,
        "audio_seconds": audio_seconds,
        "ttfb_ms": ttfb_ms,
        "rtf": rtf,
    }
    return audio, sample_rate, metrics


def _stream_pcm(request: StreamingRequest):
    started_at = time.monotonic()
    first_audio_at = None
    sample_rate = SAMPLE_RATE
    total_samples = 0
    chunk_count = 0

    with _generation_semaphore:
        for event in _demo.run_stream(request):
            if event.audio is None:
                continue
            sample_rate, chunk = event.audio
            chunk_array = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if chunk_array.size == 0:
                continue
            if first_audio_at is None:
                first_audio_at = time.monotonic()
                logger.info("MOSS realtime stream started in %.1f ms", (first_audio_at - started_at) * 1000.0)
            total_samples += int(chunk_array.size)
            chunk_count += 1
            yield _pcm_bytes(chunk_array)

    elapsed = time.monotonic() - started_at
    audio_seconds = float(total_samples) / float(sample_rate) if sample_rate else 0.0
    rtf = elapsed / audio_seconds if audio_seconds > 0 else float("inf")
    logger.info(
        "MOSS realtime stream finished: chunks=%s audio=%.2fs elapsed=%.2fs rtf=%.3f",
        chunk_count,
        audio_seconds,
        elapsed,
        rtf,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if WARMUP_ON_START:
        logger.info("Warming MOSS realtime backend on startup (loading model + compiling)")
        backend = await asyncio.to_thread(_demo.get_or_load_backend, _backend_paths())
        # Run warmup generations with varied text lengths to prime torch.compile
        # caches for common sequence shapes. This avoids cold-compile stalls.
        warmup_texts = [
            "Hello, warmup.",
            "Warmup text to prime the model compilation cache on startup for best latency.",
        ]
        for idx, wtext in enumerate(warmup_texts):
            logger.info("Running warmup %d/%d (%d chars)...", idx + 1, len(warmup_texts), len(wtext))
            warmup_request = _build_streaming_request(
                OpenAISpeechRequest(model="tts-1", input=wtext, voice="default"), None,
            )
            try:
                def _warmup(req=warmup_request):
                    for event in _demo.run_stream(req):
                        pass
                await asyncio.to_thread(_warmup)
            except Exception as exc:
                logger.warning("Warmup %d failed (non-fatal): %s", idx + 1, exc)
        logger.info("Warmup complete — torch.compile caches are hot")
    yield


app = FastAPI(
    title="MOSS-TTS-Realtime OpenAI API",
    description="OpenAI-compatible FastAPI server for MOSS-TTS-Realtime",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("MOSS_TTS_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "MOSS-TTS-Realtime OpenAI API",
        "version": "0.1.0",
        "device": DEFAULT_DEVICE,
        "attn_implementation": _resolve_attn_implementation(),
        "endpoints": {
            "speech": ["/audio/speech", "/v1/audio/speech"],
            "models": ["/audio/models", "/v1/audio/models"],
            "voices": ["/audio/voices", "/v1/audio/voices"],
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEFAULT_DEVICE,
        "model_path": DEFAULT_MODEL_PATH,
        "codec_model_path": DEFAULT_CODEC_MODEL_PATH,
        "attn_implementation": _resolve_attn_implementation(),
    }


def _model_payload():
    return {
        "models": [
            {"id": model_id, "name": model_id, "owned_by": "openmoss"}
            for model_id in sorted(_SUPPORTED_MODELS)
        ]
    }


def _voice_payload():
    voices = [
        VoiceInfo(
            id=voice_id,
            name=voice_id,
            description=(
                "No prompt audio; generic realtime voice."
                if prompt is None
                else f"Bundled prompt voice based on {prompt.name}."
            ),
        ).model_dump()
        for voice_id, prompt in _VOICE_PRESETS.items()
    ]
    return {"voices": voices}


@app.get("/audio/models")
@app.get("/v1/audio/models")
async def list_models():
    return _model_payload()


@app.get("/audio/voices")
@app.get("/v1/audio/voices")
async def list_voices():
    return _voice_payload()


@app.post("/audio/speech")
@app.post("/v1/audio/speech")
@app.post("/v1/speech/audio")
async def create_speech(payload: OpenAISpeechRequest):
    prompt_audio, resolved_voice = _voice_prompt_path(payload.voice)
    request = _build_streaming_request(payload, prompt_audio)

    logger.info(
        "speech request model=%s voice=%s format=%s stream=%s chars=%s attn=%s",
        payload.model,
        resolved_voice,
        payload.response_format,
        payload.stream,
        len(payload.input),
        request.backend.attn_impl,
    )

    if payload.stream:
        if payload.response_format != "pcm":
            raise HTTPException(
                status_code=400,
                detail="stream=true currently requires response_format='pcm'",
            )
        return StreamingResponse(
            _stream_pcm(request),
            media_type=_content_type("pcm"),
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": "inline; filename=speech.pcm",
            },
        )

    try:
        audio, sample_rate, metrics = await asyncio.to_thread(_render_audio, request)
        encoded = await asyncio.to_thread(_encode_audio, audio, sample_rate, payload.response_format)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Speech generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=encoded,
        media_type=_content_type(payload.response_format),
        headers={
            "Content-Disposition": f"attachment; filename=speech.{payload.response_format}",
            "X-MOSS-TTFB-MS": f"{metrics['ttfb_ms']:.1f}",
            "X-MOSS-RTF": f"{metrics['rtf']:.4f}",
            "X-MOSS-AUDIO-SECONDS": f"{metrics['audio_seconds']:.4f}",
        },
    )


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=os.getenv("MOSS_TTS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(
        app,
        host=os.getenv("MOSS_TTS_HOST", "0.0.0.0"),
        port=int(os.getenv("MOSS_TTS_PORT", "8012")),
        log_level=os.getenv("MOSS_TTS_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
