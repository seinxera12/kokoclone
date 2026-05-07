"""
KokoClone TTS microservice.

Wraps core.cloner.KokoClone behind a simple HTTP API so the main server
(Python 3.11) can call it without any Python version conflicts.

Usage:
    cd /home/seinxera12/robotic_robo/kokoclone
    uv run python server.py          # CPU
    uv run --extra gpu python server.py  # GPU

Endpoints:
    POST /synthesize_stream  (primary — low-latency chunked streaming)
        Body: { "text": "...", "lang": "ja", "reference_audio": "/abs/path.wav" }
        Returns: chunked binary stream of length-prefixed WAV frames.
                 Wire format: 4-byte LE uint32 length + WAV bytes, repeated.
                 One frame is emitted per sentence in the input text, so the
                 first chunk arrives as soon as the first sentence is done
                 (~2–3 s) rather than waiting for the whole response (~10+ s).

    POST /synthesize  (fallback — returns complete WAV after full synthesis)
        Body: { "text": "...", "lang": "ja", "reference_audio": "/abs/path.wav" }
        Returns: audio/wav bytes

    GET /health
        Returns: { "status": "ok" }
"""

import asyncio
import io
import logging
import os
import re
import struct
import tempfile
import time
import uuid

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from kanade_tokenizer import load_audio
from pydantic import BaseModel

from core.chunked_convert import chunked_voice_conversion
from core.logging_setup import get_kokoclone_logger

logger = get_kokoclone_logger("server")

app = FastAPI(title="KokoClone TTS Service")

# Initialise once at startup — model weights download automatically on first run
logger.info("Loading KokoClone model…")
from core.cloner import KokoClone
cloner = KokoClone()
logger.info(f"KokoClone ready — sample_rate={cloner.sample_rate}, device={cloner.device}")

# Pre-warm the reference embedding for the default reference audio so the
# very first synthesis request doesn't pay the encoding cost.
_DEFAULT_REFERENCE = os.getenv(
    "KOKOCLONE_REFERENCE_AUDIO",
    os.path.join(os.path.dirname(__file__), "..", "voices_reference", "reference_ja.wav"),
)
if os.path.exists(_DEFAULT_REFERENCE):
    logger.info(f"Pre-warming reference embedding for {_DEFAULT_REFERENCE!r}…")
    cloner.precompute_reference(_DEFAULT_REFERENCE)
    logger.info("Reference embedding cached — first request will reuse it")

# Semaphore: only one VC (Kanade) call runs at a time — the model is not
# thread-safe and the 6 GB GPU can't hold two concurrent activations.
# Using a semaphore (not a lock) so the event loop stays responsive while
# waiting; other coroutines can run freely.
_vc_sem = asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences_ja(text: str) -> list[str]:
    """
    Split Japanese (or mixed) text into individual sentences.

    Splits on 。！？…!? followed by optional whitespace/newlines.
    Strips leading/trailing whitespace and blank lines.  Sentences shorter
    than 2 characters are merged into the next one to avoid Kokoro artifacts
    on very short utterances.

    Also filters out sentences that are purely URLs or source citations
    (e.g. "この情報は...：https://...") since the Japanese G2P cannot handle
    raw URLs and they add no value when spoken aloud.
    """
    # Normalise newlines and split on sentence-ending punctuation
    parts = re.split(r'(?<=[。！？…!?])\s*', text.strip())
    sentences: list[str] = [p.strip() for p in parts if p.strip()]

    # Merge fragments that are too short
    merged: list[str] = []
    carry = ""
    for s in sentences:
        combined = carry + s
        if len(combined) < 2:
            carry = combined
        else:
            merged.append(combined)
            carry = ""
    if carry:
        if merged:
            merged[-1] += carry
        else:
            merged.append(carry)

    # Filter sentences that contain a URL — the Japanese G2P crashes on
    # raw URLs and they are meaningless when spoken aloud.
    filtered = [s for s in merged if not re.search(r'https?://', s)]

    return filtered if filtered else (merged if merged else [text.strip()])


def _sanitize_for_g2p(text: str) -> str:
    """
    Clean text before passing to the Japanese G2P (misaki/cutlet).

    misaki's cutlet crashes with AssertionError on bare digit tokens
    (e.g. "10", "2024") because it tries to romanise them as words.
    Replace standalone digit sequences with their Japanese reading so
    the G2P never sees a raw number.

    Also strips any residual URLs that slipped through sentence filtering.
    """
    # Remove URLs entirely
    text = re.sub(r'https?://\S+', '', text)

    # Convert standalone digit sequences to kanji numerals.
    # Simple single/double digit mapping covers the common cases.
    # Longer numbers are rare in conversational TTS text.
    _digit_map = {
        '0': 'ゼロ', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
        '10': '十', '11': '十一', '12': '十二', '13': '十三',
        '14': '十四', '15': '十五', '16': '十六', '17': '十七',
        '18': '十八', '19': '十九', '20': '二十', '30': '三十',
        '40': '四十', '50': '五十', '100': '百', '1000': '千',
    }

    def _replace_digits(m: re.Match) -> str:
        n = m.group(0)
        return _digit_map.get(n, n)  # fall back to original if not in map

    text = re.sub(r'\b\d+\b', _replace_digits, text)
    return text.strip()


def _numpy_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 numpy array to WAV bytes (PCM16)."""
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm16, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


async def _synthesize_sentence(
    sentence: str,
    kokoro,
    g2p,
    voice: str,
    ref_emb: torch.Tensor,
    label: str,
) -> bytes | None:
    """
    Synthesise one sentence: Kokoro TTS → Kanade VC → WAV bytes.

    Runs Kokoro in a thread executor (blocking ONNX call) then runs Kanade
    VC under the semaphore so only one VC call is active at a time.

    Returns WAV bytes, or None on error.
    """
    loop = asyncio.get_event_loop()

    # ── Kokoro TTS (blocking, run in thread) ────────────────────────────────
    _tts_t0 = time.perf_counter()

    # Sanitize before G2P: remove URLs and convert bare digits to kanji so
    # misaki/cutlet doesn't crash with AssertionError on digit tokens.
    clean_sentence = _sanitize_for_g2p(sentence)
    if not clean_sentence:
        logger.info(f"[{label}] sentence empty after sanitization, skipping")
        return None

    def _run_kokoro():
        if g2p:
            phonemes, _ = g2p(clean_sentence)
            return kokoro.create(phonemes, voice=voice, speed=1.0, is_phonemes=True)
        else:
            return kokoro.create(clean_sentence, voice=voice, speed=0.9, lang="en-us")

    try:
        audio_chunk, sr = await loop.run_in_executor(None, _run_kokoro)
    except Exception as exc:
        logger.error(f"[{label}] Kokoro TTS error: {exc}", exc_info=True)
        return None

    tts_elapsed = time.perf_counter() - _tts_t0
    audio_duration = len(audio_chunk) / sr
    logger.info(
        f"stream_tts_sentence label={label} tts_elapsed={tts_elapsed:.3f} "
        f"audio_duration={audio_duration:.3f} text={sentence[:40]!r}"
    )

    # ── Kanade VC (GPU, serialised via semaphore) ────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio_chunk.astype(np.float32), sr)

        _vc_t0 = time.perf_counter()
        async with _vc_sem:
            source_wav = load_audio(tmp_path, sample_rate=cloner.sample_rate).to(cloner.device)
            with torch.inference_mode():
                converted = chunked_voice_conversion(
                    kanade=cloner.kanade,
                    vocoder_model=cloner.vocoder,
                    source_wav=source_wav,
                    ref_wav=None,
                    sample_rate=cloner.sample_rate,
                    request_id=label,
                    ref_embedding=ref_emb,
                )
        vc_elapsed = time.perf_counter() - _vc_t0
        logger.info(f"stream_vc_sentence label={label} vc_elapsed={vc_elapsed:.3f}")

        return _numpy_to_wav_bytes(converted.numpy(), cloner.sample_rate)

    except Exception as exc:
        logger.error(f"[{label}] Kanade VC error: {exc}", exc_info=True)
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str
    lang: str = "ja"
    reference_audio: str  # absolute path on the service host


@app.get("/health")
def health():
    return {"status": "ok", "sample_rate": cloner.sample_rate}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    if not os.path.exists(req.reference_audio):
        raise HTTPException(status_code=400, detail=f"reference_audio not found: {req.reference_audio}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    request_id = uuid.uuid4().hex[:8]

    try:
        logger.info(f"request_received text={req.text[:80]!r} lang={req.lang!r} request_id={request_id} text_len={len(req.text)}")
        cloner.generate(
            text=req.text,
            lang=req.lang,
            reference_audio=req.reference_audio,
            output_path=tmp_path,
            request_id=request_id,
        )

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()

        logger.info(f"request_complete request_id={request_id} response_bytes={len(wav_bytes)}")
        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as exc:
        logger.error(f"request_failed request_id={request_id} exc_type={type(exc).__name__} message={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/synthesize_stream")
async def synthesize_stream(req: SynthesizeRequest):
    """
    Streaming synthesis endpoint.

    Splits the input text into sentences, then for each sentence:
      1. Runs Kokoro ONNX TTS in a thread executor
      2. Runs Kanade voice conversion on the GPU (serialised via semaphore)
      3. Immediately streams the resulting WAV frame back to the client

    This means the first audio chunk arrives after the first sentence is
    processed (~2–3 s) rather than waiting for the entire text (~10+ s).

    Wire format: each chunk is a 4-byte little-endian uint32 length prefix
    followed by that many bytes of WAV audio.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    if not os.path.exists(req.reference_audio):
        raise HTTPException(status_code=400, detail=f"reference_audio not found: {req.reference_audio}")

    request_id = uuid.uuid4().hex[:8]
    logger.info(
        f"stream_request_received text={req.text[:80]!r} lang={req.lang!r} "
        f"request_id={request_id} text_len={len(req.text)}"
    )

    async def _generate():
        _t_start = time.perf_counter()

        # ── Resolve Kokoro model + G2P ───────────────────────────────────────
        model_file, voices_file, vocab, g2p, voice, _ = cloner._get_config(req.lang)

        if model_file not in cloner.kokoro_cache:
            from kokoro_onnx import Kokoro
            kokoro = (
                Kokoro(model_file, voices_file, vocab_config=vocab)
                if vocab
                else Kokoro(model_file, voices_file)
            )
            cloner.kokoro_cache[model_file] = cloner._patch_kokoro_compat(kokoro)

        kokoro = cloner.kokoro_cache[model_file]

        # ── Cached reference embedding ───────────────────────────────────────
        ref_emb = cloner._get_ref_embedding(req.reference_audio)

        # ── Split text into sentences and process each one ───────────────────
        sentences = _split_sentences_ja(req.text)
        logger.info(
            f"stream_sentences request_id={request_id} "
            f"n_sentences={len(sentences)} sentences={[s[:30] for s in sentences]}"
        )

        chunks_yielded = 0
        for i, sentence in enumerate(sentences):
            label = f"{request_id}s{i}"
            wav_bytes = await _synthesize_sentence(
                sentence=sentence,
                kokoro=kokoro,
                g2p=g2p,
                voice=voice,
                ref_emb=ref_emb,
                label=label,
            )
            if wav_bytes:
                frame = struct.pack("<I", len(wav_bytes)) + wav_bytes
                yield frame
                # Yield control to the event loop so uvicorn flushes the send
                # buffer immediately.  Without this, uvicorn may hold the frame
                # in its internal buffer until the next chunk is ready or the
                # response ends — adding seconds of unnecessary delay.
                await asyncio.sleep(0)
                chunks_yielded += 1
                elapsed = time.perf_counter() - _t_start
                logger.info(
                    f"stream_chunk_sent request_id={request_id} chunk={i} "
                    f"elapsed={elapsed:.3f} bytes={len(wav_bytes)}"
                )

        total_elapsed = time.perf_counter() - _t_start
        logger.info(
            f"stream_complete request_id={request_id} "
            f"total_elapsed={total_elapsed:.3f} n_chunks={chunks_yielded}"
        )

    return StreamingResponse(
        _generate(),
        media_type="application/octet-stream",
        headers={"X-KokoClone-Request-Id": request_id},
    )


if __name__ == "__main__":
    host = os.getenv("KOKOCLONE_HOST", "0.0.0.0")
    port = int(os.getenv("KOKOCLONE_PORT", "5003"))
    uvicorn.run(app, host=host, port=port, log_level="info")
