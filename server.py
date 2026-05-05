"""
KokoClone TTS microservice.

Wraps core.cloner.KokoClone behind a simple HTTP API so the main server
(Python 3.11) can call it without any Python version conflicts.

Usage:
    cd /home/seinxera12/robotic_robo/kokoclone
    uv run python server.py          # CPU
    uv run --extra gpu python server.py  # GPU

Endpoints:
    POST /synthesize
        Body: { "text": "...", "lang": "ja", "reference_audio": "/path/to/ref.wav" }
        Returns: audio/wav bytes

    GET /health
        Returns: { "status": "ok" }
"""

import logging
import os
import tempfile
import uuid

import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from core.logging_setup import get_kokoclone_logger

logger = get_kokoclone_logger("server")

app = FastAPI(title="KokoClone TTS Service")

# Initialise once at startup — model weights download automatically on first run
logger.info("Loading KokoClone model…")
from core.cloner import KokoClone
cloner = KokoClone()
logger.info(f"KokoClone ready — sample_rate={cloner.sample_rate}, device={cloner.device}")


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

        # Read the WAV bytes directly — no need to decode with soundfile
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


if __name__ == "__main__":
    host = os.getenv("KOKOCLONE_HOST", "0.0.0.0")
    port = int(os.getenv("KOKOCLONE_PORT", "5003"))
    uvicorn.run(app, host=host, port=port, log_level="info")
