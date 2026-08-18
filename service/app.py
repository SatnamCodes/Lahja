import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .asr_engine import engine as asr_engine
from .chat_engine import engine as chat_engine
from .mt_engine import engine as mt_engine
from .tts_engine import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lahja.app")

FRONTEND_DIR = config.BASE_DIR / "frontend"

app = FastAPI(title="LAHJA TTS Service", version="0.1.0")
app.mount("/audio", StaticFiles(directory=str(config.OUTPUT_DIR)), name="audio")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="trp")


class SpeakResponse(BaseModel):
    audio_url: str
    confidence: float
    method: str


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_language: str = Field(default="trp")
    target_language: str = Field(default="eng")


class TranslateResponse(BaseModel):
    translated_text: str
    confidence: float
    method: str


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    english_bridge: str
    confidence: float
    method: str


class TranscribeResponse(BaseModel):
    text: str
    confidence: float
    method: str


def _resolve_lang_code(code: str) -> str:
    resolved = config.MT_LANG_CODE_MAP.get(code)
    if resolved is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{code}'; expected one of {sorted(config.MT_LANG_CODE_MAP)}",
        )
    return resolved


@app.get("/health")
def health():
    return {"status": "ok", "device": engine.device}


@app.post("/api/speak", response_model=SpeakResponse)
def speak(req: SpeakRequest):
    if req.language != "trp":
        logger.warning("Requested language '%s' is not 'trp'; synthesizing anyway", req.language)
    try:
        result = engine.speak(req.text, req.language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    audio_url = f"{config.PUBLIC_BASE_URL}/audio/{result.file_path.name}"
    return SpeakResponse(audio_url=audio_url, confidence=result.confidence, method=result.method)


@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    source = _resolve_lang_code(req.source_language)
    target = _resolve_lang_code(req.target_language)
    try:
        result = mt_engine.translate(req.text, source, target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return TranslateResponse(
        translated_text=result.text, confidence=result.confidence, method=result.method
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = chat_engine.ask(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatResponse(
        answer=result.answer,
        english_bridge=result.english_bridge,
        confidence=result.confidence,
        method=result.method,
    )


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    suffix = "".join(c for c in Path(audio.filename or "").suffix if c.isalnum() or c == ".") or ".wav"
    upload_path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        upload_path.write_bytes(await audio.read())
        result = asr_engine.transcribe(upload_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        upload_path.unlink(missing_ok=True)

    return TranscribeResponse(text=result.text, confidence=result.confidence, method=result.method)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service.app:app", host="0.0.0.0", port=8000, reload=False)
