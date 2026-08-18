import logging

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .tts_engine import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lahja.app")

app = FastAPI(title="LAHJA TTS Service", version="0.1.0")
app.mount("/audio", StaticFiles(directory=str(config.OUTPUT_DIR)), name="audio")


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="trp")


class SpeakResponse(BaseModel):
    audio_url: str
    confidence: float
    method: str


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service.app:app", host="0.0.0.0", port=8000, reload=False)
