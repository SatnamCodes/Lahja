"""
Kokborok (trp) question -> Kokborok answer chatbot.

No LLM has native Kokborok fluency, so this bridges through English: the
incoming Kokborok question is translated to English with mt_engine, sent to
a free-tier hosted LLM (Groq by default), and the English answer is
translated back to Kokborok before being returned. Every response is
labeled with this bridge method so the approximation is never silently
presented as native Kokborok generation.
"""

import logging

import requests

from . import config
from .mt_engine import engine as mt_engine

logger = logging.getLogger("lahja.chat")


class ChatResult:
    def __init__(self, answer: str, confidence: float, method: str, english_bridge: str):
        self.answer = answer
        self.confidence = confidence
        self.method = method
        self.english_bridge = english_bridge


class ChatEngine:
    def ask(self, text: str) -> ChatResult:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and export it as GROQ_API_KEY."
            )

        question_en = mt_engine.translate(text, config.MT_LANG_TRP, config.MT_LANG_ENG).text

        try:
            response = requests.post(
                config.GROQ_API_URL,
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                json={
                    "model": config.GROQ_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. Answer concisely in "
                                "plain, simple English sentences - your answer will "
                                "be machine-translated into Kokborok, so avoid "
                                "idioms, slang, and markdown formatting."
                            ),
                        },
                        {"role": "user", "content": question_en},
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.exception("Groq chat completion request failed")
            raise RuntimeError(f"LLM request failed: {e}") from e

        answer_en = response.json()["choices"][0]["message"]["content"].strip()
        answer_trp = mt_engine.translate(answer_en, config.MT_LANG_ENG, config.MT_LANG_TRP).text

        return ChatResult(
            answer=answer_trp,
            confidence=0.5,
            method=config.METHOD_CHAT_MT_BRIDGE,
            english_bridge=answer_en,
        )


engine = ChatEngine()
