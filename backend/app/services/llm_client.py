"""
Unified Multi-Provider LLM Client for Decisio.

Supports:
1. Groq (GroqCloud - llama-3.3-70b-versatile, mixtral-8x7b-32768)
2. OpenAI (gpt-4o, gpt-4o-mini)
3. Anthropic Claude (claude-3-5-sonnet, claude-3-7-sonnet)
4. Grok / xAI (grok-2-latest, grok-beta via OpenAI-compatible endpoint)
5. Mock Provider (rule-based offline simulator for testing with 0 API keys)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client that routes completion requests to the configured LLM provider."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON dictionary response using the specified or default LLM provider."""
        active_provider = (provider or self.settings.llm_provider).lower()

        if active_provider == "groq":
            return await self._call_groq(system_prompt, user_prompt, model)
        elif active_provider == "grok" or active_provider == "xai":
            return await self._call_grok(system_prompt, user_prompt, model)
        elif active_provider == "anthropic" or active_provider == "claude":
            return await self._call_anthropic(system_prompt, user_prompt, model)
        elif active_provider == "openai":
            return await self._call_openai(system_prompt, user_prompt, model)
        else:
            return await self._call_mock(system_prompt, user_prompt)

    async def _call_groq(
        self, system_prompt: str, user_prompt: str, model: str | None = None
    ) -> dict[str, Any]:
        """Call Groq API (GroqCloud) using OpenAI-compatible endpoint with automatic model fallback."""
        api_key = self.settings.groq_api_key
        if not api_key:
            logger.warning("GROQ_API_KEY is not set. Falling back to mock extractor.")
            return await self._call_mock(system_prompt, user_prompt)

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.groq_base_url,
        )
        target_model = model or self.settings.groq_model

        try:
            response = await client.chat.completions.create(
                model=target_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"{system_prompt}\nReturn JSON strictly."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Groq call with model {target_model} failed ({e}). Attempting auto-detection fallback...")
            try:
                # Discover available models on this key
                models_resp = await client.models.list()
                avail = [m.id for m in models_resp.data if "whisper" not in m.id and "guard" not in m.id]
                if avail:
                    fallback_model = avail[0]
                    logger.info(f"Falling back to available Groq model: {fallback_model}")
                    response = await client.chat.completions.create(
                        model=fallback_model,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": f"{system_prompt}\nReturn JSON strictly."},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                    )
                    content = response.choices[0].message.content or "{}"
                    return json.loads(content)
            except Exception as inner_e:
                logger.error(f"Groq auto-fallback failed: {inner_e}")
            
            # Fallback to mock so user always gets an actionable response
            logger.warning("Groq unavailable. Falling back to heuristic mock extractor.")
            return await self._call_mock(system_prompt, user_prompt)

    async def _call_openai(
        self, system_prompt: str, user_prompt: str, model: str | None = None
    ) -> dict[str, Any]:
        """Call OpenAI chat completions API."""
        api_key = self.settings.openai_api_key
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. Falling back to mock extractor.")
            return await self._call_mock(system_prompt, user_prompt)

        client = AsyncOpenAI(api_key=api_key)
        target_model = model or self.settings.openai_model

        response = await client.chat.completions.create(
            model=target_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    async def _call_grok(
        self, system_prompt: str, user_prompt: str, model: str | None = None
    ) -> dict[str, Any]:
        """Call Grok (xAI) using its OpenAI-compatible endpoint."""
        api_key = self.settings.effective_grok_api_key
        if not api_key:
            logger.warning("GROK_API_KEY / XAI_API_KEY is not set. Falling back to mock extractor.")
            return await self._call_mock(system_prompt, user_prompt)

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.grok_base_url,
        )
        target_model = model or self.settings.grok_model

        response = await client.chat.completions.create(
            model=target_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    async def _call_anthropic(
        self, system_prompt: str, user_prompt: str, model: str | None = None
    ) -> dict[str, Any]:
        """Call Anthropic Claude API via direct async HTTP request."""
        api_key = self.settings.anthropic_api_key
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY is not set. Falling back to mock extractor.")
            return await self._call_mock(system_prompt, user_prompt)

        target_model = model or self.settings.anthropic_model
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": target_model,
            "max_tokens": 4096,
            "system": f"{system_prompt}\nCRITICAL: Respond ONLY with valid, unescaped JSON object.",
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["content"][0]["text"]

            # Clean any accidental markdown backticks
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
            return json.loads(cleaned)

    async def _call_mock(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Heuristic offline decision extractor for zero-token testing."""
        # Simple extraction heuristics based on keywords in user prompt
        text = user_prompt.lower()

        category = "architecture"
        if "migrate" in text or "switch" in text or "replace" in text or "adopt" in text:
            category = "technology_selection"
        elif "refactor" in text or "clean" in text or "split" in text:
            category = "refactoring"
        elif "deprecat" in text or "remove" in text:
            category = "deprecation"
        elif "pattern" in text or "cqrs" in text or "repository" in text:
            category = "pattern_adoption"

        title = "Architectural decision extracted from pull request changes"
        first_line = user_prompt.strip().split("\n")[0]
        if first_line:
            title = first_line[:120].strip("#- ")

        return {
            "decisions": [
                {
                    "title": title or "Engineering Pattern & Architecture Update",
                    "summary": (
                        "Adopted async patterns and structural optimizations for the core codebase "
                        "to prevent event loop blocking under heavy request load."
                    ),
                    "rationale": "Event loop latency was identified as a blocker for real-time updates.",
                    "alternatives": "Considered maintaining synchronous connection threadpools, but discarded due to high overhead.",
                    "chosen_approach": "Adoption of AsyncSession alongside declarative base models.",
                    "category": category,
                    "confidence": 0.92,
                    "confidence_score": 0.92,
                    "evidence": [
                        {
                            "quote": first_line[:120] if first_line else "Migrate database layer to SQLAlchemy async",
                            "source_description": "PR Description",
                        }
                    ],
                    "relevant_files": ["backend/app/database.py", "backend/app/config.py"],
                    "governed_files": ["backend/app/database.py", "backend/app/config.py"],
                }
            ]
        }

    def get_provider_status(self) -> dict[str, Any]:
        """Return which providers have configured API keys."""
        return {
            "active_provider": self.settings.llm_provider,
            "providers": {
                "mock": {
                    "available": True,
                    "description": "Offline heuristic extractor (No token required)",
                },
                "groq": {
                    "available": bool(self.settings.groq_api_key),
                    "model": self.settings.groq_model,
                    "endpoint": self.settings.groq_base_url,
                },
                "openai": {
                    "available": bool(self.settings.openai_api_key),
                    "model": self.settings.openai_model,
                },
                "grok": {
                    "available": bool(self.settings.effective_grok_api_key),
                    "model": self.settings.grok_model,
                    "endpoint": self.settings.grok_base_url,
                },
                "anthropic": {
                    "available": bool(self.settings.anthropic_api_key),
                    "model": self.settings.anthropic_model,
                },
            },
        }
