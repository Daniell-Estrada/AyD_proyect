"""
LLM Service layer for interacting with various Large Language Model providers.
Supports Anthropic Claude, OpenAI GPT, GitHub Models, Google Gemini and Mistral with
automatic fallback and usage metrics tracking.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from anthropic import Anthropic, AnthropicError
from google import genai
from google.genai.types import Content, Part
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from mistralai import Mistral
from openai import OpenAI, OpenAIError

from app.shared.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMUsageMetrics:
    """Metrics for tracking LLM usage and costs."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float
    estimated_cost_usd: float


class LLMService:
    """
    Service for interacting with Large Language Models.
    Supports Anthropic Claude and OpenAI GPT models with automatic fallback.
    """

    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
        "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4.1": {"input": 10.0, "output": 30.0},
        "openai/gpt-4.1": {"input": 10.0, "output": 30.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "anthropic/claude-haiku-4-5": {"input": 1.0, "output": 5.0},
        "mistral-small-latest": {"input": 2.0, "output": 6.0},
        "mistral-large-latest": {"input": 4.0, "output": 12.0},
    }

    def __init__(self):
        self.primary_provider = settings.primary_llm_provider
        self.primary_model = settings.primary_llm_model
        self.fallback_provider = settings.fallback_llm_provider
        self.fallback_model = settings.fallback_llm_model

        self._init_clients()

    def _init_clients(self):
        """Initialize LLM provider clients."""
        if settings.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
            self.anthropic_langchain = ChatAnthropic(
                model=(
                    self.primary_model
                    if self.primary_provider == "anthropic"
                    else self.fallback_model
                ),
                anthropic_api_key=settings.anthropic_api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

        if settings.openai_api_key:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.openai_langchain = ChatOpenAI(
                model=(
                    self.primary_model
                    if self.primary_provider == "openai"
                    else self.fallback_model
                ),
                openai_api_key=settings.openai_api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

        if settings.github_token:
            self.github_client = OpenAI(
                base_url=settings.github_endpoint,
                api_key=settings.github_token,
            )

        gemini_api_key = getattr(settings, "gemini_api_key", None)
        if gemini_api_key:
            self.google_client = genai.Client(api_key=gemini_api_key)

        if getattr(settings, "mistral_api_key", None):
            self.mistral_client = Mistral(api_key=settings.mistral_api_key)

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        use_fallback: bool = True,
        max_retries: int = 3,
    ) -> tuple[str, LLMUsageMetrics]:
        """
        Invoke LLM with given prompts, handling retries and fallback.
        """
        provider = self.fallback_provider if use_fallback else self.primary_provider
        model = self.fallback_model if use_fallback else self.primary_model

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                if provider == "anthropic":
                    response_text, usage = self._invoke_anthropic(
                        system_prompt, user_prompt, model
                    )
                elif provider == "github":
                    response_text, usage = self._invoke_github(
                        system_prompt, user_prompt, model
                    )
                elif provider == "openai":
                    response_text, usage = self._invoke_openai(
                        system_prompt, user_prompt, model
                    )
                elif provider == "google":
                    response_text, usage = self._invoke_google(
                        system_prompt, user_prompt, model
                    )
                elif provider == "mistral":
                    response_text, usage = self._invoke_mistral(
                        system_prompt, user_prompt, model
                    )
                else:
                    raise ValueError(f"Unsupported LLM provider '{provider}'")

                duration_ms = (time.time() - start_time) * 1000

                cost = self._calculate_cost(
                    model, usage["prompt_tokens"], usage["completion_tokens"]
                )

                metrics = LLMUsageMetrics(
                    provider=provider,
                    model=model,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    duration_ms=duration_ms,
                    estimated_cost_usd=cost,
                )

                logger.info(
                    f"LLM call succeeded: {provider}/{model}, "
                    f"tokens={metrics.total_tokens}, "
                    f"cost=${metrics.estimated_cost_usd:.4f}, "
                    f"duration={metrics.duration_ms:.0f}ms"
                )

                return response_text, metrics

            except (AnthropicError, OpenAIError) as e:
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}"
                )

                if attempt == max_retries - 1:
                    if not use_fallback:
                        logger.info("Attempting fallback provider...")
                        return self.invoke(
                            system_prompt, user_prompt, use_fallback=True, max_retries=2
                        )
                    else:
                        raise Exception(f"All LLM providers failed: {e}")

                time.sleep(2**attempt)

    def _invoke_anthropic(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, Dict[str, int]]:
        """Invoke Anthropic Claude API."""
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return response.content[0].text, usage

    def _invoke_openai(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, Dict[str, int]]:
        """Invoke OpenAI GPT API."""
        response = self.openai_client.chat.completions.create(
            model=model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return response.choices[0].message.content, usage

    def _invoke_github(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, Dict[str, int]]:
        """Invoke GitHub Models endpoint (OpenAI-compatible)."""
        if not hasattr(self, "github_client"):
            raise OpenAIError("GitHub client is not configured")

        response = self.github_client.chat.completions.create(
            model=model or settings.github_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return response.choices[0].message.content, usage

    def _invoke_google(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, Dict[str, int]]:
        """Invoke Google Gemini API with proper Content/Part payloads."""
        if not hasattr(self, "google_client"):
            raise ValueError("Google Generative AI client is not configured")

        response = self.google_client.models.generate_content(
            model=model or settings.google_model,
            contents=[
                Content(parts=[Part(text=system_prompt)], role="model"),
                Content(parts=[Part(text=user_prompt)], role="user"),
            ],
        )

        text_response = getattr(response, "text", None) or "".join(
            part.text
            for candidate in getattr(response, "candidates", [])
            for part in getattr(candidate, "content", []).parts
            if part.text
        )

        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0),
            "completion_tokens": getattr(usage_meta, "candidates_token_count", 0),
            "total_tokens": getattr(usage_meta, "total_token_count", 0),
        }

        return text_response, usage

    def _invoke_mistral(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> tuple[str, Dict[str, int]]:
        """Invoke Mistral's chat completion API."""
        if not hasattr(self, "mistral_client"):
            raise ValueError("Mistral client is not configured")

        response = self.mistral_client.chat.complete(
            model=model or "mistral-small-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            stream=False,
        )

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        message_content = getattr(message, "content", "") if message else ""

        if isinstance(message_content, str):
            text_response = message_content
        elif isinstance(message_content, list):
            text_chunks = []
            for chunk in message_content:
                text = getattr(chunk, "text", None)
                if text is None and isinstance(chunk, dict):
                    text = chunk.get("text")
                if text:
                    text_chunks.append(text)
            text_response = "".join(text_chunks)
        else:
            text_response = str(message_content or "")

        usage_info = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": (
                getattr(usage_info, "prompt_tokens", 0) if usage_info else 0
            ),
            "completion_tokens": (
                getattr(usage_info, "completion_tokens", 0) if usage_info else 0
            ),
            "total_tokens": getattr(usage_info, "total_tokens", 0) if usage_info else 0,
        }

        return text_response, usage

    def invoke_langchain(
        self,
        messages: List[Any],
        use_fallback: bool = True,
    ) -> tuple[str, LLMUsageMetrics]:
        """
        Invoke LLM via LangChain interface, returning response and usage metrics.
        """
        provider = self.fallback_provider if use_fallback else self.primary_provider
        model = self.fallback_model if use_fallback else self.primary_model

        start_time = time.time()

        if provider == "anthropic":
            llm = self.anthropic_langchain
        elif provider == "openai":
            llm = self.openai_langchain
        else:
            raise ValueError(
                "LangChain interface is not available for the GitHub provider."
            )

        response = llm.invoke(messages)
        duration_ms = (time.time() - start_time) * 1000

        usage_data = getattr(response, "response_metadata", {}).get("usage", {})

        if provider == "anthropic":
            prompt_tokens = usage_data.get("input_tokens", 0)
            completion_tokens = usage_data.get("output_tokens", 0)
        else:
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)

        total_tokens = prompt_tokens + completion_tokens
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        metrics = LLMUsageMetrics(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            estimated_cost_usd=cost,
        )

        return response.content, metrics

    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate estimated cost in USD."""
        if model not in self.PRICING:
            logger.warning(f"No pricing data for model {model}, using default")
            return 0.0

        pricing = self.PRICING[model]
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by provider."""
        return {
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-haiku-4-5",
            ],
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo-preview",
                "gpt-4",
            ],
            "github": [
                "openai/gpt-4.1",
                "openai/gpt-4o-mini",
                "anthropic/claude-haiku-4-5",
            ],
            "google": [
                settings.google_model,
            ],
            "mistral": [
                "mistral-small-latest",
                "mistral-large-latest",
            ],
        }
