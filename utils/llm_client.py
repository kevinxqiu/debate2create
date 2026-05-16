import os
from types import SimpleNamespace


class _OpenAIWrapper:
    """OpenAI wrapper that returns an OpenAI-chat-compatible response object."""

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI  # import lazily

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI-backed LLM calls. "
                "Set OPENAI_API_KEY or choose a different provider with LLM_PROVIDER."
            )
        self._client = OpenAI(api_key=key)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature=1.0, n=1,
                reasoning_effort="high", verbosity="low", **kwargs):
        """
        Use responses.create() endpoint which supports reasoning and verbosity.

        Args:
            model: Model name (e.g., "gpt-5")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            n: Number of completions to generate
            reasoning_effort: "none", "low", "medium", "high", or "xhigh"
            verbosity: "low", "medium", or "high"
        """
        # Build choices list (responses.create doesn't support n, so we call multiple times)
        choices = []
        for _ in range(max(int(n), 1)):
            try:
                effort = str(reasoning_effort or "").strip().lower()
                if effort == "minimal":
                    effort = "low"

                request = {
                    "model": model,
                    "input": messages,
                    "text": {"verbosity": verbosity},
                    "temperature": temperature,
                }
                if effort and effort not in {"off", "false"}:
                    request["reasoning"] = {"effort": effort}

                response = self._client.responses.create(
                    **request,
                )
                # Extract text from responses.create output
                # The response has output_text or output[0].content[0].text
                text = getattr(response, "output_text", None)
                if text is None and hasattr(response, "output") and response.output:
                    try:
                        # Try to extract from output structure
                        for item in response.output:
                            if hasattr(item, "content") and item.content:
                                for content_item in item.content:
                                    if hasattr(content_item, "text"):
                                        text = content_item.text
                                        break
                            if text:
                                break
                    except Exception:
                        text = ""
                choices.append(SimpleNamespace(message=SimpleNamespace(content=text or "")))
            except Exception:
                # Fallback to chat.completions if responses.create fails
                # (e.g., for models that don't support it)
                fallback_response = self._client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, n=1
                )
                choices.append(fallback_response.choices[0])

        # Return OpenAI-like response object for compatibility
        return SimpleNamespace(choices=choices, usage=None)


def _resolve_gemini_api_key(api_key: str | None = None) -> str:
    if api_key:
        return api_key

    gemini_key = os.environ.get("GEMINI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key and google_key and gemini_key != google_key:
        raise RuntimeError(
            "Both GEMINI_API_KEY and GOOGLE_API_KEY are set with different values; "
            "set only one for Gemini provider."
        )

    key = gemini_key or google_key
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set for Gemini provider "
            "(GOOGLE_API_KEY is also accepted as a compatibility alias)"
        )
    return key


class _GeminiWrapper:
    def __init__(self, api_key: str | None = None):
        import google.generativeai as genai  # import lazily

        key = _resolve_gemini_api_key(api_key)
        genai.configure(api_key=key)
        self._genai = genai
        # Provide OpenAI-like attribute access: chat.completions.create(...)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _extract_system_and_user(self, messages):
        system_parts = []
        non_system_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(str(content))
            else:
                non_system_messages.append((role, str(content)))
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        # Flatten to a single prompt string with role tags to preserve intent
        prompt_lines = []
        for role, content in non_system_messages:
            prompt_lines.append(f"{role.upper()}:\n{content}\n")
        prompt = "\n".join(prompt_lines).strip()
        return system_instruction, prompt

    def _create(self, model, messages, temperature=1.0, n=1, **kwargs):
        # Build GenerativeModel with optional system instruction
        system_instruction, prompt = self._extract_system_and_user(messages)
        if system_instruction:
            gmodel = self._genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
        else:
            gmodel = self._genai.GenerativeModel(model_name=model)

        generation_config = {"temperature": float(temperature)}

        choices = []
        for _ in range(max(int(n), 1)):
            resp = gmodel.generate_content(prompt, generation_config=generation_config)
            text = getattr(resp, "text", None)
            if text is None and hasattr(resp, "candidates") and resp.candidates:
                # Fallback extraction
                try:
                    text = resp.candidates[0].content.parts[0].text
                except Exception:
                    text = ""
            choices.append(SimpleNamespace(message=SimpleNamespace(content=text or "")))

        # Return an OpenAI-like response object
        return SimpleNamespace(choices=choices, usage=None)


def get_llm_client(provider: str | None = None, api_key: str | None = None):
    """
    Returns a client exposing OpenAI-compatible interface:
      client.chat.completions.create(model=..., messages=[...], temperature=..., n=...)

    Provider selection order:
    - Explicit provider arg ('openai' or 'gemini')
    - Env var LLM_PROVIDER, when provider is not supplied
    - Model name prefix startswith 'gemini-' -> Gemini, else OpenAI
    """
    p = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if p in ("gemini", "google", "googleai"):  # force Gemini
        return _GeminiWrapper(api_key=api_key)
    if p in ("openai", "oai"):  # force OpenAI
        return _OpenAIWrapper(api_key=api_key)

    # Heuristic fallback by model name if provided via env default
    model = os.environ.get("LLM_MODEL", "")
    if model.startswith("gemini-"):
        return _GeminiWrapper(api_key=api_key)

    # Default to OpenAI
    return _OpenAIWrapper(api_key=api_key)
