"""Unified chat() across OpenAI, Anthropic, and Grok (xAI)."""
from openai import OpenAI
import anthropic


def chat(provider: str, messages: list, keys: dict) -> str:
    """messages is a list of {"role": "system"|"user"|"assistant", "content": str}."""
    if provider == "openai":
        key = keys.get("openai")
        if not key:
            return "ERROR: No OpenAI API key provided in Settings (⚙️)."
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return resp.choices[0].message.content

    if provider == "grok":
        key = keys.get("xai")
        if not key:
            return "ERROR: No Grok (xAI) API key provided in Settings (⚙️)."
        client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        grok_models = [
            "grok-4.6",
            "grok-4.5",
            "grok-4.3",
            "grok-4.20-0309-non-reasoning",
            "grok-2",
            "grok-2-1212",
            "grok-beta",
            "grok-2-latest",
        ]
        errors = []
        for m in grok_models:
            try:
                resp = client.chat.completions.create(model=m, messages=messages)
                return resp.choices[0].message.content
            except Exception as e:
                errors.append(str(e))
                continue
        return f"ERROR: Grok API request failed. Please check your xAI API key in Settings (⚙️).\nDetails: {errors[0] if errors else 'Unknown error'}"

    if provider == "anthropic":
        key = keys.get("anthropic")
        if not key:
            return "ERROR: No Anthropic API key provided in Settings (⚙️)."
        client = anthropic.Anthropic(api_key=key)
        # Anthropic takes the system prompt separately from the message list.
        system = ""
        convo = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        
        # Candidate model names in priority order
        model_candidates = ["claude-sonnet-4-6", "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"]
        last_error = None
        for model_name in model_candidates:
            try:
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    system=system,
                    messages=convo,
                )
                return resp.content[0].text
            except anthropic.NotFoundError as e:
                last_error = e
                continue
        if last_error:
            raise last_error

    return "ERROR: Unknown provider."
