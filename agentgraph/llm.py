import os
import requests


class GeminiLLM:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env"
            )

    def generate(self, system_prompt, user_prompt):
        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [{
                "parts": [{"text": user_prompt}]
            }],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 512,
            },
        }

        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=90,
        )

        response.raise_for_status()

        data = response.json()
        candidates = data.get("candidates", [])

        if not candidates:
            raise RuntimeError(
                "Gemini returned no candidates."
            )

        parts = candidates[0].get(
            "content", {}
        ).get("parts", [])

        answer = "".join(
            part.get("text", "")
            for part in parts
        ).strip()

        if not answer:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return answer
