import os
import requests


class GeminiLLM:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv(
            "GEMINI_MODEL", "gemini-2.5-flash"
        ).strip()

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from the environment."
            )

    def generate(self, system_prompt, user_prompt):
        url = (
            f"{self.BASE_URL}/models/"
            f"{self.model}:generateContent"
        )

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 1024,
            },
        }

        try:
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90,
            )

        except requests.RequestException:
            raise RuntimeError(
                "Could not connect to the Gemini API."
            ) from None

        if not response.ok:
            raise RuntimeError(
                f"Gemini API request failed "
                f"(HTTP {response.status_code}). "
                "Check model availability, API access, and quota."
            )

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                "Gemini returned an invalid JSON response."
            ) from None

        candidates = data.get("candidates", [])

        if not candidates:
            raise RuntimeError(
                "Gemini returned no candidates. "
                "The response may have been blocked."
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