"""Privacy utilities: PII scrubbing and style masking."""

from __future__ import annotations

from typing import Any, Callable, Iterable


SHIELD_SYSTEM_PROMPT = (
    "You are a strict privacy sanitization engine.\n"
    "The user has provided text that already had its hard PII redacted and replaced "
    "with tags like <PERSON_1> or <EMAIL_ADDRESS>.\n"
    "Your only job is to rewrite the text to mask the user's authorship style.\n"
    "Rules:\n"
    "1. Flatten the tone into a dry, objective, and generic academic format.\n"
    "2. Remove colloquialisms, unique punctuation habits, and emotional language.\n"
    "3. Obfuscate proprietary context or hyper-specific technical details into generic equivalents.\n"
    "4. Keep Presidio placeholder tags (like <PERSON_1>) intact.\n"
    "5. Do not answer the prompt.\n"
    "6. Only output the rewritten text."
)


class PrivacyShield:
    def __init__(
        self,
        fetch_models: Callable[[], list[str] | None],
        openai_stream: Callable[[list[dict], str, bool], Iterable[str]],
    ) -> None:
        self._fetch_models = fetch_models
        self._openai_stream = openai_stream
        self._analyzer: Any = None
        self._anonymizer: Any = None

    def scrub_pii(self, raw_text: str) -> tuple[str, str]:
        if not raw_text or not raw_text.strip():
            return "", "Nothing to scrub."

        try:
            analyzer, anonymizer = self._get_presidio()
        except Exception as exc:
            return raw_text, (
                f"**Error loading Presidio:** {exc}\n\n"
                "Install spaCy model: `python -m spacy download en_core_web_lg`"
            )

        results = analyzer.analyze(text=raw_text, entities=[], language="en")
        anonymized = anonymizer.anonymize(text=raw_text, analyzer_results=results)

        if not results:
            return anonymized.text, "No PII entities detected. Text appears clean."

        counts: dict[str, int] = {}
        for result in results:
            counts[result.entity_type] = counts.get(result.entity_type, 0) + 1

        lines = [
            f"- **{entity}**: {count} found" for entity, count in sorted(counts.items())
        ]
        summary = (
            f"**{len(results)} PII entities detected and replaced:**\n"
            + "\n".join(lines)
        )
        return anonymized.text, summary

    def restyle_text(self, scrubbed_text: str):
        if not scrubbed_text or not scrubbed_text.strip():
            yield "Nothing to restyle."
            return

        if self._fetch_models() is None:
            yield "**Server offline** - please go to the Server tab and load a model."
            return

        full = ""
        try:
            messages = [
                {"role": "system", "content": SHIELD_SYSTEM_PROMPT},
                {"role": "user", "content": scrubbed_text},
            ]
            for token in self._openai_stream(messages, "", False):
                full += token
                yield full
        except Exception as exc:
            full += f"\n\n*[Error: {exc}]*"
        yield full

    def _get_presidio(self) -> tuple[Any, Any]:
        if self._analyzer is None or self._anonymizer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()

        return self._analyzer, self._anonymizer
