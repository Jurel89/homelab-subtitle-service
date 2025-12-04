# src/homelab_subs/core/translation.py

"""
Translation module for subtitle files.

Provides translation capabilities using:
- Helsinki-NLP MarianMT models (fast, good for European languages)
- NLLB-200 (Facebook's No Language Left Behind - supports 200+ languages)

The module can translate SRT subtitle files from one language to another,
preserving timing information and subtitle structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

# Translation backends
TranslationBackend = Literal["helsinki", "nllb"]

# Common language codes mapping for Helsinki-NLP
# Helsinki models use format: Helsinki-NLP/opus-mt-{src}-{tgt}
HELSINKI_LANGUAGE_PAIRS: dict[str, list[str]] = {
    "en": ["es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ar", "hi", "pl", "sv", "da", "fi", "no"],
    "es": ["en", "fr", "de", "it", "pt"],
    "fr": ["en", "es", "de", "it"],
    "de": ["en", "es", "fr", "it"],
    "it": ["en", "es", "fr", "de"],
    "pt": ["en", "es"],
    "ru": ["en"],
    "zh": ["en"],
    "ja": ["en"],
    "ar": ["en"],
}

# NLLB-200 language codes (Flores-200 format)
# See: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANGUAGE_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "pl": "pol_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "cs": "ces_Latn",
    "hu": "hun_Latn",
    "ro": "ron_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "he": "heb_Hebr",
    "el": "ell_Grek",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "hr": "hrv_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "et": "est_Latn",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
}

# Reverse mapping for NLLB codes
NLLB_CODE_TO_LANG: dict[str, str] = {v: k for k, v in NLLB_LANGUAGE_CODES.items()}


@dataclass
class SubtitleEntry:
    """
    Represents a single subtitle entry from an SRT file.
    """
    index: int
    start_time: str
    end_time: str
    text: str


@dataclass
class TranslatorConfig:
    """
    Configuration for the subtitle translator.

    Attributes
    ----------
    backend:
        Translation backend to use: "helsinki" for MarianMT, "nllb" for NLLB-200.
    model_name:
        Specific model name override. If None, auto-selects based on language pair.
        For NLLB: "facebook/nllb-200-distilled-600M" (fast) or "facebook/nllb-200-3.3B" (best quality)
    device:
        Device to run on: "cpu" or "cuda".
    max_length:
        Maximum token length for translation output.
    batch_size:
        Number of subtitle lines to translate in one batch.
    """
    backend: TranslationBackend = "nllb"
    model_name: Optional[str] = None
    device: str = "cpu"
    max_length: int = 512
    batch_size: int = 8


class Translator:
    """
    High-level translator for subtitle files.

    Supports two backends:
    - Helsinki-NLP MarianMT: Fast, good for common language pairs
    - NLLB-200: Supports 200+ languages, higher quality but slower

    Example
    -------
    >>> translator = Translator(TranslatorConfig(backend="nllb"))
    >>> translator.translate_srt_file(
    ...     Path("movie.en.srt"),
    ...     Path("movie.es.srt"),
    ...     source_lang="en",
    ...     target_lang="es",
    ... )
    """

    def __init__(self, config: Optional[TranslatorConfig] = None) -> None:
        self.config = config or TranslatorConfig()
        self._model = None
        self._tokenizer = None
        self._current_model_key: Optional[str] = None

    # ---------- Internal helpers ----------

    def _get_helsinki_model_name(self, source_lang: str, target_lang: str) -> str:
        """Get the Helsinki-NLP model name for a language pair."""
        return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    def _get_nllb_model_name(self) -> str:
        """Get the NLLB model name."""
        if self.config.model_name:
            return self.config.model_name
        # Default to distilled 600M model for balance of speed and quality
        return "facebook/nllb-200-distilled-600M"

    def _ensure_model_loaded(self, source_lang: str, target_lang: str) -> None:
        """
        Lazily load the translation model on first use.
        Reloads if the language pair changes (for Helsinki backend).
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Translation requires the 'transformers' package. "
                "Install with: pip install transformers sentencepiece"
            ) from e

        if self.config.backend == "helsinki":
            model_name = self._get_helsinki_model_name(source_lang, target_lang)
            model_key = f"helsinki:{model_name}"
        else:  # nllb
            model_name = self._get_nllb_model_name()
            model_key = f"nllb:{model_name}"

        # Skip if same model is already loaded
        if self._current_model_key == model_key and self._model is not None:
            return

        logger.info(
            f"Loading translation model: {model_name} "
            f"(backend={self.config.backend}, device={self.config.device})"
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            if self.config.device == "cuda":
                self._model = self._model.cuda()

            self._current_model_key = model_key
            logger.info("Translation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise RuntimeError(f"Failed to load translation model '{model_name}': {e}") from e

    def _translate_batch_helsinki(self, texts: list[str]) -> list[str]:
        """Translate a batch of texts using Helsinki-NLP MarianMT."""
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )

        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_length=self.config.max_length,
        )

        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def _translate_batch_nllb(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate a batch of texts using NLLB-200."""
        # Get NLLB language codes
        src_code = NLLB_LANGUAGE_CODES.get(source_lang)
        tgt_code = NLLB_LANGUAGE_CODES.get(target_lang)

        if not src_code:
            raise ValueError(
                f"Source language '{source_lang}' not supported by NLLB. "
                f"Supported: {list(NLLB_LANGUAGE_CODES.keys())}"
            )
        if not tgt_code:
            raise ValueError(
                f"Target language '{target_lang}' not supported by NLLB. "
                f"Supported: {list(NLLB_LANGUAGE_CODES.keys())}"
            )

        # Set source language for tokenizer
        self._tokenizer.src_lang = src_code

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )

        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate with forced target language token
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

        outputs = self._model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=self.config.max_length,
        )

        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # ---------- SRT Parsing ----------

    def _parse_srt(self, content: str) -> list[SubtitleEntry]:
        """
        Parse SRT content into a list of SubtitleEntry objects.
        """
        entries: list[SubtitleEntry] = []

        # Split by double newlines (subtitle blocks)
        # Handle both \r\n and \n line endings
        content = content.replace("\r\n", "\n")
        blocks = re.split(r"\n\n+", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            try:
                index = int(lines[0].strip())
            except ValueError:
                logger.warning(f"Skipping invalid subtitle index: {lines[0]}")
                continue

            # Parse timestamp line
            timestamp_match = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
                lines[1].strip(),
            )
            if not timestamp_match:
                logger.warning(f"Skipping invalid timestamp: {lines[1]}")
                continue

            start_time = timestamp_match.group(1)
            end_time = timestamp_match.group(2)

            # Remaining lines are the subtitle text
            text = "\n".join(lines[2:]).strip()

            entries.append(SubtitleEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
            ))

        return entries

    def _entries_to_srt(self, entries: list[SubtitleEntry]) -> str:
        """
        Convert SubtitleEntry objects back to SRT format.
        """
        blocks: list[str] = []

        for entry in entries:
            block = f"{entry.index}\n{entry.start_time} --> {entry.end_time}\n{entry.text}\n"
            blocks.append(block)

        return "\n".join(blocks) + "\n"

    # ---------- Public API ----------

    def get_supported_languages(self, backend: Optional[TranslationBackend] = None) -> list[str]:
        """
        Get list of supported language codes for a backend.

        Parameters
        ----------
        backend : Optional[TranslationBackend]
            Backend to query. If None, uses configured backend.

        Returns
        -------
        list[str]
            List of ISO 639-1 language codes.
        """
        backend = backend or self.config.backend

        if backend == "helsinki":
            # Return all source languages that have at least one target
            return list(HELSINKI_LANGUAGE_PAIRS.keys())
        else:
            return list(NLLB_LANGUAGE_CODES.keys())

    def is_language_pair_supported(
        self, source_lang: str, target_lang: str, backend: Optional[TranslationBackend] = None
    ) -> bool:
        """
        Check if a language pair is supported.

        Parameters
        ----------
        source_lang : str
            Source language code (ISO 639-1).
        target_lang : str
            Target language code (ISO 639-1).
        backend : Optional[TranslationBackend]
            Backend to check. If None, uses configured backend.

        Returns
        -------
        bool
            True if the pair is supported.
        """
        backend = backend or self.config.backend

        if backend == "helsinki":
            targets = HELSINKI_LANGUAGE_PAIRS.get(source_lang, [])
            return target_lang in targets
        else:
            return source_lang in NLLB_LANGUAGE_CODES and target_lang in NLLB_LANGUAGE_CODES

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate a single text string.

        Parameters
        ----------
        text : str
            Text to translate.
        source_lang : str
            Source language code (ISO 639-1, e.g., "en").
        target_lang : str
            Target language code (ISO 639-1, e.g., "es").

        Returns
        -------
        str
            Translated text.
        """
        if source_lang == target_lang:
            return text

        self._ensure_model_loaded(source_lang, target_lang)

        if self.config.backend == "helsinki":
            results = self._translate_batch_helsinki([text])
        else:
            results = self._translate_batch_nllb([text], source_lang, target_lang)

        return results[0] if results else text

    def translate_texts(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[float, int], None]] = None,
    ) -> list[str]:
        """
        Translate multiple text strings with batching.

        Parameters
        ----------
        texts : list[str]
            List of texts to translate.
        source_lang : str
            Source language code.
        target_lang : str
            Target language code.
        progress_callback : Optional[Callable[[float, int], None]]
            Callback for progress updates (percentage, items processed).

        Returns
        -------
        list[str]
            List of translated texts in the same order.
        """
        if source_lang == target_lang:
            return texts

        self._ensure_model_loaded(source_lang, target_lang)

        results: list[str] = []
        total = len(texts)
        batch_size = self.config.batch_size

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]

            if self.config.backend == "helsinki":
                translated = self._translate_batch_helsinki(batch)
            else:
                translated = self._translate_batch_nllb(batch, source_lang, target_lang)

            results.extend(translated)

            if progress_callback:
                progress = min(100.0, (i + len(batch)) / total * 100)
                progress_callback(progress, len(results))

        return results

    def translate_srt_file(
        self,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        encoding: str = "utf-8",
        progress_callback: Optional[Callable[[float, int], None]] = None,
    ) -> Path:
        """
        Translate an SRT subtitle file to another language.

        Parameters
        ----------
        input_path : Path
            Path to the source SRT file.
        output_path : Path
            Path for the translated SRT file.
        source_lang : str
            Source language code (ISO 639-1).
        target_lang : str
            Target language code (ISO 639-1).
        encoding : str
            File encoding (default: utf-8).
        progress_callback : Optional[Callable[[float, int], None]]
            Callback for progress updates.

        Returns
        -------
        Path
            Path to the created translated SRT file.

        Raises
        ------
        FileNotFoundError
            If input file doesn't exist.
        ValueError
            If language pair is not supported.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"Input SRT file not found: {input_path}")

        if not self.is_language_pair_supported(source_lang, target_lang):
            raise ValueError(
                f"Language pair '{source_lang}' -> '{target_lang}' not supported "
                f"with backend '{self.config.backend}'"
            )

        logger.info(
            f"Translating {input_path.name} from '{source_lang}' to '{target_lang}' "
            f"(backend={self.config.backend})"
        )

        # Read and parse SRT
        content = input_path.read_text(encoding=encoding)
        entries = self._parse_srt(content)

        if not entries:
            logger.warning(f"No subtitle entries found in {input_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding=encoding)
            return output_path

        logger.info(f"Found {len(entries)} subtitle entries to translate")

        # Extract texts for translation
        texts = [entry.text for entry in entries]

        # Translate
        translated_texts = self.translate_texts(
            texts,
            source_lang,
            target_lang,
            progress_callback=progress_callback,
        )

        # Update entries with translations
        for entry, translated_text in zip(entries, translated_texts):
            entry.text = translated_text

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        srt_content = self._entries_to_srt(entries)
        output_path.write_text(srt_content, encoding=encoding)

        logger.info(f"Translated subtitle saved to {output_path}")

        return output_path


def get_available_backends() -> list[TranslationBackend]:
    """
    Get list of available translation backends.

    Returns list of backends that have their dependencies installed.
    """
    backends: list[TranslationBackend] = []

    try:
        import transformers  # noqa: F401
        # Both backends use transformers
        backends.extend(["helsinki", "nllb"])
    except ImportError:
        pass

    return backends


def list_supported_languages(
    backend: TranslationBackend = "nllb",
) -> dict[str, str]:
    """
    Get a dictionary of supported languages for a backend.

    Parameters
    ----------
    backend : TranslationBackend
        The translation backend to query.

    Returns
    -------
    dict[str, str]
        Dictionary mapping language codes to language names.
    """
    # Language names for common codes
    LANGUAGE_NAMES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "pl": "Polish",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "no": "Norwegian",
        "cs": "Czech",
        "hu": "Hungarian",
        "ro": "Romanian",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "vi": "Vietnamese",
        "th": "Thai",
        "id": "Indonesian",
        "ms": "Malay",
        "he": "Hebrew",
        "el": "Greek",
        "bg": "Bulgarian",
        "ca": "Catalan",
        "hr": "Croatian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "et": "Estonian",
        "lv": "Latvian",
        "lt": "Lithuanian",
    }

    if backend == "helsinki":
        codes = list(HELSINKI_LANGUAGE_PAIRS.keys())
    else:
        codes = list(NLLB_LANGUAGE_CODES.keys())

    return {code: LANGUAGE_NAMES.get(code, code) for code in codes}
