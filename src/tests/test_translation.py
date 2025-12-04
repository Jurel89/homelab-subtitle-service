# src/tests/test_translation.py

"""
Tests for the translation module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from homelab_subs.core.translation import (
    SubtitleEntry,
    Translator,
    TranslatorConfig,
    list_supported_languages,
    get_available_backends,
    NLLB_LANGUAGE_CODES,
    HELSINKI_LANGUAGE_PAIRS,
)


class TestSubtitleEntry:
    """Tests for SubtitleEntry dataclass."""

    def test_create_entry(self):
        entry = SubtitleEntry(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:03,500",
            text="Hello world",
        )
        assert entry.index == 1
        assert entry.start_time == "00:00:01,000"
        assert entry.end_time == "00:00:03,500"
        assert entry.text == "Hello world"


class TestTranslatorConfig:
    """Tests for TranslatorConfig dataclass."""

    def test_default_config(self):
        config = TranslatorConfig()
        assert config.backend == "nllb"
        assert config.model_name is None
        assert config.device == "cpu"
        assert config.max_length == 512
        assert config.batch_size == 8

    def test_custom_config(self):
        config = TranslatorConfig(
            backend="helsinki",
            model_name="custom-model",
            device="cuda",
            max_length=256,
            batch_size=16,
        )
        assert config.backend == "helsinki"
        assert config.model_name == "custom-model"
        assert config.device == "cuda"
        assert config.max_length == 256
        assert config.batch_size == 16


class TestTranslator:
    """Tests for Translator class."""

    def test_init_default_config(self):
        translator = Translator()
        assert translator.config.backend == "nllb"
        assert translator._model is None
        assert translator._tokenizer is None

    def test_init_custom_config(self):
        config = TranslatorConfig(backend="helsinki", device="cuda")
        translator = Translator(config=config)
        assert translator.config.backend == "helsinki"
        assert translator.config.device == "cuda"

    def test_get_helsinki_model_name(self):
        translator = Translator()
        assert translator._get_helsinki_model_name("en", "es") == "Helsinki-NLP/opus-mt-en-es"
        assert translator._get_helsinki_model_name("fr", "de") == "Helsinki-NLP/opus-mt-fr-de"

    def test_get_nllb_model_name_default(self):
        translator = Translator()
        assert translator._get_nllb_model_name() == "facebook/nllb-200-distilled-600M"

    def test_get_nllb_model_name_custom(self):
        config = TranslatorConfig(model_name="facebook/nllb-200-3.3B")
        translator = Translator(config=config)
        assert translator._get_nllb_model_name() == "facebook/nllb-200-3.3B"

    def test_get_supported_languages_nllb(self):
        translator = Translator(TranslatorConfig(backend="nllb"))
        languages = translator.get_supported_languages()
        assert "en" in languages
        assert "es" in languages
        assert "zh" in languages
        assert len(languages) == len(NLLB_LANGUAGE_CODES)

    def test_get_supported_languages_helsinki(self):
        translator = Translator(TranslatorConfig(backend="helsinki"))
        languages = translator.get_supported_languages()
        assert "en" in languages
        assert len(languages) == len(HELSINKI_LANGUAGE_PAIRS)

    def test_is_language_pair_supported_nllb(self):
        translator = Translator(TranslatorConfig(backend="nllb"))
        assert translator.is_language_pair_supported("en", "es") is True
        assert translator.is_language_pair_supported("en", "zh") is True
        assert translator.is_language_pair_supported("ja", "ko") is True
        # Non-existent language
        assert translator.is_language_pair_supported("en", "xyz") is False

    def test_is_language_pair_supported_helsinki(self):
        translator = Translator(TranslatorConfig(backend="helsinki"))
        assert translator.is_language_pair_supported("en", "es") is True
        assert translator.is_language_pair_supported("en", "fr") is True
        # Helsinki has limited pairs
        assert translator.is_language_pair_supported("ja", "ko") is False

    def test_parse_srt_simple(self):
        translator = Translator()
        srt_content = """1
00:00:01,000 --> 00:00:03,500
Hello world

2
00:00:04,000 --> 00:00:06,000
Second line
"""
        entries = translator._parse_srt(srt_content)
        assert len(entries) == 2
        assert entries[0].index == 1
        assert entries[0].start_time == "00:00:01,000"
        assert entries[0].end_time == "00:00:03,500"
        assert entries[0].text == "Hello world"
        assert entries[1].index == 2
        assert entries[1].text == "Second line"

    def test_parse_srt_multiline_text(self):
        translator = Translator()
        srt_content = """1
00:00:01,000 --> 00:00:03,500
Hello world
This is a second line
"""
        entries = translator._parse_srt(srt_content)
        assert len(entries) == 1
        assert entries[0].text == "Hello world\nThis is a second line"

    def test_parse_srt_windows_line_endings(self):
        translator = Translator()
        srt_content = "1\r\n00:00:01,000 --> 00:00:03,500\r\nHello world\r\n\r\n"
        entries = translator._parse_srt(srt_content)
        assert len(entries) == 1
        assert entries[0].text == "Hello world"

    def test_parse_srt_empty(self):
        translator = Translator()
        entries = translator._parse_srt("")
        assert len(entries) == 0

    def test_parse_srt_invalid_index(self):
        translator = Translator()
        srt_content = """abc
00:00:01,000 --> 00:00:03,500
Hello world
"""
        entries = translator._parse_srt(srt_content)
        assert len(entries) == 0

    def test_entries_to_srt(self):
        translator = Translator()
        entries = [
            SubtitleEntry(1, "00:00:01,000", "00:00:03,500", "Hello world"),
            SubtitleEntry(2, "00:00:04,000", "00:00:06,000", "Second line"),
        ]
        srt = translator._entries_to_srt(entries)
        assert "1\n00:00:01,000 --> 00:00:03,500\nHello world\n" in srt
        assert "2\n00:00:04,000 --> 00:00:06,000\nSecond line\n" in srt

    def test_translate_text_same_language(self):
        translator = Translator()
        result = translator.translate_text("Hello", "en", "en")
        assert result == "Hello"

    def test_translate_texts_same_language(self):
        translator = Translator()
        texts = ["Hello", "World"]
        result = translator.translate_texts(texts, "en", "en")
        assert result == texts

    @patch("homelab_subs.core.translation.Translator._ensure_model_loaded")
    @patch("homelab_subs.core.translation.Translator._translate_batch_nllb")
    def test_translate_text_with_mock(self, mock_translate, mock_load):
        mock_translate.return_value = ["Hola"]
        translator = Translator()
        result = translator.translate_text("Hello", "en", "es")
        assert result == "Hola"
        mock_load.assert_called_once_with("en", "es")

    @patch("homelab_subs.core.translation.Translator._ensure_model_loaded")
    @patch("homelab_subs.core.translation.Translator._translate_batch_nllb")
    def test_translate_texts_with_progress(self, mock_translate, mock_load):
        mock_translate.return_value = ["Hola", "Mundo"]
        translator = Translator(TranslatorConfig(batch_size=2))

        progress_calls = []

        def progress_cb(pct, count):
            progress_calls.append((pct, count))

        result = translator.translate_texts(
            ["Hello", "World"], "en", "es", progress_callback=progress_cb
        )
        assert result == ["Hola", "Mundo"]
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 100.0

    def test_translate_srt_file_not_found(self, tmp_path):
        translator = Translator()
        input_path = tmp_path / "nonexistent.srt"
        output_path = tmp_path / "output.srt"
        with pytest.raises(FileNotFoundError):
            translator.translate_srt_file(input_path, output_path, "en", "es")

    def test_translate_srt_file_unsupported_pair(self, tmp_path):
        translator = Translator(TranslatorConfig(backend="helsinki"))
        input_path = tmp_path / "input.srt"
        input_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        output_path = tmp_path / "output.srt"
        with pytest.raises(ValueError, match="not supported"):
            translator.translate_srt_file(input_path, output_path, "ja", "ko")

    @patch("homelab_subs.core.translation.Translator._ensure_model_loaded")
    @patch("homelab_subs.core.translation.Translator._translate_batch_nllb")
    def test_translate_srt_file_success(self, mock_translate, mock_load, tmp_path):
        mock_translate.return_value = ["Hola mundo"]

        input_path = tmp_path / "input.srt"
        input_path.write_text(
            "1\n00:00:01,000 --> 00:00:03,500\nHello world\n",
            encoding="utf-8",
        )

        output_path = tmp_path / "output.es.srt"
        translator = Translator()
        result = translator.translate_srt_file(input_path, output_path, "en", "es")

        assert result == output_path
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Hola mundo" in content
        assert "00:00:01,000 --> 00:00:03,500" in content

    @patch("homelab_subs.core.translation.Translator._ensure_model_loaded")
    @patch("homelab_subs.core.translation.Translator._translate_batch_nllb")
    def test_translate_srt_file_empty(self, mock_translate, mock_load, tmp_path):
        input_path = tmp_path / "empty.srt"
        input_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "output.srt"

        translator = Translator()
        result = translator.translate_srt_file(input_path, output_path, "en", "es")

        assert result == output_path
        assert output_path.exists()
        mock_translate.assert_not_called()


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_list_supported_languages_nllb(self):
        languages = list_supported_languages("nllb")
        assert "en" in languages
        assert languages["en"] == "English"
        assert "es" in languages
        assert languages["es"] == "Spanish"

    def test_list_supported_languages_helsinki(self):
        languages = list_supported_languages("helsinki")
        assert "en" in languages
        assert languages["en"] == "English"

    def test_get_available_backends(self):
        # This test depends on whether transformers is installed
        backends = get_available_backends()
        assert isinstance(backends, list)
        # If transformers is installed, both backends should be available
        # If not, the list will be empty


class TestLanguageCodes:
    """Tests for language code mappings."""

    def test_nllb_codes_format(self):
        """NLLB codes should follow flores-200 format."""
        for code, nllb_code in NLLB_LANGUAGE_CODES.items():
            assert "_" in nllb_code
            parts = nllb_code.split("_")
            assert len(parts) == 2
            assert len(parts[0]) == 3  # ISO 639-3 code
            assert len(parts[1]) == 4  # Script code

    def test_helsinki_pairs_bidirectional_not_guaranteed(self):
        """Helsinki pairs are not necessarily bidirectional."""
        # en->es is available
        assert "es" in HELSINKI_LANGUAGE_PAIRS.get("en", [])
        # es->en should also be available
        assert "en" in HELSINKI_LANGUAGE_PAIRS.get("es", [])

    def test_common_languages_supported_by_nllb(self):
        """Common languages should be supported by NLLB."""
        common = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", "ar"]
        for lang in common:
            assert lang in NLLB_LANGUAGE_CODES, f"{lang} should be in NLLB codes"
