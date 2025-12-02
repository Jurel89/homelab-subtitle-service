from homelab_subs.core.transcription import Transcriber, TranscriberConfig, Segment


class FakeWhisperSegment:
    def __init__(self, start, end, text, avg_logprob=0.0, no_speech_prob=0.0):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class FakeWhisperInfo:
    def __init__(self, language="en", language_probability=0.99, duration=2.0):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        self.called_with = (args, kwargs)

    def transcribe(self, audio_path, language=None, task=None, beam_size=None, vad_filter=None):
        segments = [
            FakeWhisperSegment(0.0, 1.0, "Hello"),
            FakeWhisperSegment(1.0, 2.0, "World"),
        ]
        info = FakeWhisperInfo()
        return segments, info


def test_transcriber_uses_model(monkeypatch, tmp_path):
    # Create a dummy audio file
    audio = tmp_path / "dummy.wav"
    audio.write_bytes(b"fake-audio-content")

    # Monkeypatch the WhisperModel to our fake implementation
    from homelab_subs.core import transcription as transcription_module

    def fake_whisper_model_constructor(model_name, **kwargs):
        return FakeWhisperModel(model_name, **kwargs)

    monkeypatch.setattr(transcription_module, "WhisperModel", fake_whisper_model_constructor)

    config = TranscriberConfig(model_name="small", device="cpu", compute_type="int8")
    transcriber = Transcriber(config=config)

    segments = transcriber.transcribe_file(audio, language="en")

    assert len(segments) == 2
    assert isinstance(segments[0], Segment)
    assert segments[0].text == "Hello"
    assert segments[1].text == "World"
