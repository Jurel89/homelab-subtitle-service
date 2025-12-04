# homelab-subtitle-service

Main repository for automatic subtitle generation - a homelab-focused tool for generating high-quality subtitles from video files using FFmpeg and Whisper.

## 🎯 Features

- **CLI Tool**: Simple command-line interface for subtitle generation
- **Batch Processing**: Process multiple videos from a YAML configuration file
- **Multiple Models**: Support for all Whisper model sizes (tiny, base, small, medium, large)
- **Language Support**: Auto-detection or specify language codes
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Structured Logging**: Comprehensive logging with JSON output for monitoring
- **Web UI Ready**: Log format designed for integration with web dashboards

## 📋 Requirements

- Python 3.10 or higher
- FFmpeg (for audio extraction)
- faster-whisper (Whisper implementation)

## 🚀 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/Jurel89/homelab-subtitle-service.git
cd homelab-subtitle-service

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## 📖 Usage

### Basic Usage

Generate subtitles for a single video:

```bash
subsvc generate video.mp4
```

This will create `video.en.srt` in the same directory.

### Specify Output Path

```bash
subsvc generate video.mp4 --output /path/to/subtitles.srt
```

### Choose Whisper Model

```bash
subsvc generate video.mp4 --model medium
```

Available models: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`

### Language Options

```bash
# Auto-detect language
subsvc generate video.mp4 --lang ""

# Specify language
subsvc generate video.mp4 --lang es  # Spanish
subsvc generate video.mp4 --lang fr  # French
```

### Translation to English

```bash
subsvc generate video.mp4 --task translate
```

### GPU Acceleration

```bash
subsvc generate video.mp4 --device cuda --compute-type float16
```

### Batch Processing

Create a YAML configuration file:

```yaml
# jobs.yaml
jobs:
  - file: /media/movies/Movie1.mkv
    lang: en
    model: small
    output: /subs/Movie1.en.srt
  
  - file: /media/movies/Movie2.mkv
    lang: es
    model: medium
```

Run batch processing:

```bash
subsvc batch jobs.yaml
```

## 📊 Logging

The service includes comprehensive structured logging for monitoring and debugging.

### Log Levels

```bash
# Info level (default)
subsvc generate video.mp4 --log-level INFO

# Debug level (verbose)
subsvc generate video.mp4 --log-level DEBUG

# Warning level (minimal output)
subsvc generate video.mp4 --log-level WARNING
```

### Log to File

Save logs in JSON format for later analysis:

```bash
subsvc generate video.mp4 --log-file /var/log/subsvc/job.log
```

### Example Log Output

```
[2025-12-01 10:30:45] [INFO] [homelab_subs.cli] [Job:a3f2b1c4] Starting subtitle generation for video.mp4
[2025-12-01 10:30:45] [INFO] [homelab_subs.core.audio] [Job:a3f2b1c4] [audio_extraction] Starting audio_extraction
[2025-12-01 10:30:48] [INFO] [homelab_subs.core.audio] Audio extracted successfully: video_audio.wav (45.23 MB)
[2025-12-01 10:30:48] [INFO] [homelab_subs.core.transcription] [Job:a3f2b1c4] [transcription] Starting transcription
[2025-12-01 10:32:15] [INFO] [homelab_subs.core.transcription] Transcription complete: 142 segments generated
[2025-12-01 10:32:15] [INFO] [homelab_subs.cli] Successfully generated subtitles: video.en.srt
```

### Web-Based Log Viewer

An example web dashboard is included for viewing logs:

```bash
# Install Flask (optional)
pip install flask

# Start the log viewer
python examples/log_viewer.py --log-file /var/log/subsvc/job.log

# Open http://localhost:5000 in your browser
```

For detailed logging documentation, see [docs/LOGGING.md](docs/LOGGING.md).

## 📈 Performance Monitoring & History

Track CPU, memory, and GPU usage during subtitle generation. Monitoring is **enabled by default** and stores job history in a local SQLite database.

### View Job History

```bash
# Show recent jobs
subsvc history

# Show overall statistics
subsvc history --stats

# Show detailed job info with metrics
subsvc history --job-id <job-id>

# Filter by status
subsvc history --status completed
```

### Disable Monitoring

```bash
# Disable all monitoring
subsvc generate video.mp4 --no-monitoring

# Keep monitoring but don't save to database
subsvc generate video.mp4 --no-db-logging
```

### Install Monitoring Dependencies

Monitoring requires additional packages:

```bash
# System monitoring (CPU, memory)
pip install psutil

# GPU monitoring (NVIDIA, Linux only)
pip install nvidia-ml-py

# Or install all dev dependencies
pip install -e ".[dev]"
```

### Features
- 📊 Real-time performance tracking (CPU, memory, disk I/O, GPU)
- 💾 Persistent SQLite database (`~/.homelab-subs/logs.db`)
- 📈 Time-series metrics for visualization
- 🔍 Query API for Web UI integration
- 🚀 Ready for dashboard integration

For complete monitoring documentation, see [docs/MONITORING.md](docs/MONITORING.md).

## 🏗️ Project Structure

```
homelab-subtitle-service/
├── src/
│   └── homelab_subs/
│       ├── cli.py              # CLI interface
│       ├── logging_config.py   # Logging configuration
│       └── core/
│           ├── audio.py        # FFmpeg wrapper
│           ├── transcription.py # Whisper wrapper
│           ├── srt.py          # SRT file generation
│           └── pipeline.py     # High-level pipeline
├── examples/
│   ├── basic_jobs.yaml         # Example batch config
│   ├── log_viewer.py           # Web-based log viewer
│   └── analyze_logs.py         # CLI log analysis tool
├── docs/
│   ├── LOGGING.md              # Logging documentation
│   └── LOGGING_QUICKREF.md     # Logging quick reference
└── src/
    └── tests/                  # Unit tests
        ├── test_audio.py
        ├── test_cli.py
        ├── test_srt.py
        ├── test_transcription.py
        └── test_logging.py     # Logging tests
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=homelab_subs

# Run specific test file
pytest src/tests/test_audio.py

# Run logging tests
pytest src/tests/test_logging.py -v
```

## 🔧 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/

# Run type checker
mypy src/
```

## 📝 API Usage

You can also use the library programmatically:

```python
from pathlib import Path
from homelab_subs.core.pipeline import generate_subtitles_for_video

video_path = Path("video.mp4")
output_path = Path("video.en.srt")

generate_subtitles_for_video(
    video_path=video_path,
    output_path=output_path,
    lang="en",
    model_name="small",
    device="cpu",
    compute_type="int8"
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

Jose Ibañez Ortiz

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing

