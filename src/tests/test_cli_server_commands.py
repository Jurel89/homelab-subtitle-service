# src/tests/test_cli_server_commands.py

"""
Tests for CLI server and worker commands.

These tests verify that the CLI properly exposes the server and worker
commands without requiring the server dependencies to be installed.
"""

from __future__ import annotations

from homelab_subs.cli import _build_parser


class TestCLIServerCommands:
    """Tests for CLI server and worker commands."""

    def test_cli_has_server_command(self):
        """CLI should have server command."""
        parser = _build_parser()

        # Parse server command
        args = parser.parse_args(["server"])
        assert args.command == "server"
        assert args.host == "127.0.0.1"  # Default is localhost for security
        assert args.port == 8000

    def test_cli_has_worker_command(self):
        """CLI should have worker command."""
        parser = _build_parser()

        # Parse worker command
        args = parser.parse_args(["worker"])
        assert args.command == "worker"
        assert args.queues == ["high", "default", "low"]
        assert args.burst is False

    def test_cli_server_custom_host(self):
        """CLI server should accept custom host."""
        parser = _build_parser()

        args = parser.parse_args(["server", "--host", "127.0.0.1"])
        assert args.host == "127.0.0.1"

    def test_cli_server_custom_port(self):
        """CLI server should accept custom port."""
        parser = _build_parser()

        args = parser.parse_args(["server", "--port", "3000"])
        assert args.port == 3000

    def test_cli_server_reload_flag(self):
        """CLI server should accept reload flag."""
        parser = _build_parser()

        args = parser.parse_args(["server", "--reload"])
        assert args.reload is True

    def test_cli_server_workers(self):
        """CLI server should accept workers count."""
        parser = _build_parser()

        args = parser.parse_args(["server", "--workers", "4"])
        assert args.workers == 4

    def test_cli_server_log_level(self):
        """CLI server should accept log level."""
        parser = _build_parser()

        args = parser.parse_args(["server", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_cli_server_all_options(self):
        """CLI server should accept all options together."""
        parser = _build_parser()

        args = parser.parse_args(
            [
                "server",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--workers",
                "2",
                "--reload",
                "--log-level",
                "INFO",
            ]
        )
        assert args.command == "server"
        assert args.host == "0.0.0.0"
        assert args.port == 8080
        assert args.workers == 2
        assert args.reload is True
        assert args.log_level == "INFO"

    def test_cli_worker_custom_queues(self):
        """CLI worker should accept custom queues."""
        parser = _build_parser()

        args = parser.parse_args(["worker", "--queues", "gpu", "high"])
        assert args.queues == ["gpu", "high"]

    def test_cli_worker_burst_mode(self):
        """CLI worker should accept burst mode."""
        parser = _build_parser()

        args = parser.parse_args(["worker", "--burst"])
        assert args.burst is True

    def test_cli_worker_custom_name(self):
        """CLI worker should accept custom name."""
        parser = _build_parser()

        args = parser.parse_args(["worker", "--name", "gpu-worker-1"])
        assert args.name == "gpu-worker-1"

    def test_cli_worker_log_level(self):
        """CLI worker should accept log level."""
        parser = _build_parser()

        args = parser.parse_args(["worker", "--log-level", "WARNING"])
        assert args.log_level == "WARNING"

    def test_cli_worker_all_options(self):
        """CLI worker should accept all options together."""
        parser = _build_parser()

        args = parser.parse_args(
            [
                "worker",
                "--queues",
                "gpu",
                "high",
                "default",
                "--burst",
                "--name",
                "worker-1",
                "--log-level",
                "DEBUG",
            ]
        )
        assert args.command == "worker"
        assert args.queues == ["gpu", "high", "default"]
        assert args.burst is True
        assert args.name == "worker-1"
        assert args.log_level == "DEBUG"

    def test_cli_worker_default_queues_include_priority(self):
        """CLI worker default queues should process in priority order."""
        parser = _build_parser()

        args = parser.parse_args(["worker"])
        # high should come before default, which should come before low
        assert args.queues.index("high") < args.queues.index("default")
        assert args.queues.index("default") < args.queues.index("low")
