import logging
import sys
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class MLLogger:
    _instances: Dict[str, "MLLogger"] = {}
    _logger: Optional[logging.Logger] = None

    def __new__(
        cls,
        name: str = "MLTraining",
        rank_zero_only: bool = False,
        level: int = logging.INFO,
    ) -> "MLLogger":
        if name not in cls._instances:
            instance = super(MLLogger, cls).__new__(cls)
            instance._initialize_logger(name, rank_zero_only, level)
            cls._instances[name] = instance
        return cls._instances[name]

    def _initialize_logger(self, name: str, rank_zero_only: bool, level: int) -> None:
        """Initialize the logger with handlers and configurations."""
        if self._logger is not None:
            return

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self.rank_zero_only = rank_zero_only

        if not self._logger.handlers:
            self._setup_handlers()
        self.capture_all_warnings()

    def set_level(self, level: int) -> None:
        """Set the logging level for the logger and its handlers."""
        if self._logger:
            self._logger.setLevel(level)
            for handler in self._logger.handlers:
                handler.setLevel(level)

    def set_log_file(self, log_file_path: str) -> None:
        """Update the log file path for file handlers."""
        if self._logger:
            for handler in self._logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    self._logger.removeHandler(handler)

            file_formatter = logging.Formatter(
                "[%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d]\n\t%(message)s",
            )
            new_handler = RotatingFileHandler(
                filename=log_file_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            new_handler.setFormatter(file_formatter)
            new_handler.setLevel(self._logger.level)
            self._logger.addHandler(new_handler)

    def _setup_handlers(self) -> None:
        """Configure console and file handlers with formatters."""
        console_formatter = logging.Formatter(
            "[%(asctime)s - %(levelname)s]\n\t%(message)s",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        file_formatter = logging.Formatter(
            "[%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d]\n\t%(message)s",
        )
        file_handler = RotatingFileHandler(
            filename=logs_dir / "training.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)

        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log a message with rank awareness."""
        if self._logger and self._logger.isEnabledFor(level):
            current_rank = getattr(rank_zero_only, "rank", 0)
            formatted_msg = rank_prefixed_message(msg, current_rank)

            if not self.rank_zero_only or current_rank == 0:
                self._logger.log(level, formatted_msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.INFO, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.ERROR, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.log(logging.WARNING, msg, *args, **kwargs)

    def capture_all_warnings(self) -> None:
        """Redirect Python warnings to the logger."""
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.propagate = False

        # Clear existing handlers and add ours
        warnings_logger.handlers = []
        if self._logger:
            warnings_logger.setLevel(self._logger.level)
            for handler in self._logger.handlers:
                warnings_logger.addHandler(handler)

    def ignore_warnings(self, action: str = "ignore") -> None:
        """Control Python warnings filtering.

        Args:
            action: One of 'error', 'ignore', 'always', 'default', 'module', or 'once'
        """
        warnings.filterwarnings(action)
        # Also ignore DeprecationWarnings from the warnings module itself
        warnings.simplefilter(action, DeprecationWarning)

    @classmethod
    def get_logger(
        cls,
        name: str = "MLTraining",
        rank_zero_only: bool = False,
    ) -> "MLLogger":
        return cls(name=name, rank_zero_only=rank_zero_only)


def get_logger(name: Optional[str] = None, rank_zero_only: bool = False) -> MLLogger:
    logger_name = name or "MLTraining"
    return MLLogger.get_logger(name=logger_name, rank_zero_only=rank_zero_only)
