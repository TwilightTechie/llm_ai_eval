"""
Comprehensive logging configuration for OCR Evaluator.
Supports detailed logging with performance metrics, memory usage, and API call tracking.
"""

import logging
import logging.handlers
import os
import sys
import time
import traceback
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
from contextlib import contextmanager
import json
import threading


# ANSI color codes for console output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }
    
    def __init__(self, fmt: str, datefmt: str = None, colorized: bool = True):
        super().__init__(fmt, datefmt)
        self.colorized = colorized
    
    def format(self, record: logging.LogRecord) -> str:
        if self.colorized and sys.stdout.isatty():
            color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"
            record.name = f"{Colors.BLUE}{record.name}{Colors.RESET}"
            if hasattr(record, 'duration'):
                record.duration = f"{Colors.MAGENTA}{record.duration}{Colors.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        for key in ['duration', 'memory_mb', 'api_call', 'document_id', 
                    'provider', 'batch_id', 'worker_id', 'metrics']:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class PerformanceLogger:
    """
    Context manager and decorator for logging performance metrics.
    Tracks execution time, memory usage, and custom metrics.
    """
    
    def __init__(self, logger: logging.Logger, operation: str, 
                 log_memory: bool = True, extra: Dict[str, Any] = None):
        self.logger = logger
        self.operation = operation
        self.log_memory = log_memory
        self.extra = extra or {}
        self.start_time = None
        self.start_memory = None
        self._lock = threading.Lock()
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.log_memory:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        self.logger.debug(
            f"Starting: {self.operation}",
            extra=self.extra
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        
        extra = {
            **self.extra,
            "duration": f"{duration:.3f}s",
            "duration_ms": round(duration * 1000, 2),
        }
        
        if self.log_memory:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)
            extra["memory_mb"] = round(current_memory, 2)
            extra["memory_delta_mb"] = round(current_memory - self.start_memory, 2)
        
        if exc_type is not None:
            self.logger.error(
                f"Failed: {self.operation} after {duration:.3f}s - {exc_val}",
                extra=extra,
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            self.logger.info(
                f"Completed: {self.operation} in {duration:.3f}s",
                extra=extra
            )
        
        return False  # Don't suppress exceptions
    
    def __call__(self, func):
        """Use as a decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceLogger(
                self.logger, 
                f"{func.__module__}.{func.__name__}",
                self.log_memory,
                self.extra
            ):
                return func(*args, **kwargs)
        return wrapper
    
    def log_metric(self, name: str, value: Any):
        """Log a custom metric during the operation."""
        with self._lock:
            if 'metrics' not in self.extra:
                self.extra['metrics'] = {}
            self.extra['metrics'][name] = value


class APICallLogger:
    """Logger specifically for API calls with rate limiting awareness."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.call_count = 0
        self.total_tokens = 0
        self._lock = threading.Lock()
    
    @contextmanager
    def log_call(self, provider: str, endpoint: str, extra: Dict[str, Any] = None):
        """Context manager for logging API calls."""
        start_time = time.perf_counter()
        call_extra = {
            "api_call": True,
            "provider": provider,
            "endpoint": endpoint,
            **(extra or {})
        }
        
        self.logger.debug(f"API Call: {provider}/{endpoint}", extra=call_extra)
        
        try:
            yield self
        except Exception as e:
            duration = time.perf_counter() - start_time
            call_extra["duration"] = f"{duration:.3f}s"
            call_extra["status"] = "failed"
            self.logger.error(
                f"API Call Failed: {provider}/{endpoint} - {e}",
                extra=call_extra,
                exc_info=True
            )
            raise
        else:
            duration = time.perf_counter() - start_time
            with self._lock:
                self.call_count += 1
            call_extra["duration"] = f"{duration:.3f}s"
            call_extra["status"] = "success"
            call_extra["total_calls"] = self.call_count
            self.logger.info(
                f"API Call Success: {provider}/{endpoint} ({duration:.3f}s)",
                extra=call_extra
            )
    
    def log_tokens(self, input_tokens: int, output_tokens: int):
        """Log token usage for rate limiting awareness."""
        with self._lock:
            self.total_tokens += input_tokens + output_tokens
        self.logger.debug(
            f"Token usage: input={input_tokens}, output={output_tokens}, total={self.total_tokens}"
        )


def setup_logging(
    config: Dict[str, Any] = None,
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_format: str = "detailed",
    colorized: bool = True,
    log_to_file: bool = True,
    max_file_size_mb: int = 50,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging for OCR Evaluator.
    
    Args:
        config: Configuration dict (from evaluation.yaml)
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format style (simple, detailed, json)
        colorized: Enable colored console output
        log_to_file: Enable file logging
        max_file_size_mb: Max log file size before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Root logger configured for the application
    """
    # Use config if provided
    if config:
        logging_config = config.get('logging', {})
        log_level = logging_config.get('level', log_level)
        log_format = logging_config.get('format', log_format)
        
        file_config = logging_config.get('file', {})
        log_to_file = file_config.get('enabled', log_to_file)
        log_dir = os.path.dirname(file_config.get('path', f"{log_dir}/evaluation.log"))
        max_file_size_mb = file_config.get('max_size_mb', max_file_size_mb)
        backup_count = file_config.get('backup_count', backup_count)
        
        console_config = logging_config.get('console', {})
        colorized = console_config.get('colorized', colorized)
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger("ocr_evaluator")
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Define format strings
    if log_format == "simple":
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
    elif log_format == "detailed":
        fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s"
    else:
        fmt = None  # JSON format
    
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter(fmt, datefmt, colorized))
    
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        log_file = log_path / "evaluation.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File always captures DEBUG
        
        if log_format == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            # File uses detailed format without colors
            file_fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s"
            file_handler.setFormatter(logging.Formatter(file_fmt, datefmt))
        
        root_logger.addHandler(file_handler)
        
        # Separate error log file
        error_file = log_path / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(file_fmt, datefmt))
        root_logger.addHandler(error_handler)
    
    # Log startup info
    root_logger.info("=" * 60)
    root_logger.info("OCR Evaluator - Logging Initialized")
    root_logger.info(f"Log Level: {log_level}")
    root_logger.info(f"Log Format: {log_format}")
    root_logger.info(f"Log Directory: {log_path.absolute()}")
    root_logger.info("=" * 60)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"ocr_evaluator.{name}")


# Convenience function for quick performance logging
@contextmanager
def log_performance(operation: str, logger: logging.Logger = None, **extra):
    """Quick context manager for performance logging."""
    if logger is None:
        logger = get_logger("performance")
    
    with PerformanceLogger(logger, operation, extra=extra):
        yield


