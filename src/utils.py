"""Shared utilities: config loading, logging, path helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load the YAML configuration file into a plain dict."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config relative to the project root, creating
    parent directories if they don't exist yet."""
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configure root logging based on the `logging` section of config.yaml."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file = log_cfg.get("log_file")

    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = resolve_path(log_file)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("kg_pipeline")


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(PROJECT_ROOT / d, exist_ok=True)
