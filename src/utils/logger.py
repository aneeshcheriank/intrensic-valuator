"""
Centralised logging configuration.

Provides a `get_logger()` helper that returns a Python stdlib Logger
configured with Rich for colourful, structured console output.
"""

from __future__ import annotations

import logging
import sys
from typing import ClassVar


class _LoggerManager:
    """Lazy-initialise the Rich logging handler once."""

    _initialised: ClassVar[bool] = False

    @classmethod
    def init(cls) -> None:
        if cls._initialised:
            return

        try:
            from rich.console import Console
            from rich.logging import RichHandler
        except ImportError:
            # Rich is not available — fall back to plain stdout logging.
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        else:
            console = Console(stderr=True)
            handler = RichHandler(
                console=console,
                show_time=True,
                show_level=True,
                show_path=False,
                rich_tracebacks=True,
            )

        root = logging.getLogger("intrensic")
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        root.propagate = False

        cls._initialised = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the ``intrensic`` hierarchy.

    Pass ``__name__`` from the calling module to get a namespaced logger
    (e.g. ``get_logger(__name__)``).
    """
    _LoggerManager.init()
    if not name.startswith("intrensic"):
        name = f"intrensic.{name}"
    return logging.getLogger(name)
