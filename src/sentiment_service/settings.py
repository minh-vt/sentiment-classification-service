"""Runtime settings, read from the process environment at process start.

Field definitions are the single source of both the environment key
(``field.name.upper()``) and the default value, so there is no second copy to
drift from the dataclass.
"""

import os
from dataclasses import MISSING, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # No default: the places that know where a bundle lives (image, .env,
    # developer shell) must say so, instead of inheriting a wrong path.
    model_dir: Path
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_positions: int = 512
    max_concurrency: int = 4

    def __post_init__(self):
        if not 1 <= self.port <= 65535:
            raise ValueError("PORT must be between 1 and 65535")
        for name in ("max_batch_size", "max_positions", "max_concurrency"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name.upper()} must be positive")

    @classmethod
    def from_env(cls, env=os.environ):
        values = {}
        for field in fields(cls):
            key = field.name.upper()
            raw = env.get(key)
            if field.default is MISSING:
                if not raw:
                    raise ValueError(f"{key} is required: point it at a packaged model bundle")
                values[field.name] = Path(raw)
            elif raw is not None:
                values[field.name] = int(raw) if field.type is int else raw
        return cls(**values)
