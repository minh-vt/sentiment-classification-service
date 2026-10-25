"""Runtime settings, read from the process environment at process start."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # No default: the three places that know where a bundle lives (image, .env,
    # developer shell) must say so, instead of inheriting a path that is wrong
    # in the other two contexts.
    model_dir: Path
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_positions: int = 512
    max_concurrency: int = 4

    def __post_init__(self):
        if self.max_batch_size < 1 or self.max_positions < 1 or self.max_concurrency < 1:
            raise ValueError("MAX_BATCH_SIZE, MAX_POSITIONS and MAX_CONCURRENCY must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("PORT must be between 1 and 65535")

    @classmethod
    def from_env(cls, env=os.environ):
        if not env.get("MODEL_DIR"):
            raise ValueError("MODEL_DIR is required: point it at a packaged model bundle")
        return cls(
            model_dir=Path(env["MODEL_DIR"]),
            host=env.get("HOST", "0.0.0.0"),
            port=int(env.get("PORT", "8000")),
            max_batch_size=int(env.get("MAX_BATCH_SIZE", "32")),
            max_positions=int(env.get("MAX_POSITIONS", "512")),
            max_concurrency=int(env.get("MAX_CONCURRENCY", "4")),
        )
