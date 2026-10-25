"""Process entrypoint: start the HTTP server and load the bundle in background."""

import logging

import uvicorn

from sentiment_service.api import create_app
from sentiment_service.inference import load_model
from sentiment_service.settings import Settings


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.from_env()
    app = create_app(settings, load_model=lambda: load_model(settings))
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
