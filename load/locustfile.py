"""Sequential /predict load; one HTTP client session per simulated user."""

import random

from locust import HttpUser, constant, task

# Fixed short inputs: load shape is the variable, not payload variety.
TEXTS = (
    "a good film with a warm ending",
    "the plot was dull and far too long",
    "not bad at all, I would watch it again",
)


class PredictUser(HttpUser):
    wait_time = constant(0)

    @task
    def predict(self):
        self.client.post("/predict", json={"text": random.choice(TEXTS)})
