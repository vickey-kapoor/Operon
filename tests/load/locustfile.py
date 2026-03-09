"""Locust load-test scenarios for UI Navigator."""

import os
import time

from locust import HttpUser, between, task

_API_KEY = os.environ.get("API_KEY", "test-key")
_HEADERS = {"X-API-Key": _API_KEY, "Content-Type": "application/json"}

_POLL_TIMEOUT = 60  # seconds to wait for a task to complete
_POLL_INTERVAL = 2  # seconds between polls


class HealthUser(HttpUser):
    """Steady-state health-check traffic — simulates monitoring probes."""

    wait_time = between(1, 3)

    @task
    def health(self):
        self.client.get("/health")


class NavigateUser(HttpUser):
    """
    Realistic browse-and-poll scenario.

    POST /navigate → poll GET /tasks/{id} until done or timeout.
    """

    wait_time = between(5, 15)

    @task
    def navigate_and_poll(self):
        payload = {
            "task": "Go to example.com and return the page title.",
            "max_steps": 3,
        }
        with self.client.post(
            "/navigate",
            json=payload,
            headers=_HEADERS,
            catch_response=True,
        ) as resp:
            if resp.status_code != 202:
                resp.failure(f"POST /navigate returned {resp.status_code}")
                return
            task_id = resp.json().get("task_id")

        if not task_id:
            return

        deadline = time.time() + _POLL_TIMEOUT
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            with self.client.get(
                f"/tasks/{task_id}",
                headers=_HEADERS,
                name="/tasks/[id]",
                catch_response=True,
            ) as poll:
                if poll.status_code != 200:
                    poll.failure(f"GET /tasks returned {poll.status_code}")
                    return
                status = poll.json().get("status", "")
                if status in ("done", "error"):
                    return
        # Timed out — mark as failure.
        self.client.get(
            f"/tasks/{task_id}",
            headers=_HEADERS,
            name="/tasks/[id] (timeout)",
        )


class BurstUser(HttpUser):
    """
    Burst scenario — rapid concurrent POST /navigate requests.

    Models a sudden spike of 10 tasks with minimal wait between them.
    """

    wait_time = between(0.1, 0.5)

    @task
    def burst_navigate(self):
        payload = {
            "task": "Take a screenshot and describe the page.",
            "max_steps": 1,
        }
        self.client.post(
            "/navigate",
            json=payload,
            headers=_HEADERS,
        )
