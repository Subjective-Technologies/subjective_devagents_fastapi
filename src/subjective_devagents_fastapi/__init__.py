"""Subjective DevAgents FastAPI backend.

Python port of the Node backend that used to live inside the DevAgents client
(clients/desktop/subjective_devagents/src). The client is now web-only: the browser
talks to this service over HTTP + one WebSocket.
"""

__all__ = ["app"]


def __getattr__(name):  # lazy so importing the package doesn't require fastapi
    if name == "app":
        from .app import app

        return app
    raise AttributeError(name)
