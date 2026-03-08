"""FastAPI application for the ADHD Coaching Environment.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app

from models import ADHDAction, ADHDObservation
from .adhd_env_environment import ADHDEnvironment

app = create_app(
    ADHDEnvironment,
    ADHDAction,
    ADHDObservation,
    env_name="adhd_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for: uv run --project . server"""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
