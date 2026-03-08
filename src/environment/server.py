"""FastAPI server for HF Spaces deployment.

Exposes ADHD environment via HTTP API for OpenEnv integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openenv.core.client_types import StepResult
from typing import Dict, Any

from .adhd_env import ADHDEnvironment
from .models import ADHDAction


# Global environment instance
env = ADHDEnvironment()

# Create FastAPI app
app = FastAPI(
    title="ADHD Task Initiation Coaching Environment",
    description="OpenEnv environment for evaluating ADHD coaching quality",
    version="0.1.0-v1"
)

# CORS middleware for client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint - environment info"""

    return {
        "name": "ADHD Task Initiation Coaching Environment",
        "version": "0.1.0-v1",
        "stage": "minimal",
        "description": "OpenEnv environment for evaluating ADHD coaching responses",
        "endpoints": {
            "info": "GET /info",
            "reset": "POST /reset",
            "step": "POST /step",
            "health": "GET /health",
        },
        "repository": "https://github.com/user/adhd-coach",
        "documentation": "See /info for environment details",
    }


@app.get("/info")
async def get_info() -> Dict[str, Any]:
    """Get environment metadata"""

    return env.get_info()


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint"""

    return {"status": "healthy"}


@app.post("/reset")
async def reset() -> StepResult:
    """Reset environment - generate new scenario

    Returns:
        StepResult with initial observation
    """

    try:
        result = env.reset()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(action: ADHDAction) -> StepResult:
    """Score a coaching response

    Args:
        action: ADHDAction with tool_calls and message

    Returns:
        StepResult with score and explanation

    Raises:
        HTTPException: If step fails (e.g., reset not called)
    """

    try:
        result = await env.step(action)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
