from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from env import OpenStudyBalanceEnv
from models import Action, Observation
from tasks import get_balanced_assignment_week, get_deadline_collision_week, get_burnout_prevention_mid_sems

app = FastAPI(title="OpenStudyBalance Environment API")

# Global environment instance
env = None
current_task = None

class ResetRequest(BaseModel):
    task: str

class StepRequest(BaseModel):
    action: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "name": "OpenStudyBalance",
        "description": "An OpenEnv environment that simulates burnout-aware academic planning for students",
        "version": "0.1.0",
        "tasks": ["balanced_assignment_week", "deadline_collision_week", "burnout_prevention_mid_sems"]
    }

@app.get("/state")
async def get_state():
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    return env.state()

@app.post("/reset")
async def reset(request: ResetRequest) -> Observation:
    global env, current_task

    current_task = request.task
    env = OpenStudyBalanceEnv()
    observation = env.reset(request.task)

    return observation

@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    global env

    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        action = Action(**request.action)
        observation, reward, done, info = env.step(action)

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)