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

@app.post("/reset")
async def reset(request: ResetRequest) -> Observation:
    global env, current_task

    task_map = {
        "balanced_assignment_week": get_balanced_assignment_week,
        "deadline_collision_week": get_deadline_collision_week,
        "burnout_prevention_mid_sems": get_burnout_prevention_mid_sems
    }

    if request.task not in task_map:
        raise HTTPException(status_code=400, detail=f"Unknown task: {request.task}")

    current_task = request.task
    task_config = task_map[request.task]
    env = OpenStudyBalanceEnv(task_config)
    observation = env.reset()

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