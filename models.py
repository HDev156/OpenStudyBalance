from pydantic import BaseModel
from typing import List, Optional, Dict


class TaskItem(BaseModel):
    name: str
    hours_needed: float
    deadline_days: int
    priority: int
    scheduled_hours: float


class Observation(BaseModel):
    pending_tasks: List[TaskItem]
    available_hours_per_day: float
    stress_level: float
    sleep_hours: float
    message: str


class Action(BaseModel):
    action_type: str
    task_name: Optional[str] = None
    day: Optional[int] = None
    hours: Optional[float] = None


class Reward(BaseModel):
    value: float
    reason: str


from pydantic import Field

class EnvironmentState(BaseModel):
    current_day: int
    tasks: List[TaskItem]
    available_hours_per_day: float
    stress_level: float
    sleep_hours: float
    daily_scheduled: Dict[int, float] = Field(default_factory=dict)
    burnout_risk: float = 0.0  # New: burnout risk score 0.0-1.0