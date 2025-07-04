from pydantic import BaseSettings, Field
from typing import Dict, Any

class Settings(BaseSettings):
    ...
    apply_refine_default: bool = Field(True, env="SWAI_REFINE_DEFAULT")
    refine_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"max_gap_x": 24, "max_gap_y": 6}
    )