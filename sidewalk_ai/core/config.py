from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by core pipeline components."""

    model_config = SettingsConfigDict(env_prefix="SWAI_", populate_by_name=True)

    apply_refine_default: bool = Field(default=True, validation_alias="SWAI_REFINE_DEFAULT")
    refine_kwargs: dict[str, Any] = Field(default_factory=lambda: {"max_gap_x": 24, "max_gap_y": 6})
