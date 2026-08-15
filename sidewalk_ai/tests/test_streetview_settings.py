"""io.streetview.Settings must tolerate a real .env file, not just a bare one.

Regression: Settings reads GOOGLE_API_KEY, which has no SWAI_ prefix, so its
model_config points env_file at ".env" directly instead of using env_prefix.
pydantic-settings' default extra="forbid" then rejected every unrelated key the
same .env file carries for other Settings classes (SWAI_DEBUG, SWAI_LOG_LEVEL,
SWAI_DEPTH, SWAI_IMG_*, ...), so importing sidewalk_ai.io.streetview raised
ValidationError and the CLI/API failed before doing anything -- this was only
caught once ``.env.example`` was actually copied to ``.env`` and filled in, as
the README instructs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sidewalk_ai.io.streetview import Settings

ENV_EXAMPLE = Path(__file__).parents[2] / ".env.example"


def test_settings_accepts_the_full_env_example_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        ENV_EXAMPLE.read_text(encoding="utf-8").replace(
            "GOOGLE_API_KEY=", "GOOGLE_API_KEY=test-key"
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = Settings()

    assert cfg.google_api_key == "test-key"


def test_settings_ignores_keys_meant_for_other_settings_classes(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GOOGLE_API_KEY=test-key\n"
        "SWAI_DEBUG=1\n"
        "SWAI_LOG_LEVEL=DEBUG\n"
        "SWAI_DEPTH=midas\n"
        "SWAI_RANK_MID_RATIO=0.5\n"
        "SWAI_IMG_AUTO_CROP_GOOGLE_LOGO=true\n"
        "SWAI_IMG_GOOGLE_BAR_HEIGHT_PX=20\n"
        "TORCH_HOME=/models/torch\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = Settings()

    assert cfg.google_api_key == "test-key"
    assert cfg.default_fov == 90  # unrelated keys did not perturb declared fields


def test_settings_still_validates_its_own_fields(tmp_path, monkeypatch):
    # extra="ignore" only widens what is tolerated around the model; a bad
    # value for a field this class actually declares must still fail loudly.
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=test-key\nDEFAULT_FOV=not-a-number\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError):
        Settings()
