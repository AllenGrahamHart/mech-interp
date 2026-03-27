from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "gemma-morality-interp"
HF_CACHE_DIR = "/cache/hf"
RUNS_DIR = "/root/project/runs"
SOURCE_DIR = "/root/src"

hf_cache_volume = modal.Volume.from_name("mech-interp-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("mech-interp-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("src", remote_path=SOURCE_DIR)
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
    .env(
        {
            "PYTHONPATH": SOURCE_DIR,
            "HF_HOME": HF_CACHE_DIR,
            "HUGGINGFACE_HUB_CACHE": HF_CACHE_DIR,
            "TRANSFORMERS_CACHE": HF_CACHE_DIR,
        }
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        RUNS_DIR: runs_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_remote(config: str, statements: str | None = None) -> dict[str, object]:
    from mech_interp.pipeline import run_morality_direction

    os.makedirs("/root/project", exist_ok=True)
    os.chdir("/root/project")

    result = run_morality_direction(config_path=config, statements_path=statements)
    hf_cache_volume.commit()
    runs_volume.commit()
    return result


@app.local_entrypoint()
def main(
    config: str = "configs/morality_direction_it.toml",
    statements: str | None = None,
) -> None:
    local_config = Path(config)
    if local_config.parts and local_config.parts[0] == "configs":
        remote_config = str(Path("/root/configs") / local_config.relative_to("configs"))
    else:
        remote_config = str(Path("/root/configs") / local_config.name)

    remote_statements = None
    if statements is not None:
        local_statements = Path(statements)
        if local_statements.parts and local_statements.parts[0] == "data":
            remote_statements = str(Path("/root/data") / local_statements.relative_to("data"))
        else:
            remote_statements = str(Path("/root/data") / local_statements.name)

    result = run_remote.remote(config=remote_config, statements=remote_statements)
    print(result)
