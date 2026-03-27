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
    .env(
        {
            "PYTHONPATH": SOURCE_DIR,
            "HF_HOME": HF_CACHE_DIR,
            "HUGGINGFACE_HUB_CACHE": HF_CACHE_DIR,
            "TRANSFORMERS_CACHE": HF_CACHE_DIR,
        }
    )
    .add_local_dir("src", remote_path=SOURCE_DIR)
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
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
def run_remote(
    mode: str,
    config: str,
    statements: str | None = None,
    layer: int | None = None,
    control_config: str | None = None,
) -> dict[str, object]:
    from mech_interp.pipeline import (
        run_control_causal_benchmark,
        run_causal_benchmark,
        run_layer_sweep,
        run_morality_direction,
        run_orthogonalized_causal_benchmark,
        run_orthogonalized_control,
    )

    os.makedirs("/root/project", exist_ok=True)
    os.chdir("/root/project")

    if mode == "run":
        result = run_morality_direction(config_path=config, statements_path=statements)
    elif mode == "sweep":
        result = run_layer_sweep(config_path=config)
    elif mode == "causal":
        result = run_causal_benchmark(config_path=config, layer=layer)
    elif mode == "orthogonalize":
        if control_config is None:
            raise ValueError("control_config is required for orthogonalize mode.")
        result = run_orthogonalized_control(
            config_path=config,
            control_config_path=control_config,
            layer=layer,
        )
    elif mode == "orthogonalized-causal":
        if control_config is None:
            raise ValueError("control_config is required for orthogonalized-causal mode.")
        result = run_orthogonalized_causal_benchmark(
            config_path=config,
            control_config_path=control_config,
            layer=layer,
        )
    elif mode == "control-causal":
        if control_config is None:
            raise ValueError("control_config is required for control-causal mode.")
        result = run_control_causal_benchmark(
            config_path=config,
            control_config_path=control_config,
            layer=layer,
        )
    else:
        raise ValueError(f"Unsupported mode '{mode}'.")

    hf_cache_volume.commit()
    runs_volume.commit()
    return result


@app.local_entrypoint()
def main(
    mode: str = "run",
    config: str = "configs/morality_direction_it.toml",
    statements: str | None = None,
    layer: int | None = None,
    control_config: str | None = None,
) -> None:
    local_config = Path(config)
    if local_config.parts and local_config.parts[0] == "configs":
        remote_config = str(Path("/root/configs") / local_config.relative_to("configs"))
    else:
        remote_config = str(Path("/root/configs") / local_config.name)

    remote_control_config = None
    if control_config is not None:
        local_control_config = Path(control_config)
        if local_control_config.parts and local_control_config.parts[0] == "configs":
            remote_control_config = str(
                Path("/root/configs") / local_control_config.relative_to("configs")
            )
        else:
            remote_control_config = str(Path("/root/configs") / local_control_config.name)

    remote_statements = None
    if statements is not None:
        local_statements = Path(statements)
        if local_statements.parts and local_statements.parts[0] == "data":
            remote_statements = str(Path("/root/data") / local_statements.relative_to("data"))
        else:
            remote_statements = str(Path("/root/data") / local_statements.name)

    result = run_remote.remote(
        mode=mode,
        config=remote_config,
        statements=remote_statements,
        layer=layer,
        control_config=remote_control_config,
    )
    print(result)
