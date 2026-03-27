from __future__ import annotations

import argparse
import json

from mech_interp.pipeline import run_causal_benchmark, run_layer_sweep, run_morality_direction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mech-interp",
        description="Baseline mechanistic interpretability experiments for Gemma 3 morality directions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Fit a morality direction from ETHICS and optionally score custom statements.",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML experiment config.",
    )
    run_parser.add_argument(
        "--statements",
        default=None,
        help="Optional path to a plain-text file of custom statements.",
    )

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Fit raw mean-difference morality directions for every layer and measure stability.",
    )
    sweep_parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML experiment config.",
    )

    causal_parser = subparsers.add_parser(
        "causal",
        help="Run a prompt-only causal benchmark by ablating or steering a candidate morality direction.",
    )
    causal_parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML experiment config.",
    )
    causal_parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer override. Defaults to activations.layer from the config.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        result = run_morality_direction(
            config_path=args.config,
            statements_path=args.statements,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "sweep":
        result = run_layer_sweep(config_path=args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "causal":
        result = run_causal_benchmark(config_path=args.config, layer=args.layer)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    raise ValueError(f"Unhandled command '{args.command}'.")


if __name__ == "__main__":
    main()
