from __future__ import annotations

import argparse
import json

from mech_interp.pipeline import (
    run_control_causal_benchmark,
    run_causal_benchmark,
    run_layer_sweep,
    run_morality_direction,
    run_random_direction_causal_benchmark,
    run_orthogonalized_causal_benchmark,
    run_orthogonalized_control,
)


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

    orthogonalize_parser = subparsers.add_parser(
        "orthogonalize",
        help="Fit a control direction and remove its linear component from the primary direction.",
    )
    orthogonalize_parser.add_argument(
        "--config",
        required=True,
        help="Path to the primary TOML experiment config.",
    )
    orthogonalize_parser.add_argument(
        "--control-config",
        required=True,
        help="Path to the control TOML experiment config.",
    )
    orthogonalize_parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer override. Defaults to activations.layer from the primary config.",
    )

    orthogonalized_causal_parser = subparsers.add_parser(
        "orthogonalized-causal",
        help="Run the causal benchmark using the primary direction after removing a control direction.",
    )
    orthogonalized_causal_parser.add_argument(
        "--config",
        required=True,
        help="Path to the primary TOML experiment config.",
    )
    orthogonalized_causal_parser.add_argument(
        "--control-config",
        required=True,
        help="Path to the control TOML experiment config.",
    )
    orthogonalized_causal_parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer override. Defaults to activations.layer from the primary config.",
    )

    control_causal_parser = subparsers.add_parser(
        "control-causal",
        help="Run the causal benchmark on the primary prompts using a direction fit from a control dataset.",
    )
    control_causal_parser.add_argument(
        "--config",
        required=True,
        help="Path to the primary TOML experiment config.",
    )
    control_causal_parser.add_argument(
        "--control-config",
        required=True,
        help="Path to the control TOML experiment config.",
    )
    control_causal_parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer override. Defaults to activations.layer from the primary config.",
    )

    random_causal_parser = subparsers.add_parser(
        "random-causal",
        help="Run the causal benchmark with many random-direction removals as a null control.",
    )
    random_causal_parser.add_argument(
        "--config",
        required=True,
        help="Path to the primary TOML experiment config.",
    )
    random_causal_parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer override. Defaults to activations.layer from the primary config.",
    )
    random_causal_parser.add_argument(
        "--num-random-directions",
        type=int,
        default=32,
        help="Number of random unit vectors to test.",
    )
    random_causal_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for drawing null directions.",
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

    if args.command == "orthogonalize":
        result = run_orthogonalized_control(
            config_path=args.config,
            control_config_path=args.control_config,
            layer=args.layer,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "orthogonalized-causal":
        result = run_orthogonalized_causal_benchmark(
            config_path=args.config,
            control_config_path=args.control_config,
            layer=args.layer,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "control-causal":
        result = run_control_causal_benchmark(
            config_path=args.config,
            control_config_path=args.control_config,
            layer=args.layer,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "random-causal":
        result = run_random_direction_causal_benchmark(
            config_path=args.config,
            layer=args.layer,
            num_random_directions=args.num_random_directions,
            random_seed=args.random_seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    raise ValueError(f"Unhandled command '{args.command}'.")


if __name__ == "__main__":
    main()
