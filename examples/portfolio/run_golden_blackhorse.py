"""Utility script to run the Golden Black Horse backtest."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

import qlib
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.evaluate import risk_analysis
from qlib.data.dataset import Dataset
from qlib.utils import init_instance_by_config


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _instantiate_dataset(cfg: Dict[str, Any]) -> Dataset:
    dataset_cfg = copy.deepcopy(cfg)
    dataset = init_instance_by_config(dataset_cfg, accept_types=Dataset)
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset configuration must produce a Dataset instance")
    return dataset


def _prepare_strategy(cfg: Dict[str, Any], dataset: Dataset) -> Dict[str, Any]:
    strategy_cfg = copy.deepcopy(cfg)
    strategy_kwargs = strategy_cfg.setdefault("kwargs", {})
    strategy_kwargs["dataset"] = dataset
    return strategy_cfg


def _prepare_executor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg is not None:
        return copy.deepcopy(cfg)
    return {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }


def _save_results(output: Path, report: pd.DataFrame, indicator: pd.DataFrame) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "portfolio_report.csv")
    indicator.to_csv(output / "trade_indicator.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Golden Black Horse backtest")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to store portfolio and position reports",
    )
    args = parser.parse_args()

    config = _load_yaml(args.config)

    qlib.init(**config["qlib_init"])

    dataset = _instantiate_dataset(config["dataset"])
    strategy_cfg = _prepare_strategy(config["strategy"], dataset)
    executor_cfg = _prepare_executor(config.get("executor"))

    backtest_cfg = config["backtest"]
    account_cfg = backtest_cfg.get("account", {})
    account_cash = account_cfg.get("init_cash", 1_000_000)
    exchange_kwargs = backtest_cfg.get("exchange_kwargs", {})

    portfolio_dict, indicator_dict = qlib_backtest(
        start_time=backtest_cfg["start_time"],
        end_time=backtest_cfg["end_time"],
        strategy=strategy_cfg,
        executor=executor_cfg,
        benchmark=backtest_cfg.get("benchmark"),
        account=account_cash,
        exchange_kwargs=exchange_kwargs,
    )

    freq_key, (report_df, _) = next(iter(portfolio_dict.items()))
    trade_indicator = indicator_dict[freq_key][0]

    print("=== Portfolio summary ===")
    print(report_df.tail())

    daily_returns = report_df["return"].dropna()
    if not daily_returns.empty:
        risk_df = risk_analysis(daily_returns, freq="day")
        print("\n=== Risk analysis ===")
        print(risk_df)

    print("\n=== Trade indicator (head) ===")
    print(trade_indicator.head())

    if args.output:
        _save_results(args.output, report_df, trade_indicator)
        print(f"Saved reports to {args.output.resolve()}")


if __name__ == "__main__":
    main()
