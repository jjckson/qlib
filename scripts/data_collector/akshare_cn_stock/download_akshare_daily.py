"""Download daily A-share data from AkShare and store it in CSV format."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger("akshare_downloader")


def _normalize_dates(value: str) -> str:
    return value.replace("-", "")


def _convert_symbol(code: str) -> str:
    return ("SH" if code.startswith("6") else "SZ") + code


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "复权因子": "factor",
    }
    renamed = frame.rename(columns=rename_map)
    for column in ["open", "close", "high", "low", "volume", "amount", "factor"]:
        if column not in renamed:
            renamed[column] = 1.0 if column == "factor" else 0.0
    renamed["volume"] = renamed["volume"] * 100.0
    return renamed[["date", "open", "high", "low", "close", "volume", "amount", "factor"]]


def _download_single(code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    return _rename_columns(raw)


def _iter_symbols() -> Iterable[str]:
    spot = ak.stock_zh_a_spot_em()
    return spot["代码"].astype(str).tolist()


def download(output_dir: Path, start: str, end: str, adjust: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = _iter_symbols()
    LOGGER.info("Found %d symbols", len(symbols))
    for code in tqdm(symbols, desc="download", unit="stock"):
        try:
            data = _download_single(code, start, end, adjust)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to download %s: %s", code, exc)
            continue
        if data.empty:
            continue
        data.insert(0, "symbol", _convert_symbol(code))
        target = output_dir / f"{_convert_symbol(code)}.csv"
        data.to_csv(target, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AkShare A-share daily bars")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save CSV files")
    parser.add_argument("--start", type=str, default="2005-01-01", help="Start date, e.g. 2010-01-01")
    parser.add_argument("--end", type=str, default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--adjust",
        type=str,
        default="qfq",
        choices=["qfq", "hfq", "none"],
        help="AkShare adjustment method",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    start_date = _normalize_dates(args.start)
    end_date = _normalize_dates(args.end)

    download(args.output, start_date, end_date, args.adjust)


if __name__ == "__main__":
    main()
