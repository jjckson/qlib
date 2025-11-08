# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.decision import OrderHelper
from qlib.data.dataset import Dataset, DatasetH
from qlib.strategy.base import BaseStrategy
from qlib.utils import init_instance_by_config


@dataclass
class _PendingEntry:
    target_price: float
    stop_price: Optional[float]
    take_profit_pct: float
    signal_date: pd.Timestamp
    days_waited: int = 0


@dataclass
class _ActivePosition:
    entry_price: float
    stop_price: Optional[float]
    take_profit_pct: float
    shares: float
    signal_date: pd.Timestamp
    added: bool = False

    def update_entry(self, additional_shares: float, trade_price: float) -> None:
        if additional_shares <= 0:
            return
        total_shares = self.shares + additional_shares
        if total_shares <= 0:
            return
        self.entry_price = (self.entry_price * self.shares + trade_price * additional_shares) / total_shares
        self.shares = total_shares
        self.added = True

    @property
    def take_profit_price(self) -> float:
        return self.entry_price * (1 + self.take_profit_pct)


class GoldenBlackHorseStrategy(BaseStrategy):
    """Rule-based implementation of the Golden Black Horse pattern."""

    def __init__(
        self,
        *,
        dataset: Union[Dataset, DatasetH, Dict],
        buy_signal_col: str = "golden_blackhorse_signal",
        body_col: str = "golden_blackhorse_body_pct",
        fast_entry_col: str = "golden_blackhorse_fast_entry",
        slow_entry_col: str = "golden_blackhorse_slow_entry",
        pattern_low_col: str = "golden_blackhorse_pattern_low",
        body_threshold: float = 0.06,
        stop_loss_buffer: float = 0.03,
        take_profit: Union[float, Tuple[float, float], List[float]] = 0.2,
        initial_pos: float = 0.1,
        add_pos: float = 0.1,
        add_pullback: float = 0.03,
        entry_price_tolerance: float = 0.01,
        entry_expire_days: int = 2,
        max_pending: int = 200,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ) -> None:
        super().__init__(trade_exchange=trade_exchange, level_infra=level_infra, common_infra=common_infra, **kwargs)

        self.dataset: Dataset = init_instance_by_config(dataset, accept_types=Dataset)
        if not isinstance(self.dataset, DatasetH):
            raise TypeError("GoldenBlackHorseStrategy requires a DatasetH instance")

        self.buy_signal_col = buy_signal_col
        self.body_col = body_col
        self.fast_entry_col = fast_entry_col
        self.slow_entry_col = slow_entry_col
        self.pattern_low_col = pattern_low_col
        self.body_threshold = body_threshold
        self.stop_loss_buffer = stop_loss_buffer
        self.initial_pos = initial_pos
        self.add_pos = add_pos
        self.add_pullback = add_pullback
        self.entry_price_tolerance = entry_price_tolerance
        self.entry_expire_days = entry_expire_days
        self.max_pending = max_pending

        if isinstance(take_profit, (tuple, list)):
            if len(take_profit) != 2:
                raise ValueError("take_profit should be a float or a pair [min, max]")
            self.take_profit_pct = float(np.mean(take_profit))
        else:
            self.take_profit_pct = float(take_profit)

        self._order_helper: Optional[OrderHelper] = None
        self._daily_frames: Dict[pd.Timestamp, pd.DataFrame] = {}
        self._dates: List[pd.Timestamp] = []
        self._date_to_idx: Dict[pd.Timestamp, int] = {}
        self._pending_entries: Dict[str, _PendingEntry] = {}
        self._active_positions: Dict[str, _ActivePosition] = {}

    def reset(self, *, level_infra=None, common_infra=None, outer_trade_decision=None, **kwargs) -> None:
        super().reset(level_infra=level_infra, common_infra=common_infra, outer_trade_decision=outer_trade_decision, **kwargs)

        self._order_helper = OrderHelper(self.trade_exchange)
        self._pending_entries.clear()
        self._active_positions.clear()
        self._prepare_data()

    def _prepare_data(self) -> None:
        start_time, end_time = self.trade_calendar.get_all_time()
        slice_obj = slice(pd.Timestamp(start_time), pd.Timestamp(end_time))
        data = self.dataset.prepare(slice_obj)

        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("GoldenBlackHorseStrategy expects dataset with multi-index columns")

        feature_cols = data.columns.get_level_values(0) == "feature"
        feature_view = data.loc[:, feature_cols]
        feature_view.columns = feature_view.columns.get_level_values(-1)

        if not isinstance(feature_view.index, pd.MultiIndex):
            raise ValueError("GoldenBlackHorseStrategy expects multi-index index with datetime and instrument")

        time_level = "datetime" if "datetime" in feature_view.index.names else feature_view.index.names[0]
        grouped = feature_view.groupby(level=time_level, sort=True)

        self._daily_frames = {}
        for idx, frame in grouped:
            ts = pd.Timestamp(idx)
            self._daily_frames[ts] = frame.droplevel(time_level)

        self._dates = sorted(self._daily_frames.keys())
        self._date_to_idx = {date: idx for idx, date in enumerate(self._dates)}

    def _get_prev_date(self, current: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = self._date_to_idx.get(current)
        if idx is None or idx == 0:
            return None
        return self._dates[idx - 1]

    def _update_from_execution(self, execute_result: Iterable) -> None:
        for order, _, _, trade_price in execute_result or []:
            if order.deal_amount <= 0:
                continue

            code = order.stock_id
            if order.direction == Order.BUY:
                pending = self._pending_entries.get(code)
                if code not in self._active_positions:
                    stop_price = pending.stop_price if pending is not None else None
                    take_profit_pct = pending.take_profit_pct if pending is not None else self.take_profit_pct
                    signal_date = pending.signal_date if pending is not None else pd.Timestamp(order.start_time)
                    self._active_positions[code] = _ActivePosition(
                        entry_price=trade_price,
                        stop_price=stop_price,
                        take_profit_pct=take_profit_pct,
                        shares=order.deal_amount,
                        signal_date=signal_date,
                    )
                else:
                    self._active_positions[code].update_entry(order.deal_amount, trade_price)

                if pending is not None:
                    self._pending_entries.pop(code, None)
            else:
                if code in self._active_positions:
                    position = self._active_positions[code]
                    position.shares = max(position.shares - order.deal_amount, 0)
                    if position.shares <= 0:
                        self._active_positions.pop(code, None)

    def generate_trade_decision(self, execute_result: Iterable = None):
        if execute_result:
            self._update_from_execution(execute_result)

        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_date = pd.Timestamp(trade_start_time)

        if current_date not in self._daily_frames:
            return TradeDecisionWO(order_list=[], strategy=self)

        prev_date = self._get_prev_date(current_date)
        prev_frame = self._daily_frames.get(prev_date)
        current_frame = self._daily_frames[current_date]

        if prev_frame is not None:
            self._collect_pending(prev_frame, prev_date)

        orders: List[Order] = []
        total_value = float(self.trade_position.calculate_value())
        cash_available = float(self.trade_position.get_cash())

        orders.extend(self._prepare_sell_orders(prev_frame, trade_start_time, trade_end_time))

        if cash_available > 0:
            buy_orders, remaining_cash = self._prepare_buy_orders(
                current_frame,
                trade_start_time,
                trade_end_time,
                total_value,
                cash_available,
            )
            orders.extend(buy_orders)
            cash_available = remaining_cash

        return TradeDecisionWO(order_list=orders, strategy=self)

    def _collect_pending(self, prev_frame: pd.DataFrame, prev_date: pd.Timestamp) -> None:
        if self.buy_signal_col not in prev_frame.columns:
            return

        signals = prev_frame[self.buy_signal_col] > 0.5
        for code, triggered in signals[signals].items():
            if code in self._pending_entries or code in self._active_positions:
                continue
            if len(self._pending_entries) >= self.max_pending:
                break

            target_price = self._select_entry_price(prev_frame, code)
            if target_price is None or not np.isfinite(target_price):
                continue

            stop_price = self._compute_stop_price(prev_frame, code)
            pending = _PendingEntry(
                target_price=float(target_price),
                stop_price=None if stop_price is None else float(stop_price),
                take_profit_pct=self.take_profit_pct,
                signal_date=prev_date,
            )
            self._pending_entries[code] = pending

        for code in list(self._pending_entries.keys()):
            if code in self._active_positions:
                self._pending_entries.pop(code, None)

    def _select_entry_price(self, frame: pd.DataFrame, code: str) -> Optional[float]:
        body = frame.get(self.body_col)
        fast = frame.get(self.fast_entry_col)
        slow = frame.get(self.slow_entry_col)

        if body is None or fast is None or slow is None:
            return None

        body_val = body.get(code)
        fast_val = fast.get(code)
        slow_val = slow.get(code)

        if pd.isna(body_val):
            return slow_val
        if body_val >= self.body_threshold:
            return fast_val
        return slow_val

    def _compute_stop_price(self, frame: pd.DataFrame, code: str) -> Optional[float]:
        if self.pattern_low_col not in frame.columns:
            return None
        base_low = frame[self.pattern_low_col].get(code)
        if pd.isna(base_low):
            return None
        return base_low * (1 - self.stop_loss_buffer)

    def _prepare_sell_orders(
        self,
        prev_frame: Optional[pd.DataFrame],
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> List[Order]:
        if prev_frame is None:
            return []

        orders: List[Order] = []
        for code, position in list(self._active_positions.items()):
            if code not in prev_frame.index:
                continue

            prev_low = prev_frame.get("$low", pd.Series()).get(code)
            prev_high = prev_frame.get("$high", pd.Series()).get(code)
            prev_close = prev_frame.get("$close", pd.Series()).get(code)

            exit_flag = False
            if position.stop_price is not None:
                if (pd.notna(prev_low) and prev_low <= position.stop_price) or (
                    pd.notna(prev_close) and prev_close <= position.stop_price
                ):
                    exit_flag = True

            if not exit_flag:
                take_price = position.take_profit_price
                if (pd.notna(prev_high) and prev_high >= take_price) or (
                    pd.notna(prev_close) and prev_close >= take_price
                ):
                    exit_flag = True

            if not exit_flag:
                continue

            amount = self.trade_position.get_stock_amount(code)
            if amount <= 0:
                continue

            orders.append(
                self._order_helper.create(
                    code=code,
                    amount=amount,
                    direction=OrderDir.SELL,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            )

        return orders

    def _prepare_buy_orders(
        self,
        current_frame: pd.DataFrame,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        total_value: float,
        cash_available: float,
    ) -> Tuple[List[Order], float]:
        orders: List[Order] = []

        add_orders = self._prepare_add_orders(
            current_frame,
            trade_start_time,
            trade_end_time,
            total_value,
            cash_available,
        )
        orders.extend(add_orders)

        add_cost = 0.0
        open_prices = current_frame.get("$open", pd.Series())
        for order in add_orders:
            price = open_prices.get(order.stock_id, np.nan)
            if pd.notna(price):
                add_cost += order.amount * price
        remaining_cash = cash_available - add_cost

        buy_orders: List[Order] = []
        price_series = current_frame.get("$open")

        for code, pending in list(self._pending_entries.items()):
            open_price = price_series.get(code) if price_series is not None else np.nan
            if pd.isna(open_price) or open_price <= 0:
                pending.days_waited += 1
                continue

            if remaining_cash <= 0:
                pending.days_waited += 1
                continue

            tolerance_price = pending.target_price * (1 + self.entry_price_tolerance)
            if open_price > tolerance_price:
                pending.days_waited += 1
                if pending.days_waited > self.entry_expire_days:
                    self._pending_entries.pop(code, None)
                continue

            order_value = min(self.initial_pos * total_value, remaining_cash)
            if order_value <= 0:
                pending.days_waited += 1
                continue

            amount = order_value / open_price
            if amount <= 0:
                pending.days_waited += 1
                continue

            buy_orders.append(
                self._order_helper.create(
                    code=code,
                    amount=amount,
                    direction=OrderDir.BUY,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            )
            remaining_cash -= order_value

        for code, pending in list(self._pending_entries.items()):
            if pending.days_waited > self.entry_expire_days:
                self._pending_entries.pop(code, None)

        orders.extend(buy_orders)
        return orders, remaining_cash

    def _prepare_add_orders(
        self,
        current_frame: pd.DataFrame,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        total_value: float,
        cash_available: float,
    ) -> List[Order]:
        if self.add_pos <= 0:
            return []

        prev_date = self._get_prev_date(pd.Timestamp(trade_start_time))
        prev_frame = self._daily_frames.get(prev_date)
        if prev_frame is None:
            return []

        orders: List[Order] = []
        open_prices = current_frame.get("$open", pd.Series())
        for code, position in self._active_positions.items():
            if position.added:
                continue
            if code not in prev_frame.index:
                continue

            prev_low = prev_frame.get("$low", pd.Series()).get(code)
            if pd.isna(prev_low):
                continue

            trigger_price = position.entry_price * (1 - self.add_pullback)
            if prev_low > trigger_price:
                continue

            open_price = open_prices.get(code, np.nan)
            if pd.isna(open_price) or open_price <= 0:
                continue

            order_value = min(self.add_pos * total_value, cash_available)
            if order_value <= 0:
                continue

            amount = order_value / open_price
            if amount <= 0:
                continue

            orders.append(
                self._order_helper.create(
                    code=code,
                    amount=amount,
                    direction=OrderDir.BUY,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            )
            cash_available -= order_value

        return orders
