# Portfolio Optimization Strategy

## Introduction

In `qlib/examples/benchmarks` we have various **alpha** models that predict
the stock returns. We also use a simple rule based `TopkDropoutStrategy` to
evaluate the investing performance of these models. However, such a strategy
is too simple to control the portfolio risk like correlation and volatility.

To this end, an optimization based strategy should be used to for the
trade-off between return and risk. In this doc, we will show how to use
`EnhancedIndexingStrategy` to maximize portfolio return while minimizing
tracking error relative to a benchmark.


## Preparation

We use China stock market data for our example.

1. Prepare CSI300 weight:

   ```bash
   wget https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/csi300_weight.zip
   unzip -d ~/.qlib/qlib_data/cn_data csi300_weight.zip
   rm -f csi300_weight.zip
   ```
   NOTE:  We don't find any public free resource to get the weight in the benchmark. To run the example, we manually create this weight data.

2. Prepare risk model data:

   ```bash
   python prepare_riskdata.py
   ```

Here we use a **Statistical Risk Model** implemented in `qlib.model.riskmodel`.
However users are strongly recommended to use other risk models for better quality:
* **Fundamental Risk Model** like MSCI BARRA
* [Deep Risk Model](https://arxiv.org/abs/2107.05201)


## End-to-End Workflow

You can finish workflow with `EnhancedIndexingStrategy` by running
`qrun config_enhanced_indexing.yaml`.

In this config, we mainly changed the strategy section compared to
`qlib/examples/benchmarks/workflow_config_lightgbm_Alpha158.yaml`.

## Golden Black Horse 日频策略

`config_golden_blackhorse.yaml` 展示了如何在全 A 股上复现“黄金黑马”模式：

1. **准备数据**：可使用 `scripts/data_collector/akshare_cn_stock` 目录下的脚本通过 AkShare 下载日线行情，并配合 `scripts/dump_bin.py` 导入 Qlib 数据目录。
2. **运行回测**：

   ```bash
   python examples/portfolio/run_golden_blackhorse.py \
     --config examples/portfolio/config_golden_blackhorse.yaml \
     --output outputs/golden_blackhorse
   ```

   脚本会自动加载配置文件中的数据集、策略与执行器设置，完成回测并输出组合表现及风险指标；如指定 `--output`，同时会保存投资组合报告与仓位明细。

策略核心逻辑在 `qlib/contrib/strategy/golden_blackhorse.py`，使用 `GoldenBlackhorseSignal` 处理器（定义于 `qlib/data/dataset/processor.py`）计算 3 日 K 线特征并生成买入信号。策略实现支持：

- 信号触发后的分批建仓（初始 10% 仓位 + 回撤加仓）；
- 基于形态低点的动态止损，以及 15%–25% 的可调止盈目标；
- 买入价格容忍区间与信号失效天数的参数化控制。

通过调整配置文件即可快速探索不同的止盈/止损、加仓阈值或回测区间。
