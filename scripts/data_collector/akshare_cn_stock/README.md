# AkShare 日线数据采集

该工具脚本使用 [AkShare](https://akshare.xyz/) 抓取全 A 股日线行情，并生成与 Qlib `dump_bin.py` 脚本兼容的 CSV 文件。可配合 `scripts/dump_bin.py` 快速导入 Qlib 本地数据目录。

## 使用步骤

1. **下载日线数据（前复权）**

   ```bash
   python scripts/data_collector/akshare_cn_stock/download_akshare_daily.py \
     --output ~/.qlib/stock_data/source/cn_akshare \
     --start 2010-01-01 \
     --end 2024-12-31 \
     --adjust qfq
   ```

   - `--adjust` 支持 `qfq`（前复权）、`hfq`（后复权）和 `none`（不复权）。
   - 输出目录下会以 `SZ000001.csv`/`SH600000.csv` 的形式保存个股数据。

2. **转换为 Qlib 二进制格式**

   ```bash
   python scripts/dump_bin.py dump_all \
     --data_path ~/.qlib/stock_data/source/cn_akshare \
     --qlib_dir ~/.qlib/qlib_data/cn_data \
     --freq day \
     --file_suffix .csv \
     --exclude_fields symbol \
     --date_field_name date
   ```

   如需跳过成交额等字段，可通过 `--include_fields` 指定需要保留的列。

3. **初始化 Qlib 数据目录**

   ```bash
   qlib init --provider_uri ~/.qlib/qlib_data/cn_data --region cn
   ```

完成上述流程后，即可在 `examples/portfolio/config_golden_blackhorse.yaml` 中直接引用本地数据进行回测。
