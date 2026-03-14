# Local Price Source (fetch_prices v2)

This folder is the preferred input location for `local_file` ingestion mode.

## Supported files
- `*.csv`
- `*.parquet`

## Minimum required columns (aliases supported)
- `date` (`trade_date`, `datetime`, `timestamp`)
- `ticker` (`symbol`)
- `open`
- `high`
- `low`
- `close`
- `volume`

## Optional columns
- `adj_close_raw` (`adj_close`, `adjclose`, `adjusted_close`)
- `vendor`

The fetcher normalizes columns, maps `(date, ticker)` to `instrument_id` using PIT universe/reference mapping, drops unmappable rows with traceability, and writes:
- `data/price/raw_prices.parquet`
- `data/price/raw_prices.ingestion_report.json`
