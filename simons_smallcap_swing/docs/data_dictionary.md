# Data Dictionary

Este documento define los datasets canonicos del proyecto, su granularidad, claves, columnas criticas y reglas de uso point-in-time.

## Reglas generales

- Todas las fechas de mercado usan calendario de trading de EE. UU.
- La clave temporal estandar es `date` (sesion) o `as_of_date` (fecha disponible real).
- Los joins entre fundamentals y precios se hacen por `as_of_date <= date`.
- Todas las tablas incluyen `source`, `ingestion_ts`, `run_id` y `data_version` cuando aplique.

## 1) `reference.trading_calendar`

Descripcion:
- Calendario maestro de sesiones de mercado.

Granularidad:
- 1 fila por sesion.

Columnas:
- `date` (`date`)
- `is_month_end` (`bool`)
- `is_quarter_end` (`bool`)
- `next_trade_date` (`date`)
- `prev_trade_date` (`date`)

## 2) `universe.daily_membership`

Descripcion:
- Universo invertible por fecha con filtros historicos.

Granularidad:
- 1 fila por `date` + `ticker`.

PK logica:
- `date`, `ticker`

Columnas:
- `date` (`date`)
- `ticker` (`string`)
- `perm_id` (`string`)
- `is_active` (`bool`)
- `price_close` (`float64`)
- `adv_20d_usd` (`float64`)
- `free_float_mcap` (`float64`)
- `sector` (`string`)
- `industry` (`string`)
- `universe_flag` (`bool`)
- `exclusion_reason` (`string|null`)

## 3) `price.daily_ohlcv`

Descripcion:
- Serie de precios y volumen ajustada para backtest.

Granularidad:
- 1 fila por `date` + `ticker`.

Columnas:
- `date` (`date`)
- `ticker` (`string`)
- `open` (`float64`)
- `high` (`float64`)
- `low` (`float64`)
- `close` (`float64`)
- `adj_close` (`float64`)
- `volume` (`int64`)
- `vwap_proxy` (`float64|null`)
- `split_factor` (`float64`)
- `dividend_cash` (`float64`)

## 4) `edgar.fundamentals_pit`

Descripcion:
- Fundamentales parseados desde XBRL con disponibilidad real.

Granularidad:
- 1 fila por `ticker` + `fact_tag` + `period_end` + `as_of_date`.

Columnas:
- `ticker` (`string`)
- `cik` (`string`)
- `fact_tag` (`string`)
- `value` (`float64`)
- `unit` (`string`)
- `fiscal_period` (`string`)
- `period_end` (`date`)
- `filing_date` (`date`)
- `acceptance_ts` (`datetime64[ns, UTC]`)
- `as_of_date` (`date`)
- `amendment_flag` (`bool`)

Regla PIT:
- En entrenamiento e inferencia diaria solo se permiten filas con `as_of_date <= date`.

## 5) `features.daily_matrix`

Descripcion:
- Matriz final de features para modelado.

Granularidad:
- 1 fila por `date` + `ticker`.

Columnas base:
- `date`, `ticker`
- `feature_*` (`float32` o `float64`)
- `feature_missing_count` (`int16`)
- `feature_qc_flag` (`bool`)

Familias:
- Fundamentales en nivel.
- Deltas QoQ y YoY.
- Microestructura (ADV, iliquidez, gap-risk, ATR, turnover).
- Transformaciones cross-sectional por sector y size.
- Interacciones no lineales seleccionadas.

## 6) `labels.forward_returns`

Descripcion:
- Targets de retorno futuro para horizontes de swing.

Granularidad:
- 1 fila por `date` + `ticker`.

Columnas:
- `date` (`date`)
- `ticker` (`string`)
- `ret_fwd_5d` (`float64`)
- `ret_fwd_10d` (`float64`)
- `ret_fwd_20d` (`float64`)
- `ret_fwd_10d_neutral_iwm` (`float64`)
- `ret_fwd_10d_neutral_sector` (`float64`)
- `label_rank_cs_10d` (`float64`)
- `label_valid_flag` (`bool`)

## 7) `models.daily_scores`

Descripcion:
- Predicciones por modelo y blend final.

Granularidad:
- 1 fila por `date` + `ticker` + `model_id`.

Columnas:
- `date` (`date`)
- `ticker` (`string`)
- `model_id` (`string`)
- `score_raw` (`float64`)
- `score_calibrated` (`float64`)
- `score_rank_cs` (`float64`)
- `regime_id` (`string|null`)
- `inference_run_id` (`string`)

## 8) `portfolio.target_positions`

Descripcion:
- Posiciones objetivo antes de ejecucion.

Granularidad:
- 1 fila por `date` + `ticker`.

Columnas:
- `date` (`date`)
- `ticker` (`string`)
- `side` (`int8`) con valores `{-1, 0, +1}`
- `target_weight` (`float64`)
- `max_weight_cap` (`float64`)
- `beta_contrib` (`float64`)
- `sector_exposure_contrib` (`float64`)

## 9) `execution.trades_fills`

Descripcion:
- Ordenes simuladas, fills y costes.

Granularidad:
- 1 fila por `trade_id`.

Columnas:
- `trade_id` (`string`)
- `date` (`date`)
- `ticker` (`string`)
- `qty` (`float64`)
- `arrival_px` (`float64`)
- `fill_px` (`float64`)
- `fees_bps` (`float64`)
- `slippage_bps` (`float64`)
- `impact_bps` (`float64`)
- `borrow_bps_day` (`float64|null`)
- `notional_usd` (`float64`)

## 10) `backtest.pnl_daily`

Descripcion:
- Resultado diario descompuesto para atribucion.

Granularidad:
- 1 fila por `date`.

Columnas:
- `date` (`date`)
- `gross_return` (`float64`)
- `cost_return` (`float64`)
- `borrow_return` (`float64`)
- `net_return` (`float64`)
- `turnover` (`float64`)
- `gross_exposure` (`float64`)
- `net_exposure` (`float64`)
- `active_names` (`int32`)
- `drawdown` (`float64`)

## Reglas minimas de integridad

- Unicidad de clave por dataset.
- Sin duplicados `date+ticker` en matrices diarias.
- Tipos estrictos por contrato.
- Porcentaje maximo de faltantes por feature bajo umbral.
- Sin valores infinitos ni NaN no tratados en columnas de modelado.
