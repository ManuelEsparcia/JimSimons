# Schemas and Contracts

Este documento define contratos de datos y artefactos para asegurar consistencia, trazabilidad y reproducibilidad.

## Convenciones globales

- Timezone canonico: `UTC`.
- Fechas de trading como `date`.
- Identidad de instrumento: `ticker` operativo, `perm_id` historico.
- Identidad de corrida: `run_id`, `config_hash`, `code_hash`.

## Convencion de nombres

- Datasets en `snake_case`.
- Features por prefijo:
- `fnd_` para fundamentales.
- `mic_` para microestructura.
- `cs_` para transformaciones cross-sectional.
- `int_` para interacciones.
- Labels prefijadas con `y_`.

## Contrato parquet

- Particionado recomendado por `date`.
- Compresion `zstd` o `snappy`.
- Metadatos minimos:
- `dataset_name`
- `schema_version`
- `source`
- `created_at_utc`
- `run_id`
- `row_count`

## Versionado de esquema

- Formato: `major.minor.patch`.
- `major`: cambio incompatible.
- `minor`: cambio compatible.
- `patch`: correccion sin cambio de contrato.

## Invariantes por capa

### `data/`

- Sin filas futuras para la fecha de proceso.
- Mapping ticker-perm_id sin ambiguedad por fecha.
- Corporate actions consistentes.

### `features/`

- Sin leakage temporal.
- Estandarizacion robusta por fecha, no global.
- Missingness reportada por feature y dia.

### `labels/`

- Targets estrictamente forward.
- Purga y embargo configurables.

### `models/`

- Hiperparametros persistidos.
- Ventanas temporales de entrenamiento registradas.
- Cobertura y calibracion auditables.

### `backtest/`

- Accounting determinista.
- Desglose de retorno bruto, costes, impacto y borrow.
- Reproducibilidad con misma entrada y config.

## Contrato de experimento

Todo experimento debe registrar:
- `experiment_id`
- `hypothesis_id`
- `dataset_snapshot_id`
- `config_hash`
- `code_hash`
- ventanas train/test
- metricas OOS
- artefactos (scores, posiciones, pnl, reportes)

## Checklist de aceptacion

- Tipos validados contra contrato.
- Unicidad de PK logica.
- Densidad de datos en rangos esperados.
- Drift registrado frente a baseline.
- Compatibilidad con capas consumidoras.
