# Roadmap

Plan de 10 semanas para rellenar, validar y empezar a experimentar con los `.py` de `simons_smallcap_swing`.

## Propósito del documento

Ordenar el desarrollo del repositorio en una secuencia realista: primero contratos, datos y validación; después pipeline mínimo end-to-end; y solo más tarde datos avanzados, modelos complejos, riesgo, capacidad y operación diaria. Cada etapa equivale a una semana de trabajo y deja un bloque ejecutable, testeado y listo para la siguiente capa.

## Principios de ejecución

- No ejecutar 100 scripts manualmente: la meta es converger hacia tres entrypoints visibles: `research/research_pipeline.py`, `backtest/run_walkforward.py` y `ops/daily_pipeline.py`.
- Cada módulo nuevo debe nacer con mini-tests: smoke test, invariantes, casos borde e integración con el módulo anterior/siguiente.
- No empezar por GBDT, ensembles ni optimización avanzada: primero un circuito completo y auditable.
- La regla de cierre de semana es simple: el bloque funciona, los tests mínimos pasan y deja outputs con schema estable.

## Vista rápida de las 10 semanas

| Semana | Bloque principal | Qué se construye | Resultado esperado |
|---|---|---|---|
| 1 | `simons_core` + tests base | Contratos, schemas, calendario, IO, fixtures | Infraestructura común estable |
| 2 | `data/universe` + `data/price` | Universo histórico, precios raw, QC y ajustes | Datos PIT mínimos usables |
| 3 | `labels` + `leakage` | Splits purgados, labels y auditoría anti-fuga | Targets válidos para modelar |
| 4 | Features MVP | Features de precio/microestructura y feature store | Matriz `X` reproducible |
| 5 | Baselines + inference | Ridge/Logistic/ElasticNet + predict + registry | Primer modelo entrenable |
| 6 | Portfolio + execution + backtest | Sizing, constraints, costes, fills y engine | Pipeline monetizable mínimo |
| 7 | Validation | Walk-forward bias, suite, PBO y stress | Gates de promoción operativos |
| 8 | EDGAR + borrow + enrichments | Fundamentales PIT, borrow proxy, labels extra | Datos avanzados integrados |
| 9 | GBDT + regimes + ensemble | Modelos más potentes y gating por regímenes | Comparación seria contra baseline |
| 10 | Risk + optimizer + ops/research | Riesgo, capacidad, daily pipeline y tracking | Versión 1 del framework completo |

---

## Semana 1 · Núcleo contractual y laboratorio de tests

### Objetivo

Dejar listas las piezas que dan coherencia al resto del repositorio: geometría temporal, schemas, contratos comunes, IO básico y fixtures deterministas para testear sin depender todavía de datos reales.

### Orden de ficheros a tocar

- `simons_core/calendar.py`
- `simons_core/schemas.py`
- `simons_core/interfaces.py`
- `simons_core/logging.py`
- `simons_core/io/paths.py`
- `simons_core/io/parquet_store.py`
- `simons_core/io/cache.py`
- `simons_core/math/stats.py`
- `simons_core/math/robust.py`
- `simons_core/math/optimization.py`
- `tests/conftest.py`

### Qué experimentar esa semana

- Definir el objeto/contrato base de run: `run_id`, `logical_date`, `seed`, `artifacts_dir`, `config_hash`.
- Acordar naming de esquemas para `universe`, `prices`, `labels`, `features`, `predictions`, `orders` y `fills`.
- Dejar un mini ejemplo que cree datos sintéticos y los persista/recupere desde `parquet_store`.
- Verificar que cualquier módulo futuro pueda enchufarse a interfaces comunes sin import caótico.

### Mini-tests mínimos por módulo

- Smoke test de importación de cada submódulo del core.
- Test de determinismo: mismo seed -> mismo output sintético.
- Test de schemas: columnas obligatorias, tipos, PK y orden temporal.
- Test de IO: escribir parquet, leerlo y recuperar exactamente los mismos registros.

### Criterio de cierre de etapa

La etapa cierra cuando el repo tiene una base estable para correr `pytest`, crear fixtures sintéticos y definir contratos comunes sin ambigüedad.

---

## Semana 2 · Datos mínimos: universo histórico y precios PIT

### Objetivo

Construir el primer bloque de verdad histórica: universo invertible por fecha, precios raw, controles de calidad y series ajustadas utilizables por features y labels.

### Orden de ficheros a tocar

- `data/universe/build_universe.py`
- `data/universe/universe_qc.py`
- `data/universe/survivorship.py`
- `data/universe/corporate_actions.py`
- `data/price/fetch_prices.py`
- `data/price/qc_prices.py`
- `data/price/adjust_prices.py`
- `data/price/market_proxies.py`

### Qué experimentar esa semana

- Generar un universo pequeño pero realista, por ejemplo 50–200 nombres con altas y bajas históricas.
- Cargar OHLCV raw y comparar dos convenciones de ajuste: solo split vs split + dividendos según el diseño del repo.
- Probar cómo se comporta `adjust_prices` ante splits, delistings y huecos de datos.
- Construir `market_proxies` mínimos para contexto de régimen y normalización.

### Mini-tests mínimos por módulo

- Test PIT: un activo no puede existir en el universo antes de su fecha de alta.
- Test de no survivorship: un activo excluido o delistado deja de aparecer a partir de su baja.
- Test de precios: `open/high/low/close` coherentes, volumen no negativo y fechas únicas por ticker.
- Test de ajustes: la serie ajustada no introduce discontinuidades artificiales en eventos corporativos conocidos.

### Criterio de cierre de etapa

La etapa cierra cuando puedes pedir “universo + precios ajustados” para una fecha y obtener una tabla limpia, estable y sin sesgos obvios de supervivencia.

---

## Semana 3 · Splits válidos, labels y auditoría anti-leakage

### Objetivo

Crear el bloque que impide el autoengaño: particiones temporales correctas, labels forward alineados con el horizonte operativo y auditoría explícita de fuga de información.

### Orden de ficheros a tocar

- `labels/purged_splits.py`
- `labels/build_labels.py`
- `labels/label_qc.py`
- `validation/leakage_audit.py`

### Qué experimentar esa semana

- Probar dos o tres horizontes de label razonables para swing trading, por ejemplo `3d`, `5d`, `10d`.
- Comparar target bruto frente a target neutralizado o normalizado, aunque todavía sea solo a nivel exploratorio.
- Ver cómo cambian los labels con distintas convenciones de costes incorporados en el target.
- Dejar claro el `decision_ts` y el `forward window` exactos que usarás más adelante.

### Mini-tests mínimos por módulo

- Test de purga/embargo: ninguna observación de train puede contaminar el horizonte de test.
- Test de label alignment: el retorno forward debe empezar después del `decision_ts` y no antes.
- Test de cobertura: el porcentaje de labels nulos o extremos queda monitorizado.
- Test de `leakage_audit` con casos sintéticos trucados que deban fallar de forma explícita.

### Criterio de cierre de etapa

La etapa cierra cuando ya existe una función reproducible que produce `train/test` válidos y labels forward sin usar información futura.

---

## Semana 4 · Features MVP para poner a correr el primer pipeline

### Objetivo

Construir la primera matriz `X` útil, sin intentar todavía capturar todo el edge del sistema. La prioridad es tener features causales, auditables y enlazadas al universo y precios ajustados.

### Orden de ficheros a tocar

- `features/microstructure.py`
- `features/cross_sectional.py`
- `features/build_features.py`
- `features/feature_qc.py`
- `features/feature_store.py`

### Qué experimentar esa semana

- Crear un MVP con features de retornos, volatilidad, volumen relativo, rango, gap, reversión y momentum corto.
- Comparar ventanas cortas vs medias para ver si la matriz se vuelve demasiado ruidosa o demasiado lenta.
- Guardar snapshots reproducibles en `feature_store` para no recalcular todo cada vez.
- Medir cobertura y estabilidad temporal de cada feature antes de pensar en modelar más fuerte.

### Mini-tests mínimos por módulo

- Smoke test de `build_features` sobre un universo pequeño.
- Test de causalidad: ninguna feature usa valores posteriores al `decision_ts`.
- Test de schemas y `NaN`: columnas esperadas, rango numérico razonable y missing controlado.
- Test de integración: `feature_store` guarda y devuelve exactamente la misma matriz.

### Criterio de cierre de etapa

La etapa cierra cuando ya puedes construir un dataset tabular `X/y/metadata` por fecha con features básicas listas para entrenamiento.

---

## Semana 5 · Baselines e inferencia: primer modelo serio

### Objetivo

Tener un baseline canónico entrenable y comparable. Antes de tocar GBDT o ensembles, necesitas un punto de referencia limpio, barato y reproducible.

### Orden de ficheros a tocar

- `models/baselines/train_ridge.py`
- `models/inference/predict.py`
- `models/inference/model_registry.py`
- `models/baselines/train_logistic.py`
- `models/baselines/train_elasticnet.py`

### Qué experimentar esa semana

- Empezar con `Ridge` como baseline principal y comparar después con `Logistic` o `ElasticNet` según el tipo de target.
- Evaluar estabilidad de coeficientes, sensibilidad a regularización y consistencia OOS básica.
- Guardar artefactos de modelo, métricas y metadata en `model_registry` desde el primer día.
- Ver si el baseline aprende algo mínimamente consistente antes de sofisticar las señales.

### Mini-tests mínimos por módulo

- Smoke test de `fit/predict` end-to-end con datos sintéticos y con el dataset MVP real.
- Test de determinismo con seed fija y misma partición temporal.
- Test de `predict`: no devuelve `NaN`, respeta schema y conserva el índice activo-fecha.
- Test de `registry`: un modelo entrenado se serializa, registra y se puede recargar.

### Criterio de cierre de etapa

La etapa cierra cuando puedes entrenar un baseline, registrar el artefacto y generar predicciones OOS con un procedimiento repetible.

---

## Semana 6 · Portfolio, execution y backtest mínimo viable

### Objetivo

Cerrar el primer circuito monetizable: pasar de predicciones a pesos, órdenes, costes, fills, posiciones y PnL con reconciliación interna.

### Orden de ficheros a tocar

- `portfolio/sizing.py`
- `portfolio/constraints.py`
- `portfolio/neutralize.py`
- `portfolio/rebalance.py`
- `execution/cost_model.py`
- `execution/impact_model.py`
- `execution/fills_simulator.py`
- `execution/turnover_control.py`
- `backtest/engine.py`
- `backtest/diagnostics.py`
- `backtest/attribution.py`
- `backtest/report.py`
- `backtest/run_walkforward.py`

### Qué experimentar esa semana

- Probar dos formas de sizing: proporcional a score vs rank/buckets.
- Simular costes sencillos primero y después estrés con costes e impacto más duros.
- Ejecutar un `walk-forward` corto para verificar toda la tubería, aunque los resultados aún no sean brillantes.
- Comprobar que el backtest deja rastros claros: `orders`, `fills`, `holdings`, `cash`, `NAV`, `turnover` y `attribution`.

### Mini-tests mínimos por módulo

- Test contable: `cash + posiciones + PnL` reconcilia en todo momento.
- Test de `constraints`: no se supera concentración, gross/net ni turnover máximos.
- Test de `fills`: órdenes imposibles o de tamaño no permitido deben ser rechazadas o truncadas.
- Test de `run_walkforward`: al menos una corrida corta completa desde `train` hasta `report`.

### Criterio de cierre de etapa

La etapa cierra cuando ya existe un backtest mínimo end-to-end que produce `NAV`, métricas y reportes sin inconsistencias contables.

---

## Semana 7 · Validation: gates de promoción y control de fragilidad

### Objetivo

Formalizar el sistema que decide si una idea merece avanzar. Esta semana no busca mejorar Sharpe sino medir si la mejora es creíble.

### Orden de ficheros a tocar

- `validation/walkforward_bias.py`
- `validation/validation_suite.py`
- `validation/multiple_testing.py`
- `validation/pbo_cscv.py`
- `validation/synthetic_shocks.py`
- `validation/capacity_sanity.py`

### Qué experimentar esa semana

- Correr la validación sobre el baseline y verificar que el framework castiga correctamente la fragilidad.
- Crear escenarios sintéticos buenos y malos para confirmar que la suite distingue señal real de sobreajuste.
- Definir outputs compactos: `PASS/WARN/FAIL` por gate y resumen global por run.
- Introducir capacidad y shocks aunque todavía sea en versión simple.

### Mini-tests mínimos por módulo

- Test de lógica de gates: un `FAIL` estructural no puede quedar compensado por otras métricas.
- Test de `PBO/multiple testing` con casos sintéticos que deban disparar alarmas.
- Test de `synthetic_shocks`: ante inputs adversos el módulo sigue respondiendo y deja trazabilidad.
- Test de `validation_suite`: compone correctamente todos los subresultados en un `dict/schema` estable.

### Criterio de cierre de etapa

La etapa cierra cuando ya puedes decir no solo “la estrategia gana”, sino “la estrategia supera o no supera los gates mínimos de credibilidad”.

---

## Semana 8 · Datos avanzados: EDGAR, borrow y enriquecimiento de señales

### Objetivo

Subir un escalón de calidad informacional integrando fundamentales point-in-time y proxies de shortability/borrow realistas.

### Orden de ficheros a tocar

- `data/edgar/ticker_cik.py`
- `data/edgar/fetch_submissions.py`
- `data/edgar/fetch_companyfacts.py`
- `data/edgar/point_in_time.py`
- `data/edgar/parse_xbrl.py`
- `data/edgar/filings_flags.py`
- `data/edgar/edgar_qc.py`
- `data/borrow/borrow_cost_proxy.py`
- `data/borrow/locate_filter.py`
- `data/borrow/borrow_qc.py`
- `labels/neutralized_targets.py`
- `labels/event_labels.py`
- `features/fundamentals_core.py`
- `features/fundamentals_deltas.py`
- `features/interactions.py`

### Qué experimentar esa semana

- Montar primero el flujo PIT de EDGAR y solo después derivar features fundamentales.
- Comparar features de fundamentales en nivel vs deltas/cambios para evitar información demasiado lenta.
- Construir el borrow proxy con buckets o tiering y medir su cobertura real.
- Añadir labels de eventos o neutralizados si empiezan a mejorar claridad económica.

### Mini-tests mínimos por módulo

- Test PIT de fundamentales: `acceptance_ts` y `availability_ts` respetados.
- Test de parseo XBRL sobre casos pequeños y conocidos.
- Test de `borrow_qc`: cobertura mínima, buckets válidos y monotonicidad de stress/fee.
- Test de integración: `build_features` puede incorporar fundamentales y borrow sin romper el schema previo.

### Criterio de cierre de etapa

La etapa cierra cuando los datos avanzados ya se pueden unir al pipeline básico sin romper causalidad ni estabilidad.

---

## Semana 9 · Modelos avanzados: GBDT, regímenes y ensemble

### Objetivo

Solo ahora toca sofisticar el motor predictivo. La pregunta de la semana no es “¿puedo entrenarlo?” sino “¿mejora de forma robusta a la baseline?”.

### Orden de ficheros a tocar

- `models/gbdt/train_lgbm.py`
- `models/gbdt/train_xgb.py`
- `models/gbdt/calibrate.py`
- `models/gbdt/tune_hyperparams.py`
- `models/regimes/regime_features.py`
- `models/regimes/hmm.py`
- `models/regimes/regime_gating.py`
- `models/ensemble/blender.py`
- `models/ensemble/stacking.py`
- `models/ensemble/diversity_metrics.py`

### Qué experimentar esa semana

- Comparar `LightGBM` y `XGBoost` contra `Ridge` bajo exactamente el mismo protocolo temporal.
- Medir si el gating por regímenes reduce drawdown o simplemente añade complejidad sin premio.
- Explorar ensembles solo si las fuentes de señal son realmente diversas y no redundantes.
- Registrar todo en tablas comparables: OOS, turnover, estabilidad, sensibilidad a costes y gates.

### Mini-tests mínimos por módulo

- Smoke test de cada trainer con un dataset reducido.
- Test de `calibrate/predict`: probabilidades o scores bien formados.
- Test de régimen: el gating produce estados válidos y no hace look-ahead.
- Test de ensemble: mantiene alineación temporal y schema de predicciones.

### Criterio de cierre de etapa

La etapa cierra cuando puedes demostrar, con el mismo backtest y la misma validación, si los modelos complejos baten o no al baseline.

---

## Semana 10 · Riesgo, optimización, operación diaria y research institucional

### Objetivo

Cerrar la primera versión completa del framework: riesgo medible, optimización de cartera más seria, pipeline diario y sistema de experiment tracking.

### Orden de ficheros a tocar

- `risk/factor_model.py`
- `risk/cov_shrinkage.py`
- `risk/exposure_report.py`
- `risk/drawdown_control.py`
- `risk/stress_tests.py`
- `portfolio/optimizer.py`
- `portfolio/capacity.py`
- `research/research_pipeline.py`
- `research/alpha_discovery/*`
- `research/ablation/*`
- `research/capacity_analysis/*`
- `research/experiment_tracker/*`
- `ops/data_health.py`
- `ops/drift_detection.py`
- `ops/monitoring.py`
- `ops/daily_pipeline.py`

### Qué experimentar esa semana

- Integrar `optimizer` y `risk` solo cuando ya exista una señal o modelo digno de optimizar.
- Montar `research_pipeline.py` como entrypoint de investigación y `daily_pipeline.py` como orquestador operativo.
- Añadir reporting de exposiciones, stress y capacidad para que el framework deje de ser solo un backtester.
- Cerrar experiment tracking y runbook mínimo para repetir runs sin caos.

### Mini-tests mínimos por módulo

- Test de `optimizer` con constraints y soluciones factibles/inviables.
- Test de `risk`: exposiciones, covarianza, drawdown y stress devuelven objetos bien definidos.
- Test del `daily pipeline`: ejecución idempotente de una fecha lógica sin duplicar outputs.
- Test del `research pipeline`: una hipótesis pequeña recorre `discovery -> modelado -> backtest -> validation`.

### Criterio de cierre de etapa

La etapa cierra cuando el repositorio ya tiene una versión 1 coherente: investiga, valida, backtestea y puede ejecutarse con entrypoints claros.

---

## Notas finales de uso

- Este roadmap no implica que todo quede perfecto en 10 semanas; implica que al final de la semana 10 ya existe una versión 1 end-to-end sobre la que iterar.
- Si una semana se atasca, no conviene abrir tres frentes nuevos. La regla buena es cerrar dependencias primero y posponer sofisticación.
- El orden está diseñado para evitar autoengaño: primero datos y causalidad, luego baseline, luego backtest, luego validación y solo después complejidad.
- En paralelo, cada semana conviene dejar al menos un notebook o script de sanity-check que ilustre el bloque recién construido.

**Sugerencia práctica:** cuando termines este roadmap, el siguiente documento útil sería un checklist por commits, con nombres concretos de PR o hitos de Git.
