# simons_smallcap_swing

Marco de investigación cuantitativa para **swing trading en small caps de EE. UU.**, diseñado como un **framework de research reproducible, auditable y orientado a promoción de estrategias**.

El objetivo del proyecto no es solo encontrar señales con buen backtest, sino construir un proceso que sobreviva a las preguntas difíciles:

- ¿la señal es **point-in-time** de verdad?
- ¿aguanta fuera de muestra?
- ¿sigue funcionando **neta de costes e impacto**?
- ¿tiene capacidad real?
- ¿está protegida frente a overfitting y autoengaño estadístico?

---

## Objetivo del proyecto

Construir y validar un proceso de investigación cuantitativa que pueda identificar señales robustas con potencial de rentabilidad elevada en términos **netos**, dentro de un universo de small caps estadounidenses.

### Hipótesis de referencia

- **CAGR objetivo de investigación:** `30% anual`
- Se trata como una **hipótesis exigente**, no como un resultado garantizado.
- Cualquier resultado se considera válido solo si supera controles estrictos de leakage, robustez, fricción, riesgo, capacidad y reproducibilidad.

---

## Filosofía del sistema

Este repositorio está construido sobre unos principios metodológicos muy concretos:

1. **Point-in-time primero**  
   Ningún dato puede entrar antes de estar disponible en tiempo real.

2. **Neto o no existe**  
   Toda métrica relevante debe evaluarse después de costes, impacto, borrow y restricciones operativas.

3. **Validación temporal seria**  
   Purged CV, embargo, walk-forward, stress tests, PBO/CSCV, multiple testing y auditorías de leakage.

4. **Capacidad desde el diseño**  
   No se optimiza una señal ignorando ADV, slippage, fills realistas o límites de concentración.

5. **Reproducibilidad total**  
   Cada run debe ser trazable por `run_id`, configuración, hashes, inputs y artefactos generados.

6. **Arquitectura modular**  
   Los módulos `.py` implementan lógica reusable; los `.txt` definen el contrato conceptual, criterios de calidad y papel del módulo dentro del sistema.

---

## Cómo está pensado el repositorio

Este proyecto **no está pensado para ejecutar 100 scripts manualmente**.  
La idea es separar:

- **módulos de librería**: piezas reutilizables (`data/`, `features/`, `labels/`, `models/`, `risk/`, `portfolio/`, `execution/`, `validation/`)
- **orquestadores / entrypoints**: piezas que lanzan flujos completos (`research/`, `backtest/`, `ops/`)
- **núcleo común**: contratos, utilidades, esquemas y geometría temporal (`simons_core/`)
- **tests**: validación automática mínima de integridad (`tests/`)

---

## Entry points principales

El diseño natural del proyecto gira en torno a tres ejecutores:

### 1. `research/research_pipeline.py`
Pipeline de investigación completo:

- define una hipótesis
- prepara datos y labels
- entrena baseline/modelos
- ejecuta validaciones
- produce decisión final tipo:
  - `Promote`
  - `Iterate`
  - `Reject`

### 2. `backtest/run_walkforward.py`
Ejecutor del backtest serio en modo walk-forward:

- reentrena en ventanas temporales
- predice OOS
- rebalancea cartera
- aplica costes e impacto
- genera performance agregada y diagnósticos

### 3. `ops/daily_pipeline.py`
Orquestador operativo diario:

- chequea salud de datos
- detecta drift
- prepara inputs del día
- genera salidas operativas reproducibles
- deja trazabilidad para producción o paper trading

> Recomendación: añadir en el futuro un `cli.py` o `main.py` que envuelva estos tres entrypoints y simplifique la ejecución del proyecto.

---

## Flujo end-to-end del sistema

El circuito lógico del proyecto es el siguiente:

1. **Construcción del universo**  
   Qué nombres son elegibles en cada fecha y bajo qué reglas históricas.

2. **Ingesta y ajuste de precios**  
   Series históricas limpias, ajustadas y consistentes con corporate actions.

3. **Datos fundamentales y referencias PIT**  
   EDGAR, ticker mapping, calendario, metadata, borrow proxy, etc.

4. **Construcción de labels**  
   Targets forward, labels de eventos, targets neutralizados, splits purgados.

5. **Construcción de features**  
   Señales cross-sectional, microestructura, fundamentales, deltas e interacciones.

6. **Entrenamiento de modelos**  
   Baselines, GBDT, modelos por régimen, ensembles.

7. **Inferencia**  
   Registro de modelos, predicción y explicabilidad.

8. **Portfolio construction**  
   Sizing, neutralización, constraints, optimizer, rebalance y capacidad.

9. **Execution simulation**  
   Costes, impacto, fills, borrow execution y control de turnover.

10. **Backtest / Walk-forward**  
    Accounting, attribution, diagnostics y reporting.

11. **Validation gates**  
    Leakage, multiple testing, PBO, shocks sintéticos, bias temporal, capacidad.

12. **Promotion decision**  
    La estrategia o señal pasa a la siguiente fase o se rechaza.

---

## Estructura del repositorio

```text
simons_smallcap_swing/
├── backtest/
├── configs/
├── data/
├── docs/
├── execution/
├── features/
├── labels/
├── models/
├── notebooks/
├── ops/
├── portfolio/
├── research/
├── risk/
├── simons_core/
├── tests/
├── validation/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Árbol comentado por módulos

### `backtest/`
Motor de simulación histórica y evaluación secuencial.

- `engine.py` — núcleo del backtest; reconcilia órdenes, fills, posiciones, cash, NAV y P&L.
- `diagnostics.py` — diagnósticos de calidad del backtest y de la trayectoria de resultados.
- `attribution.py` — atribución del rendimiento: por señal, bloque, bucket o componente.
- `report.py` — generación de reportes consolidados de performance, riesgo y fricción.
- `run_walkforward.py` — ejecutor de backtesting walk-forward.

Cada archivo tiene su `.txt` asociado como contrato de diseño.

---

### `data/`
Capa de datos históricos y referencias PIT.

#### `data/universe/`
Define qué activos existen y son elegibles en cada instante.

- `build_universe.py` — construye el universo histórico.
- `corporate_actions.py` — gestiona split, mergers, ticker changes y otros eventos corporativos.
- `survivorship.py` — evita sesgo de supervivencia.
- `universe_qc.py` — controles de integridad del universo.

#### `data/price/`
Precios históricos y proxies de mercado.

- `fetch_prices.py` — ingesta de precios raw.
- `qc_prices.py` — control de calidad de OHLCV y consistencia temporal.
- `adjust_prices.py` — ajuste por corporate actions.
- `market_proxies.py` — series de referencia de mercado, beta context, benchmarks, etc.

#### `data/edgar/`
Fundamentales y eventos regulatorios desde EDGAR, tratados en modo point-in-time.

- `ticker_cik.py` — mapeo ticker ↔ CIK.
- `fetch_submissions.py` — descarga de submissions de SEC.
- `fetch_companyfacts.py` — descarga de company facts.
- `parse_xbrl.py` — parseo de información XBRL.
- `point_in_time.py` — alineación temporal causal de facts y filings.
- `filings_flags.py` — flags/eventos derivados de filings.
- `edgar_qc.py` — QC de integridad, cobertura y consistencia de EDGAR.

#### `data/borrow/`
Capa para shortability, costes de préstamo y filtros operativos de borrow.

- `borrow_cost_proxy.py` — proxy estructurado del borrow fee / tensión de préstamo.
- `locate_filter.py` — filtro de names no localizables o con riesgo operativo elevado.
- `borrow_qc.py` — QC de cobertura, coherencia y estabilidad del borrow proxy.

#### `data/reference/`
Datos de referencia ya materializados en formato parquet.

- `sector_industry_map.parquet` — mapa sector/industria.
- `symbols_metadata.parquet` — metadata maestra de símbolos.
- `ticker_history_map.parquet` — historial de cambios de ticker / mapping histórico.
- `trading_calendar.parquet` — calendario de trading.

---

### `features/`
Ingeniería de señales y preparación de matriz de entrada al modelado.

- `build_features.py` — ensamblador principal de features.
- `cross_sectional.py` — señales cross-sectional (rankings, normalizaciones, spreads relativos, etc.).
- `microstructure.py` — señales de microestructura y comportamiento del precio/volumen.
- `fundamentals_core.py` — features fundamentales base.
- `fundamentals_deltas.py` — cambios/deltas en fundamentales.
- `interactions.py` — interacciones entre señales o familias de features.
- `feature_qc.py` — validación de rango, NA, drift, leak sospechoso y consistencia.
- `feature_store.py` — persistencia y versionado de matrices de features.

---

### `labels/`
Targets, particiones temporales válidas y convenciones de entrenamiento.

- `build_labels.py` — construcción de labels forward base.
- `event_labels.py` — labels asociados a eventos específicos.
- `neutralized_targets.py` — targets neutralizados frente a factores o exposiciones.
- `purged_splits.py` — splits temporales válidos con purga y embargo.
- `label_qc.py` — control de integridad, solape y leakage en labels.

---

### `models/`
Modelado predictivo por familias.

#### `models/baselines/`
Modelos simples y transparentes que actúan como referencia mínima.

- `train_ridge.py` — baseline lineal canónica.
- `train_logistic.py` — baseline logística para clasificación.
- `train_elasticnet.py` — baseline regularizada mixta.

#### `models/gbdt/`
Modelos de gradient boosting.

- `train_lgbm.py` — entrenamiento LightGBM.
- `train_xgb.py` — entrenamiento XGBoost.
- `calibrate.py` — calibración de outputs/scores.
- `tune_hyperparams.py` — búsqueda/tuning de hiperparámetros.

#### `models/regimes/`
Modelos sensibles a estados de mercado.

- `regime_features.py` — features descriptivas de régimen.
- `hmm.py` — detección o modelado de estados con HMM.
- `regime_gating.py` — routing o activación condicionada por régimen.

#### `models/ensemble/`
Combinación de modelos y diversidad de predictores.

- `blender.py` — mezcla directa de modelos.
- `stacking.py` — meta-modelo / stacking.
- `diversity_metrics.py` — medidas de complementariedad entre señales/modelos.

#### `models/inference/`
Salida final del bloque de modelado.

- `predict.py` — predicción sobre features preparados.
- `model_registry.py` — registro/versionado de modelos aprobados.
- `explain.py` — explicabilidad y análisis de contribuciones.

---

### `risk/`
Medición y control del riesgo.

- `factor_model.py` — estimación de exposiciones y factores.
- `cov_shrinkage.py` — estimación robusta de covarianza.
- `exposure_report.py` — reporting de exposiciones por cartera.
- `drawdown_control.py` — controles específicos sobre drawdown.
- `stress_tests.py` — escenarios y shocks de estrés.

---

### `portfolio/`
Construcción de cartera desde señales a pesos ejecutables.

- `sizing.py` — conversión score → tamaño de posición.
- `constraints.py` — restricciones duras y blandas de cartera.
- `neutralize.py` — neutralización por beta, sector, industria, tamaño, etc.
- `optimizer.py` — optimización de cartera.
- `rebalance.py` — lógica de rebalance temporal.
- `capacity.py` — capacidad de la cartera frente a liquidez/ADV/impacto.

---

### `execution/`
Modelización de fricción y ejecución realista.

- `cost_model.py` — comisiones, tasas y costes explícitos.
- `impact_model.py` — impacto implícito de mercado.
- `fills_simulator.py` — simulación de fills y ejecución parcial.
- `borrow_execution.py` — detalles operativos específicos de la pata short/borrow.
- `turnover_control.py` — control de rotación para evitar degradación por fricción.

---

### `validation/`
Sistema de control estadístico y validación de promoción.

- `leakage_audit.py` — auditoría formal de leakage.
- `multiple_testing.py` — corrección/control por múltiples pruebas.
- `pbo_cscv.py` — Probability of Backtest Overfitting y CSCV.
- `walkforward_bias.py` — análisis de sesgos en el protocolo temporal.
- `synthetic_shocks.py` — perturbaciones sintéticas para robustez.
- `capacity_sanity.py` — chequeos de plausibilidad de capacidad.
- `validation_suite.py` — agregador maestro de gates y decisiones.

---

### `research/`
Capa de investigación y experimentación.

- `research_pipeline.py` — pipeline maestro de research.
- `alpha_discovery/` — exploración estructurada de nuevas hipótesis/señales.
- `ablation/` — estudios de contribución marginal por bloque o componente.
- `capacity_analysis/` — estudios dedicados a capacidad y escalabilidad.
- `experiment_tracker/` — seguimiento, versionado y comparación de experimentos.

---

### `ops/`
Operativa diaria, monitorización y runbooks.

- `daily_pipeline.py` — orquestador operativo diario.
- `data_health.py` — salud e integridad operativa de datos.
- `drift_detection.py` — detección de drift en inputs, outputs o comportamiento.
- `monitoring.py` — monitoreo técnico y cuantitativo del sistema.
- `runbook.md` — guía operativa de incidentes, revisión y procedimientos.

---

### `simons_core/`
Núcleo compartido del proyecto.

- `__init__.py` — inicialización del paquete.
- `calendar.py` — geometría temporal y lógica de calendario.
- `schemas.py` — esquemas canónicos de datos.
- `interfaces.py` — contratos comunes entre componentes.
- `logging.py` — logging estructurado y trazabilidad.

#### `simons_core/io/`
- `paths.py` — rutas canónicas del proyecto.
- `parquet_store.py` — persistencia estándar en parquet.
- `cache.py` — caché y reutilización de artefactos.

#### `simons_core/math/`
- `stats.py` — utilidades estadísticas.
- `robust.py` — rutinas robustas para estimación y saneamiento.
- `optimization.py` — helpers de optimización comunes.

---

### `tests/`
Suite mínima de verificación estructural.

- `conftest.py` — fixtures y utilidades compartidas para tests.
- `test_data_pit.py` — asegura causalidad y point-in-time en los datos.
- `test_labels_no_leakage.py` — asegura ausencia de leakage en labels/splits.
- `test_features_schema.py` — valida forma, tipos y contratos de features.
- `test_backtest_accounting.py` — valida reconciliación de accounting en backtest.
- `test_validation_gates.py` — valida el comportamiento de los gates de validación.

---

### `docs/`
Documentación metodológica del framework.

- `data_dictionary.md` — diccionario de datos.
- `schemas.md` — definición de esquemas canónicos.
- `research_protocols.md` — protocolo formal de investigación.
- `validation_gates.md` — definición de PASS/WARN/FAIL y criterios de promoción.
- `milestones_6m.md` — hoja de ruta a 6 meses.

---

### Otros directorios / archivos

- `configs/` — configuraciones por entorno, experimento o pipeline.
- `notebooks/` — notebooks exploratorios o de análisis puntual.
- `Dockerfile` — entorno contenedorizado del proyecto.
- `docker-compose.yml` — orquestación de servicios locales.
- `pyproject.toml` — configuración principal del proyecto Python.
- `requirements.txt` — dependencias directas.
- `poetry.lock` — lockfile reproducible de dependencias.

---

## Convención `.py + .txt`

Cada componente del framework sigue una convención muy importante:

- el archivo **`.py`** contiene la implementación
- el archivo **`.txt`** contiene la especificación funcional / metodológica

Esto permite trabajar en modo **spec-first**:

1. definir el papel del módulo,
2. fijar invariantes y responsabilidades,
3. implementar de forma coherente,
4. testear sin perder la intención original.

Esta convención es una de las bases del repositorio y ayuda mucho a mantener consistencia arquitectónica cuando el proyecto crece.

---

## Qué significa que una señal esté “hecha”

Una idea no se considera válida por tener buen Sharpe en una ventana histórica.  
Una señal o estrategia solo se considera candidata a promoción si cumple, como mínimo:

- consistencia en OOS walk-forward,
- estabilidad temporal,
- robustez a costes e impacto,
- riesgo controlado,
- capacidad razonable,
- baja probabilidad de overfitting,
- reproducibilidad completa,
- explicación económica mínimamente defendible.

---

## Cómo se debería ejecutar el sistema

La forma natural de usar este proyecto es a través de **entrypoints** y no lanzando módulos sueltos a mano.

Ejemplos conceptuales:

```bash
python -m simons_smallcap_swing.research.research_pipeline --config configs/research.yaml
python -m simons_smallcap_swing.backtest.run_walkforward --config configs/backtest.yaml
python -m simons_smallcap_swing.ops.daily_pipeline --config configs/prod.yaml --date 2026-03-12
pytest
```

En una fase posterior, lo ideal es añadir un único `cli.py` con subcomandos como:

```bash
python -m simons_smallcap_swing.cli research ...
python -m simons_smallcap_swing.cli backtest ...
python -m simons_smallcap_swing.cli daily ...
```

---

## Ruta recomendada de implementación

Orden sugerido para ir rellenando los `.py`:

1. `simons_core/`
2. `data/universe/`
3. `data/price/`
4. `labels/`
5. `validation/leakage_audit.py`
6. `features/` (MVP)
7. `models/baselines/`
8. `models/inference/`
9. `portfolio/`
10. `execution/`
11. `backtest/`
12. `validation/` completa
13. `data/edgar/` y `data/borrow/`
14. `models/gbdt/`, `models/regimes/`, `models/ensemble/`
15. `risk/`
16. `research/` y `ops/`

La prioridad no es “tener muchos módulos implementados”, sino **cerrar cuanto antes un primer circuito completo y verificable end-to-end**.

---

## Estrategia de tests

Además de la suite global en `tests/`, cada módulo importante debería tener mini-tests implícitos en su desarrollo:

- smoke test de importación y ejecución mínima,
- tests de invariantes,
- tests de edge cases,
- tests de integración con el módulo aguas abajo.

El objetivo es que cada pieza sea confiable por sí sola y también dentro del pipeline completo.

---

## Estado actual del proyecto

El repositorio está en una fase **muy fuerte en arquitectura y metodología**, con la mayor parte de los componentes ya definidos a nivel de especificación, aunque todavía no completamente implementados.

Eso es deliberado: primero se ha levantado una base seria de research, validación y contratos modulares; después se rellenarán las implementaciones de forma ordenada.

En otras palabras:

- **menos improvisación**,
- **menos riesgo de código caótico**,
- **más probabilidad de terminar con un framework cuantitativo realmente profesional**.

---

## Resumen

`simons_smallcap_swing` no pretende ser solo un backtester, sino un **sistema integral de investigación cuantitativa** con:

- datos PIT,
- features auditables,
- labels causales,
- modelado modular,
- portfolio + execution realistas,
- validación estadística seria,
- orquestación reproducible,
- y criterio formal de promoción.

Si se implementa respetando los contratos definidos en los `.txt`, este repositorio puede convertirse en una base muy sólida para research cuantitativo serio sobre small caps swing en EE. UU.
