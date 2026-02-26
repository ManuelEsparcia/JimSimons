# simons_smallcap_swing

Marco de investigacion cuantitativa para swing trading en small caps de EE. UU. Disenado para un equipo de research avanzado, con foco en reproducibilidad, control de sesgos y validacion fuera de muestra.

## Objetivo del proyecto

Construir y validar un proceso de investigacion que pueda identificar senales robustas con potencial de rentabilidad elevada en terminos netos de costes.

Objetivo de referencia:
- CAGR objetivo de investigacion: `30% anual`.
- Se trata como hipotesis exigente, no como resultado garantizado.

## Principios rectores

- Point-in-time primero.
- Control de autoengano: purged CV, embargo, walk-forward, PBO y multiple testing.
- Neto o no existe: toda metrica se mide despues de costes e impacto.
- Capacidad real desde el diseno.
- Reproducibilidad total por run_id y hashes.

## Arquitectura por capas

- `data/`: universo, precios, EDGAR PIT, borrow proxy.
- `features/`: ingenieria de senales y feature store.
- `labels/`: targets forward y neutralizados.
- `models/`: baselines, GBDT, regimenes, ensembles, inferencia.
- `risk/` y `portfolio/`: exposiciones, constraints, sizing.
- `execution/`: costes, impacto, fills, turnover.
- `backtest/`: motor walk-forward y atribucion.
- `validation/`: gates de promocion y control de overfitting.
- `research/`: discovery, ablation, capacity, tracking.
- `ops/`: salud de datos, drift, monitoreo y runbook.
- `simons_core/`: contratos, esquemas, utilidades canonicas.

## Definicion de "hecho"

Un resultado se considera valido solo si cumple:
- Metricas OOS consistentes en ventanas walk-forward.
- Robustez a cambios de costes e impacto.
- Riesgo controlado (drawdown, concentracion, exposiciones).
- Baja probabilidad de overfitting.
- Reproducibilidad completa de resultados.

## Flujo de research

1. Definir hipotesis y mecanismo economico.
2. Preparar datos PIT y universo con QC.
3. Construir features y labels sin leakage.
4. Entrenar baseline y validar temporalmente.
5. Auditar performance neta.
6. Ejecutar ablation, stress y estabilidad.
7. Pasar gates para promocion.

## Documentacion clave

- `docs/data_dictionary.md`
- `docs/schemas.md`
- `docs/research_protocols.md`
- `docs/validation_gates.md`
- `docs/milestones_6m.md`
- `ops/runbook.md`

## Estado actual

Repositorio en fase de documentacion y esqueleto estructural. La implementacion comenzara sobre esta base metodologica.
