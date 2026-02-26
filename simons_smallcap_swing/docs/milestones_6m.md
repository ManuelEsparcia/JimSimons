# Milestones 6M

Plan de 6 meses para equipo de 10 doctores orientado a investigacion robusta.

## Estructura de trabajo

- Duracion: 24 semanas.
- Cadencia: sprints de 2 semanas (12 sprints).
- Entrega por sprint: demo, reporte tecnico, decision go/no-go.

## Roles sugeridos

- 2 personas en data (universo, PIT, QC).
- 2 personas en features/labels.
- 2 personas en modeling.
- 1 persona en validation.
- 1 persona en portfolio/risk.
- 1 persona en execution/backtest.
- 1 persona en MLOps y trazabilidad.

## Sprint 1-2 (Semanas 1-4)

Objetivos:
- Universo historico reproducible.
- Precios ajustados con QC.
- EDGAR PIT base con joins auditables.

Entregables:
- Datasets canonicos iniciales.
- Reporte de calidad de datos.
- Tests PIT y survivorship activos.

## Sprint 3-4 (Semanas 5-8)

Objetivos:
- Feature store minimo usable.
- Labels forward y neutralizados.
- Controles de leakage en features y labels.

Entregables:
- Matriz de features v1.
- Labels 5d/10d/20d.
- Reporte de cobertura y missingness.

## Sprint 5-6 (Semanas 9-12)

Objetivos:
- Baselines lineales y primer GBDT.
- Primer walk-forward neto.
- Pipeline trazable end-to-end.

Entregables:
- Benchmark de modelos base.
- Dashboard de metricas OOS.
- Diagnostico inicial de costes/capacidad.

## Sprint 7-8 (Semanas 13-16)

Objetivos:
- Alpha discovery controlado.
- Ablations sistematicas.
- Control de multiples pruebas activo.

Entregables:
- Ranking de familias de senales.
- Reporte de estabilidad IC.
- Candidatos preliminares.

## Sprint 9-10 (Semanas 17-20)

Objetivos:
- Neutralizacion y constraints robustos.
- Modelo de impacto y control de turnover.
- Backtest con atribucion completa.

Entregables:
- Politica de sizing y rebalance documentada.
- Reporte de exposicion factorial.
- Escenarios de stress operativos.

## Sprint 11-12 (Semanas 21-24)

Objetivos:
- Pasar validation gates 0-7.
- Monitoreo y runbook listos.
- Seleccion final para paper/live pilot.

Entregables:
- Informe final de research.
- Decision PROMOTE/HOLD/REJECT por estrategia.
- Checklist de operacion continua.

## KPIs del programa

- Porcentaje de experimentos reproducibles.
- Tiempo medio hipotesis -> decision.
- Ratio de hipotesis descartadas por sesgo o leakage.
- Mejora neta incremental vs baseline.
- Estabilidad OOS por bloque temporal.
