# Validation Gates

Gates cuantitativos para decidir promocion o descarte de estrategias.

## Filosofia

- Se evalua sobre resultados OOS walk-forward.
- Se evalua en neto (costes, impacto, borrow).
- Se rechaza por defecto si falta evidencia.

## Gate 0: Integridad de datos

Condiciones:
- Sin leakage PIT o de timestamps.
- Sin survivorship bias no controlado.
- Accounting consistente (positions -> trades -> fills -> pnl).

## Gate 1: Calidad estadistica minima

Umbrales recomendados OOS:
- `Sharpe neto >= 1.2`
- `Calmar >= 0.9`
- `Max Drawdown <= 20%`

## Gate 2: Estabilidad temporal

- Al menos `65%` de bloques walk-forward positivos.
- IC medio por bloque con signo consistente.
- Ningun subperiodo explica >40% del pnl total.

## Gate 3: Robustez de friccion

Escenarios:
- Base
- Costes `+25%`
- Impacto `+25%`
- Costes + impacto `+25%`

Condicion:
- Sharpe neto en escenario severo >= `0.9`.

## Gate 4: Overfitting y multiple testing

- `PBO <= 0.20`
- Pasa control de multiples pruebas
- Mejora vs baseline no atribuible a azar

## Gate 5: Riesgo y exposicion

- Beta promedio dentro de banda objetivo.
- Concentracion por nombre y sector bajo limites.
- Turnover en rango operativo.

## Gate 6: Capacidad

- Retorno neto aceptable en al menos 2 niveles de capital.
- Participacion ADV por nombre dentro de limite.
- Degradacion por escala documentada.

## Gate 7: Explicabilidad

- Tesis causal defendible para senales dominantes.
- Atribucion clara por alpha/factor/costes.
- Plan de monitoreo en paper trading.

## Decision final

Estados:
- `PROMOTE_TO_PAPER`
- `HOLD_FOR_RESEARCH`
- `REJECT`

Regla:
- Para promocion, gates 0-7 en PASS.
- Leakage o accounting invalido implica REJECT inmediato.
