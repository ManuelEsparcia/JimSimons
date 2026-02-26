# Research Protocols

Protocolos obligatorios para investigacion cuantitativa en este proyecto.

## 1) Ciclo de hipotesis

Cada linea de research debe definir:
- `hypothesis_id`
- mecanismo economico esperado
- universo objetivo
- horizonte temporal
- riesgo principal de invalidez
- criterio de exito y fallo

Sin hipotesis explicita no se lanzan barridos masivos.

## 2) Diseno experimental

Minimos obligatorios:
- split temporal estricto
- purged K-Fold con embargo
- walk-forward para OOS final
- metricas netas de costes
- baseline obligatorio

## 3) Control de leakage

- Features computables en `t` con info disponible en `t`.
- Fundamentals solo con `as_of_date <= t`.
- Sin agregaciones con mirada al futuro.
- `validation/leakage_audit.py` es gate bloqueante.

## 4) Multiple testing

- Registrar numero total de hipotesis probadas.
- Reportar distribucion de resultados, no solo el mejor.
- Aplicar Reality Check o esquema FDR-like.
- Penalizar seleccion oportunista.

## 5) Walk-forward

- Entrenamiento en ventana expandida o rodante.
- Test por bloques temporales fijos.
- Reentrenamiento con calendario predefinido.
- Sin ajuste manual tras ver bloques OOS.

## 6) Ablation obligatoria

Todo candidato debe pasar:
- ablation de features
- ablation de modelo
- ablation de costes e impacto

Si el edge desaparece con supuestos ligeramente peores, no se promueve.

## 7) Capacidad

- Curva de rentabilidad vs capital.
- Limites de participacion ADV por nombre.
- Sensibilidad a turnover.
- Rechazo de estrategias no escalables.

## 8) Robustez

Pruebas minimas:
- shocks de precio y volatilidad
- shocks de costes y slippage
- submuestras por regimen
- estabilidad de IC

## 9) Registro y trazabilidad

Todo experimento debe registrar:
- configuracion completa
- version de datos
- hash de codigo
- outputs intermedios
- reporte final reproducible

## 10) Criterio de promocion

Se promueve solo si:
- supera `validation_gates.md`
- mejora baseline en OOS
- tiene tesis causal defendible
- mantiene neto positivo tras fricciones

## 11) Criterio de descarte

Descartar de inmediato si:
- hay leakage confirmado
- la mejora depende de un solo periodo
- hay sensibilidad extrema a hiperparametros
- la mejora es solo beta/factor no deseado
