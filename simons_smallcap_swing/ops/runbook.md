# Runbook Operativo

Guia operativa para mantener continuidad, calidad y respuesta a incidencias en el ciclo diario de research y paper.

## 1) Operacion diaria

Checklist diario:
- Verificar estado de feeds de precio y EDGAR.
- Ejecutar `data_health` y revisar alertas.
- Validar integridad de tablas del dia.
- Ejecutar pipeline diario ligero.
- Registrar metricas de drift y cobertura.
- Publicar resumen de estado.

## 2) Operacion semanal

- Revisar estabilidad de IC y performance OOS reciente.
- Revisar costes efectivos vs supuestos.
- Revisar exposicion factor y sector.
- Priorizar hipotesis activas.

## 3) Operacion mensual

- Recalibrar supuestos de costes e impacto.
- Revisar capacidad por nivel de capital.
- Auditar reproducibilidad de experimentos clave.
- Revisar deuda tecnica del pipeline.

## 4) Clasificacion de incidencias

Severidad:
- `SEV-1`: riesgo alto de decisiones invalidas.
- `SEV-2`: degradacion relevante.
- `SEV-3`: anomalia menor.

SLA objetivo:
- `SEV-1`: mitigacion inicial en menos de 2h.
- `SEV-2`: mitigacion en menos de 1 dia habil.
- `SEV-3`: resolver en backlog priorizado.

## 5) Protocolo de respuesta

1. Detectar y clasificar severidad.
2. Congelar decisiones si es SEV-1.
3. Identificar componente y ventana afectada.
4. Aplicar mitigacion temporal segura.
5. Revalidar datasets y modelos impactados.
6. Cerrar con postmortem corto.

## 6) Escenarios criticos

Caso A: falla feed de precios
- Usar fuente secundaria si existe.
- Marcar dataset incompleto.
- Reejecutar pipeline al restablecer la fuente.

Caso B: error PIT en fundamentals
- Invalidar features y labels dependientes.
- Reejecutar ventanas afectadas.
- Correr leakage audit antes de reactivar.

Caso C: drift severo de features
- Activar alerta preventiva.
- Comparar distribucion con baseline.
- Congelar cambios de modelo hasta diagnostico.

## 7) Cambios de configuracion

- Todo cambio requiere `change_note`.
- Versionar config con timestamp y autor.
- Si afecta backtest, ejecutar validacion minima.

## 8) Control de calidad continuo

Metricas minimas:
- Cobertura de universo.
- Missing por feature critica.
- Turnover medio y percentiles.
- Coste neto realizado vs supuesto.
- Error de tracking simulacion vs paper.

## 9) Criterio de pausa operativa

Pausar promociones si ocurre cualquiera:
- Leakage sin resolver.
- Fallo repetido de accounting.
- Divergencia no explicada entre backtest y paper.
- Degradacion abrupta bajo umbral de riesgo.

## 10) Postmortem estandar

Todo incidente SEV-1 o SEV-2 debe cerrar con:
- timeline
- causa raiz
- impacto cuantificado
- mitigacion aplicada
- accion preventiva permanente
