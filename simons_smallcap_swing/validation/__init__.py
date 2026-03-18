"""
validation — Research integrity verification suite.

Ensures that backtested performance is not an artifact of leakage,
overfitting, data snooping, or capacity constraints.

Modules:
    leakage_audit     Temporal leakage detection (5 classes)
    walkforward_bias  Walk-forward geometry validation
    multiple_testing  FWER/FDR correction + effective N
    pbo_cscv          Probability of Backtest Overfitting (Bailey et al.)
    synthetic_shocks  Stress testing under adverse scenarios
    capacity_sanity   Strategy capacity under cost scaling
    validation_suite  Orchestrator with hard/soft gate taxonomy
"""
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Any


class ValidationError(RuntimeError):
    pass

class LeakageDetected(ValidationError):
    pass


class GateStatus(str, enum.Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class GateResult:
    gate_name: str
    category: str       # "structural" | "statistical" | "economic"
    status: str         # PASS / WARN / FAIL
    metric_value: Any
    threshold: Any
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"
