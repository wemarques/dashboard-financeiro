"""
Módulos comportamentais do Dashboard Financeiro

Fase 1:
- ImpulseGuard: Proteção contra compras por impulso (horário noturno, valores altos)

Fase 2:
- InterventionEngine: Sistema de intervenções personalizadas
"""
from behavioral.impulse_guard import (
    ImpulseGuard,
    check_transaction_risk,
    is_night_mode,
    get_impulse_guard
)
from behavioral.intervention import (
    InterventionEngine,
    InterventionType,
    InterventionLevel,
    generate_intervention,
    get_reflective_questions,
    get_intervention_engine
)

__all__ = [
    # Fase 1
    "ImpulseGuard",
    "check_transaction_risk",
    "is_night_mode",
    "get_impulse_guard",
    # Fase 2
    "InterventionEngine",
    "InterventionType",
    "InterventionLevel",
    "generate_intervention",
    "get_reflective_questions",
    "get_intervention_engine"
]
