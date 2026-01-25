"""
Módulos de Machine Learning do Dashboard Financeiro

Fase 1:
- Categorizer: Categorização automática de transações (NLP + regras)

Fase 2:
- AnomalyDetector: Detecção de anomalias com Isolation Forest
- GastosAnalyzer: Análise, previsão e otimização de gastos
"""
from ml.categorizer import Categorizer, categorize_transaction
from ml.anomaly_detector import AnomalyDetector, detect_anomalies, get_anomaly_report
from ml.otimizador_gastos import (
    GastosAnalyzer,
    analyze_spending,
    predict_spending,
    get_user_profile,
    get_savings_suggestions
)

__all__ = [
    # Fase 1
    "Categorizer",
    "categorize_transaction",
    # Fase 2
    "AnomalyDetector",
    "detect_anomalies",
    "get_anomaly_report",
    "GastosAnalyzer",
    "analyze_spending",
    "predict_spending",
    "get_user_profile",
    "get_savings_suggestions"
]
