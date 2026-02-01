"""
Detector de anomalias em transações financeiras
Usa Isolation Forest para identificar gastos suspeitos ou fora do padrão
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from config import DATA_DIR
    from utils.logger import get_logger
except ImportError:
    from pathlib import Path
    DATA_DIR = Path("data")
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Detector de anomalias em transações financeiras usando Isolation Forest.

    O Isolation Forest é ideal para detecção de anomalias porque:
    - Não requer dados rotulados
    - Funciona bem com dados de alta dimensionalidade
    - É eficiente computacionalmente
    - Identifica outliers de forma natural
    """

    # Colunas usadas para detecção
    FEATURE_COLUMNS = ['amount', 'hour', 'day_of_week', 'day_of_month', 'category_encoded']

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Inicializa o detector de anomalias.

        Args:
            contamination: Proporção esperada de anomalias (0.0 a 0.5)
            n_estimators: Número de árvores no ensemble
            random_state: Seed para reproducibilidade
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn é necessário para detecção de anomalias")

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

        # Estatísticas para análise
        self.feature_means = {}
        self.feature_stds = {}

        logger.info(f"AnomalyDetector inicializado (contamination={contamination})")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features para o modelo.

        Args:
            df: DataFrame com transações

        Returns:
            DataFrame com features processadas
        """
        features = pd.DataFrame()

        # Valor da transação
        features['amount'] = df['amount'].fillna(0)

        # Features temporais
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            features['hour'] = timestamps.dt.hour
            features['day_of_week'] = timestamps.dt.dayofweek
            features['day_of_month'] = timestamps.dt.day
        elif 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            features['hour'] = 12  # Valor padrão se não tiver hora
            features['day_of_week'] = dates.dt.dayofweek
            features['day_of_month'] = dates.dt.day
        else:
            features['hour'] = 12
            features['day_of_week'] = 0
            features['day_of_month'] = 15

        # Categoria codificada
        if 'category' in df.columns:
            categories = df['category'].fillna('outros').astype(str)
            if not self.is_fitted:
                features['category_encoded'] = self.label_encoder.fit_transform(categories)
            else:
                # Para categorias não vistas, usar valor padrão
                features['category_encoded'] = categories.apply(
                    lambda x: self.label_encoder.transform([x])[0]
                    if x in self.label_encoder.classes_
                    else -1
                )
        else:
            features['category_encoded'] = 0

        # Features adicionais derivadas
        features['is_high_amount'] = (features['amount'] > features['amount'].quantile(0.9)).astype(int)
        features['is_night'] = ((features['hour'] >= 0) & (features['hour'] <= 6)).astype(int)
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)

        return features

    def fit(self, transactions: List[Dict[str, Any]]) -> 'AnomalyDetector':
        """
        Treina o modelo com histórico de transações.

        Args:
            transactions: Lista de dicionários com transações

        Returns:
            Self para encadeamento
        """
        if len(transactions) < 10:
            logger.warning("Poucas transações para treinar o modelo. Mínimo recomendado: 50")
            return self

        df = pd.DataFrame(transactions)
        features = self._prepare_features(df)

        # Armazenar estatísticas
        for col in features.columns:
            self.feature_means[col] = features[col].mean()
            self.feature_stds[col] = features[col].std()

        # Normalizar features
        features_scaled = self.scaler.fit_transform(features)

        # Treinar modelo
        self.model.fit(features_scaled)
        self.is_fitted = True

        logger.info(f"Modelo treinado com {len(transactions)} transações")

        return self

    def predict(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detecta anomalias em transações.

        Args:
            transactions: Lista de transações para analisar

        Returns:
            Lista de transações com score de anomalia
        """
        if not self.is_fitted:
            logger.warning("Modelo não treinado. Usando heurísticas simples.")
            return self._predict_heuristic(transactions)

        df = pd.DataFrame(transactions)
        features = self._prepare_features(df)
        features_scaled = self.scaler.transform(features)

        # Predições: -1 = anomalia, 1 = normal
        predictions = self.model.predict(features_scaled)

        # Scores de anomalia (quanto menor, mais anômalo)
        scores = self.model.decision_function(features_scaled)

        # Normalizar scores para 0-100 (100 = mais anômalo)
        scores_normalized = self._normalize_scores(scores)

        results = []
        for i, transaction in enumerate(transactions):
            is_anomaly = predictions[i] == -1
            anomaly_score = scores_normalized[i]

            result = {
                **transaction,
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'anomaly_reasons': self._get_anomaly_reasons(
                    transaction,
                    features.iloc[i],
                    is_anomaly,
                    anomaly_score
                )
            }
            results.append(result)

        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        logger.info(f"Detectadas {anomaly_count} anomalias em {len(transactions)} transações")

        return results

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normaliza scores para escala 0-100"""
        # Scores do Isolation Forest são negativos para anomalias
        # Quanto mais negativo, mais anômalo
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.full_like(scores, 50.0)

        # Inverter: valores mais negativos (anômalos) -> scores mais altos
        normalized = 100 * (1 - (scores - min_score) / (max_score - min_score))
        return normalized

    def _get_anomaly_reasons(
        self,
        transaction: Dict[str, Any],
        features: pd.Series,
        is_anomaly: bool,
        score: float
    ) -> List[str]:
        """
        Identifica razões para a anomalia.

        Args:
            transaction: Transação original
            features: Features processadas
            is_anomaly: Se é anomalia
            score: Score de anomalia

        Returns:
            Lista de razões explicativas
        """
        reasons = []

        if not is_anomaly:
            return reasons

        amount = transaction.get('amount', 0)

        # Verificar valor
        if 'amount' in self.feature_means:
            mean_amount = self.feature_means['amount']
            std_amount = self.feature_stds['amount']
            if std_amount > 0 and amount > mean_amount + 2 * std_amount:
                reasons.append(f"Valor muito acima da média (R$ {amount:.2f} vs média R$ {mean_amount:.2f})")

        # Verificar horário
        hour = features.get('hour', 12)
        if 0 <= hour <= 6:
            reasons.append(f"Transação em horário incomum ({hour}h)")

        # Verificar dia da semana
        if features.get('is_weekend', 0) == 1:
            reasons.append("Transação no fim de semana")

        # Verificar categoria
        category = transaction.get('category', 'outros')
        if category in ['jogos', 'apostas', 'cassino']:
            reasons.append(f"Categoria de alto risco: {category}")

        # Se não encontrou razão específica
        if not reasons:
            reasons.append(f"Padrão incomum detectado (score: {score:.0f})")

        return reasons

    def _predict_heuristic(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detecta anomalias usando heurísticas simples (fallback).

        Args:
            transactions: Lista de transações

        Returns:
            Lista com análise heurística
        """
        if not transactions:
            return []

        # Calcular estatísticas básicas
        amounts = [t.get('amount', 0) for t in transactions]
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts) if len(amounts) > 1 else mean_amount * 0.5

        results = []
        for transaction in transactions:
            amount = transaction.get('amount', 0)
            reasons = []
            score = 0

            # Regra 1: Valor muito alto
            if std_amount > 0:
                z_score = abs(amount - mean_amount) / std_amount
                if z_score > 2:
                    score += min(z_score * 20, 50)
                    reasons.append(f"Valor atípico: {z_score:.1f} desvios da média")

            # Regra 2: Horário noturno
            timestamp = transaction.get('timestamp')
            if timestamp:
                hour = pd.to_datetime(timestamp).hour
                if 0 <= hour <= 6:
                    score += 25
                    reasons.append(f"Horário noturno: {hour}h")

            # Regra 3: Categoria de risco
            category = transaction.get('category', '').lower()
            if category in ['jogos', 'apostas', 'cassino', 'bet']:
                score += 30
                reasons.append(f"Categoria de risco: {category}")

            is_anomaly = score >= 50

            results.append({
                **transaction,
                'is_anomaly': is_anomaly,
                'anomaly_score': min(score, 100),
                'anomaly_reasons': reasons
            })

        return results

    def get_anomaly_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gera resumo das anomalias detectadas.

        Args:
            results: Resultados da detecção

        Returns:
            Dicionário com estatísticas
        """
        anomalies = [r for r in results if r.get('is_anomaly')]

        if not anomalies:
            return {
                'total_anomalies': 0,
                'percentage': 0,
                'total_value': 0,
                'categories': {},
                'time_distribution': {}
            }

        # Agrupar por categoria
        categories = {}
        for a in anomalies:
            cat = a.get('category', 'outros')
            if cat not in categories:
                categories[cat] = {'count': 0, 'total': 0}
            categories[cat]['count'] += 1
            categories[cat]['total'] += a.get('amount', 0)

        # Distribuição por hora
        time_dist = {}
        for a in anomalies:
            ts = a.get('timestamp')
            if ts:
                hour = pd.to_datetime(ts).hour
                period = f"{hour:02d}h"
                time_dist[period] = time_dist.get(period, 0) + 1

        return {
            'total_anomalies': len(anomalies),
            'percentage': len(anomalies) / len(results) * 100 if results else 0,
            'total_value': sum(a.get('amount', 0) for a in anomalies),
            'average_score': np.mean([a.get('anomaly_score', 0) for a in anomalies]),
            'categories': categories,
            'time_distribution': time_dist,
            'top_anomalies': sorted(anomalies, key=lambda x: x.get('anomaly_score', 0), reverse=True)[:5]
        }


# === Funções de conveniência ===

_detector = None


def get_detector() -> AnomalyDetector:
    """Retorna instância global do detector"""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector


def detect_anomalies(
    transactions: List[Dict[str, Any]],
    train_if_needed: bool = True
) -> List[Dict[str, Any]]:
    """
    Detecta anomalias em transações (função de conveniência).

    Args:
        transactions: Lista de transações
        train_if_needed: Se deve treinar o modelo se não estiver treinado

    Returns:
        Lista de transações com análise de anomalias
    """
    detector = get_detector()

    if train_if_needed and not detector.is_fitted and len(transactions) >= 10:
        detector.fit(transactions)

    return detector.predict(transactions)


def get_anomaly_report(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gera relatório de anomalias.

    Args:
        transactions: Lista de transações

    Returns:
        Relatório detalhado
    """
    results = detect_anomalies(transactions)
    detector = get_detector()
    return detector.get_anomaly_summary(results)
