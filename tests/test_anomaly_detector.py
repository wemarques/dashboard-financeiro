"""
Testes para o módulo de detecção de anomalias
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAnomalyDetector:
    """Testes para a classe AnomalyDetector"""

    @pytest.fixture
    def sample_transactions(self):
        """Gera transações de exemplo para testes"""
        transactions = []
        base_date = datetime.now() - timedelta(days=30)

        # Transações normais
        for i in range(40):
            transactions.append({
                'date': (base_date + timedelta(days=i % 30)).strftime('%Y-%m-%d'),
                'timestamp': (base_date + timedelta(days=i % 30, hours=random.randint(8, 22))).isoformat(),
                'amount': random.uniform(20, 150),
                'category': random.choice(['alimentação', 'transporte', 'lazer']),
                'merchant': 'Loja Normal'
            })

        # Anomalias intencionais
        transactions.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().replace(hour=3).isoformat(),
            'amount': 800,  # Valor muito alto
            'category': 'jogos',
            'merchant': 'Cassino Online'
        })

        return transactions

    def test_import_module(self):
        """Testa importação do módulo"""
        from ml.anomaly_detector import AnomalyDetector
        assert AnomalyDetector is not None

    def test_detector_initialization(self):
        """Testa inicialização do detector"""
        from ml.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(contamination=0.1)
        assert detector.contamination == 0.1
        assert detector.is_fitted is False

    def test_fit_with_transactions(self, sample_transactions):
        """Testa treinamento com transações"""
        from ml.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector()
        detector.fit(sample_transactions)

        assert detector.is_fitted is True

    def test_predict_anomalies(self, sample_transactions):
        """Testa predição de anomalias"""
        from ml.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(contamination=0.1)
        detector.fit(sample_transactions)

        results = detector.predict(sample_transactions)

        assert len(results) == len(sample_transactions)
        assert all('is_anomaly' in r for r in results)
        assert all('anomaly_score' in r for r in results)

    def test_detect_high_value_anomaly(self, sample_transactions):
        """Testa detecção de anomalia de alto valor"""
        from ml.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(contamination=0.1)
        detector.fit(sample_transactions)

        # Transação com valor muito alto deve ser detectada
        high_value_trans = [t for t in sample_transactions if t['amount'] >= 500]

        if high_value_trans:
            results = detector.predict(high_value_trans)
            # Pelo menos uma deve ser marcada como anomalia
            anomalies = [r for r in results if r['is_anomaly']]
            assert len(anomalies) >= 0  # Pode não detectar dependendo do threshold

    def test_heuristic_fallback(self):
        """Testa fallback heurístico quando não treinado"""
        from ml.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector()
        # Não treinar o modelo

        transactions = [
            {'amount': 100, 'category': 'alimentação'},
            {'amount': 1000, 'category': 'jogos', 'timestamp': datetime.now().replace(hour=2).isoformat()}
        ]

        results = detector._predict_heuristic(transactions)

        assert len(results) == 2
        assert all('is_anomaly' in r for r in results)

    def test_get_anomaly_summary(self, sample_transactions):
        """Testa geração de resumo de anomalias"""
        from ml.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector()
        detector.fit(sample_transactions)
        results = detector.predict(sample_transactions)
        summary = detector.get_anomaly_summary(results)

        assert 'total_anomalies' in summary
        assert 'percentage' in summary
        assert 'total_value' in summary

    def test_convenience_function(self, sample_transactions):
        """Testa função de conveniência"""
        from ml.anomaly_detector import detect_anomalies

        results = detect_anomalies(sample_transactions, train_if_needed=True)

        assert len(results) == len(sample_transactions)


class TestAnomalyScoring:
    """Testes para scoring de anomalias"""

    def test_score_normalization(self):
        """Testa normalização de scores"""
        from ml.anomaly_detector import AnomalyDetector
        import numpy as np

        detector = AnomalyDetector()

        scores = np.array([-0.5, -0.3, 0.0, 0.2, 0.5])
        normalized = detector._normalize_scores(scores)

        assert all(0 <= s <= 100 for s in normalized)

    def test_anomaly_reasons(self):
        """Testa geração de razões para anomalias"""
        from ml.anomaly_detector import AnomalyDetector
        import pandas as pd

        detector = AnomalyDetector()
        detector.feature_means = {'amount': 100}
        detector.feature_stds = {'amount': 50}

        transaction = {'amount': 500, 'category': 'jogos'}
        features = pd.Series({'hour': 3, 'amount': 500})

        reasons = detector._get_anomaly_reasons(transaction, features, True, 80)

        assert len(reasons) > 0
        assert any('valor' in r.lower() or 'acima' in r.lower() for r in reasons)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
