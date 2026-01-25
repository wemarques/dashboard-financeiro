"""
Testes para o módulo de otimização de gastos
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGastosAnalyzer:
    """Testes para a classe GastosAnalyzer"""

    @pytest.fixture
    def sample_transactions(self):
        """Gera transações de exemplo"""
        transactions = []
        categories = ['alimentação', 'delivery', 'transporte', 'lazer']
        base_date = datetime.now() - timedelta(days=30)

        for i in range(50):
            cat = random.choice(categories)
            transactions.append({
                'date': (base_date + timedelta(days=i % 30)).strftime('%Y-%m-%d'),
                'timestamp': (base_date + timedelta(days=i % 30, hours=random.randint(8, 22))).isoformat(),
                'amount': random.uniform(20, 200),
                'category': cat,
                'merchant': f'Loja {cat}'
            })

        return transactions

    def test_import_module(self):
        """Testa importação do módulo"""
        from ml.otimizador_gastos import GastosAnalyzer
        assert GastosAnalyzer is not None

    def test_analyzer_initialization(self):
        """Testa inicialização do analisador"""
        from ml.otimizador_gastos import GastosAnalyzer

        analyzer = GastosAnalyzer()
        assert analyzer is not None

    def test_analyze_spending_patterns(self, sample_transactions):
        """Testa análise de padrões de gastos"""
        from ml.otimizador_gastos import GastosAnalyzer

        analyzer = GastosAnalyzer()
        result = analyzer.analyze_spending_patterns(sample_transactions)

        assert 'summary' in result
        assert 'by_category' in result
        assert 'temporal' in result
        assert 'trends' in result

    def test_analyze_by_category(self, sample_transactions):
        """Testa análise por categoria"""
        from ml.otimizador_gastos import GastosAnalyzer
        import pandas as pd

        analyzer = GastosAnalyzer()
        df = pd.DataFrame(sample_transactions)
        result = analyzer._analyze_by_category(df)

        assert isinstance(result, dict)
        # Deve ter categorias
        assert len(result) > 0

    def test_predict_spending(self, sample_transactions):
        """Testa previsão de gastos"""
        from ml.otimizador_gastos import GastosAnalyzer

        analyzer = GastosAnalyzer()
        result = analyzer.predict_spending(sample_transactions, days_ahead=30)

        assert 'method' in result
        assert 'total_predicted' in result
        assert result['total_predicted'] > 0

    def test_predict_simple_fallback(self, sample_transactions):
        """Testa previsão simples (fallback)"""
        from ml.otimizador_gastos import GastosAnalyzer

        analyzer = GastosAnalyzer()
        result = analyzer._predict_simple(sample_transactions, days_ahead=30)

        assert 'method' in result
        assert result['method'] == 'Média histórica'
        assert 'total_predicted' in result

    def test_identify_behavioral_profile(self, sample_transactions):
        """Testa identificação de perfil comportamental"""
        from ml.otimizador_gastos import GastosAnalyzer

        analyzer = GastosAnalyzer()
        result = analyzer.identify_behavioral_profile(sample_transactions)

        assert 'profile' in result
        assert 'confidence' in result
        assert result['profile'] in ['notívago', 'impulsivo', 'social', 'controlado', 'sazonal']

    def test_generate_savings_suggestions(self, sample_transactions):
        """Testa geração de sugestões de economia"""
        from ml.otimizador_gastos import GastosAnalyzer

        analyzer = GastosAnalyzer()
        result = analyzer.generate_savings_suggestions(sample_transactions, target_reduction=0.1)

        assert 'suggestions' in result
        assert 'total_potential_savings' in result
        assert 'target_savings' in result
        assert 'achievable' in result

    def test_convenience_functions(self, sample_transactions):
        """Testa funções de conveniência"""
        from ml.otimizador_gastos import (
            analyze_spending,
            predict_spending,
            get_user_profile,
            get_savings_suggestions
        )

        # Todas devem retornar resultados válidos
        result1 = analyze_spending(sample_transactions)
        assert 'summary' in result1

        result2 = predict_spending(sample_transactions, 30)
        assert 'total_predicted' in result2

        result3 = get_user_profile(sample_transactions)
        assert 'profile' in result3

        result4 = get_savings_suggestions(sample_transactions)
        assert 'suggestions' in result4


class TestBehavioralProfiles:
    """Testes específicos para perfis comportamentais"""

    def test_night_profile_detection(self):
        """Testa detecção de perfil notívago"""
        from ml.otimizador_gastos import GastosAnalyzer

        # Criar transações noturnas
        transactions = []
        for i in range(20):
            transactions.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().replace(hour=2 + (i % 4)).isoformat(),
                'amount': 100,
                'category': 'delivery'
            })

        analyzer = GastosAnalyzer()
        result = analyzer.identify_behavioral_profile(transactions)

        # Deve identificar tendência notívaga
        scores = result.get('all_scores', {})
        assert 'notívago' in scores or result['profile'] == 'notívago'

    def test_social_profile_detection(self):
        """Testa detecção de perfil social"""
        from ml.otimizador_gastos import GastosAnalyzer

        # Criar transações de lazer/social
        transactions = []
        for i in range(30):
            transactions.append({
                'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat(),
                'amount': 80,
                'category': random.choice(['lazer', 'delivery', 'restaurante'])
            })

        analyzer = GastosAnalyzer()
        result = analyzer.identify_behavioral_profile(transactions)

        scores = result.get('all_scores', {})
        assert 'social' in scores


class TestTrendAnalysis:
    """Testes para análise de tendências"""

    def test_trend_direction(self):
        """Testa identificação de direção da tendência"""
        from ml.otimizador_gastos import GastosAnalyzer
        import pandas as pd

        # Criar tendência crescente
        transactions = []
        base_date = datetime.now() - timedelta(days=30)

        for i in range(30):
            transactions.append({
                'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                'amount': 50 + i * 5  # Valores crescentes
            })

        analyzer = GastosAnalyzer()
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])

        result = analyzer._analyze_trends(df)

        assert 'direction' in result
        assert result['direction'] == 'crescente'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
