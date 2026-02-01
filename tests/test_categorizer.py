"""
Testes para o módulo de categorização automática
"""
import pytest
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.categorizer import Categorizer, categorize_transaction, get_category_suggestions


class TestCategorizer:
    """Testes para a classe Categorizer"""

    def setup_method(self):
        """Setup para cada teste"""
        self.categorizer = Categorizer()

    def test_categorize_by_rules_alimentacao(self):
        """Testa categorização de estabelecimentos de alimentação"""
        test_cases = [
            ("Supermercado Extra", "alimentação"),
            ("Carrefour", "alimentação"),
            ("Padaria do João", "alimentação"),
        ]

        for description, expected_category in test_cases:
            result = self.categorizer.categorize_by_rules(description)
            assert result[0] == expected_category, f"Falhou para: {description}"

    def test_categorize_by_rules_transporte(self):
        """Testa categorização de transporte"""
        test_cases = [
            ("Uber do Brasil", "transporte"),
            ("99 Táxi", "transporte"),
            ("Posto Shell", "transporte"),
        ]

        for description, expected_category in test_cases:
            result = self.categorizer.categorize_by_rules(description)
            assert result[0] == expected_category, f"Falhou para: {description}"

    def test_categorize_by_rules_delivery(self):
        """Testa categorização de delivery"""
        test_cases = [
            ("iFood", "delivery"),
            ("Rappi", "delivery"),
            ("Uber Eats", "delivery"),
        ]

        for description, expected_category in test_cases:
            result = self.categorizer.categorize_by_rules(description)
            assert result[0] == expected_category, f"Falhou para: {description}"

    def test_categorize_by_rules_unknown(self):
        """Testa categorização de descrição desconhecida"""
        result = self.categorizer.categorize_by_rules("XYZ Desconhecido 123")
        assert result[0] == "outros"
        assert result[1] < 0.5  # Baixa confiança

    def test_categorize_combined(self):
        """Testa método de categorização combinada"""
        result = self.categorizer.categorize("Pagamento Netflix", "Netflix")
        assert "category" in result
        assert "confidence" in result
        assert "method" in result

    def test_get_category_suggestions(self):
        """Testa sugestões de categoria"""
        suggestions = get_category_suggestions("uber")
        assert "transporte" in suggestions

        suggestions = get_category_suggestions("mercado")
        assert "alimentação" in suggestions


class TestCategorizeTransaction:
    """Testes para função de conveniência"""

    def test_categorize_transaction_returns_dict(self):
        """Testa que a função retorna dicionário"""
        result = categorize_transaction("Compra no Carrefour")
        assert isinstance(result, dict)
        assert "category" in result

    def test_categorize_transaction_with_merchant(self):
        """Testa categorização com nome do estabelecimento"""
        result = categorize_transaction("Compra", "Supermercado Extra")
        assert result["category"] == "alimentação"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
