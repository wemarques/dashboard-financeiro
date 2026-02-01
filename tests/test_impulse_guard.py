"""
Testes para o módulo de proteção contra impulsos
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, time

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from behavioral.impulse_guard import ImpulseGuard, check_transaction_risk


class TestImpulseGuard:
    """Testes para a classe ImpulseGuard"""

    def setup_method(self):
        """Setup para cada teste"""
        self.guard = ImpulseGuard(
            night_start="00:00",
            night_end="06:00",
            amount_threshold=100.0
        )

    def test_is_night_period_during_night(self):
        """Testa detecção de período noturno às 2h"""
        night_time = datetime(2024, 1, 15, 2, 30)  # 2:30 AM
        assert self.guard.is_night_period(night_time) is True

    def test_is_night_period_during_day(self):
        """Testa detecção de período diurno às 14h"""
        day_time = datetime(2024, 1, 15, 14, 30)  # 2:30 PM
        assert self.guard.is_night_period(day_time) is False

    def test_is_night_period_at_boundary_start(self):
        """Testa detecção no início do período noturno"""
        boundary_time = datetime(2024, 1, 15, 0, 0)  # Meia-noite
        assert self.guard.is_night_period(boundary_time) is True

    def test_is_night_period_at_boundary_end(self):
        """Testa detecção no fim do período noturno"""
        boundary_time = datetime(2024, 1, 15, 6, 0)  # 6:00 AM
        assert self.guard.is_night_period(boundary_time) is True

    def test_hour_risk_multiplier_peak(self):
        """Testa multiplicador de risco no pico (2-3h)"""
        peak_time = datetime(2024, 1, 15, 2, 30)
        multiplier = self.guard.get_hour_risk_multiplier(peak_time)
        assert multiplier == 2.0

    def test_hour_risk_multiplier_day(self):
        """Testa multiplicador de risco durante o dia"""
        day_time = datetime(2024, 1, 15, 14, 0)
        multiplier = self.guard.get_hour_risk_multiplier(day_time)
        assert multiplier == 1.0

    def test_calculate_risk_score_low_risk(self):
        """Testa cálculo de risco para transação normal"""
        day_time = datetime(2024, 1, 15, 10, 0)
        result = self.guard.calculate_risk_score(
            amount=50.0,
            category="alimentação",
            transaction_time=day_time
        )
        assert result["score"] < 30
        assert result["is_high_risk"] is False

    def test_calculate_risk_score_high_amount(self):
        """Testa cálculo de risco para valor alto"""
        day_time = datetime(2024, 1, 15, 10, 0)
        result = self.guard.calculate_risk_score(
            amount=500.0,  # 5x o limite
            category="alimentação",
            transaction_time=day_time
        )
        assert result["score"] >= 25  # Pelo menos o fator de valor

    def test_calculate_risk_score_night_high_risk_category(self):
        """Testa cálculo de risco para compra noturna em categoria de risco"""
        night_time = datetime(2024, 1, 15, 2, 0)
        result = self.guard.calculate_risk_score(
            amount=150.0,
            category="jogos",
            transaction_time=night_time
        )
        assert result["is_high_risk"] is True
        assert result["is_night"] is True

    def test_check_transaction_allowed(self):
        """Testa verificação de transação permitida"""
        day_time = datetime(2024, 1, 15, 10, 0)
        result = self.guard.check_transaction(
            amount=50.0,
            category="alimentação",
            transaction_time=day_time
        )
        assert result["allowed"] is True

    def test_check_transaction_blocked(self):
        """Testa verificação de transação bloqueada"""
        night_time = datetime(2024, 1, 15, 3, 0)
        result = self.guard.check_transaction(
            amount=200.0,
            category="jogos",
            transaction_time=night_time
        )
        assert result["allowed"] is False
        assert result["is_high_risk"] is True

    def test_enable_disable_protection(self):
        """Testa ativação/desativação da proteção"""
        self.guard.disable_protection()
        status = self.guard.get_protection_status()
        assert status["enabled"] is False

        self.guard.enable_protection()
        status = self.guard.get_protection_status()
        assert status["enabled"] is True

    def test_temporary_bypass(self):
        """Testa bypass temporário"""
        bypass_until = self.guard.temporary_bypass(minutes=30)
        assert bypass_until > datetime.now()

        status = self.guard.get_protection_status()
        assert status["bypass_active"] is True


class TestCheckTransactionRisk:
    """Testes para função de conveniência"""

    def test_check_transaction_risk_returns_dict(self):
        """Testa que a função retorna dicionário"""
        result = check_transaction_risk(amount=100.0, category="lazer")
        assert isinstance(result, dict)
        assert "score" in result
        assert "allowed" in result

    def test_check_transaction_risk_with_description(self):
        """Testa verificação com descrição"""
        result = check_transaction_risk(
            amount=50.0,
            category="alimentação",
            description="Supermercado"
        )
        assert "recommendation" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
