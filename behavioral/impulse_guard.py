"""
Sistema de proteção contra compras por impulso
Monitora horários de risco e valores suspeitos
"""
from datetime import datetime, time
from typing import Dict, Any, Optional, List, Tuple

try:
    from config import (
        NIGHT_START, NIGHT_END,
        IMPULSE_AMOUNT_THRESHOLD,
        IMPULSE_RISK_SCORE_THRESHOLD,
        DELAY_MINUTES_HIGH_RISK,
        BEHAVIORAL_PROFILES
    )
    from utils.logger import get_logger, log_alert
except ImportError:
    NIGHT_START = "00:00"
    NIGHT_END = "06:00"
    IMPULSE_AMOUNT_THRESHOLD = 100.0
    IMPULSE_RISK_SCORE_THRESHOLD = 70
    DELAY_MINUTES_HIGH_RISK = 5
    BEHAVIORAL_PROFILES = {}
    from logger import get_logger, log_alert

logger = get_logger(__name__)


class ImpulseGuard:
    """Sistema de proteção contra compras por impulso"""

    # Categorias de alto risco para compras noturnas
    HIGH_RISK_CATEGORIES = [
        "jogos",
        "delivery",
        "lazer",
        "compras",
        "assinaturas"
    ]

    # Multiplicadores de risco por hora
    HOUR_RISK_MULTIPLIERS = {
        0: 1.5,   # Meia-noite
        1: 1.8,   # 1h
        2: 2.0,   # 2h - pico de risco
        3: 2.0,   # 3h - pico de risco
        4: 1.8,   # 4h
        5: 1.5,   # 5h
        6: 1.0,   # 6h - início do dia
    }

    def __init__(
        self,
        night_start: str = None,
        night_end: str = None,
        amount_threshold: float = None
    ):
        """
        Inicializa o ImpulseGuard

        Args:
            night_start: Horário de início da proteção noturna (HH:MM)
            night_end: Horário de fim da proteção noturna (HH:MM)
            amount_threshold: Limite de valor para alertas
        """
        self.night_start = self._parse_time(night_start or NIGHT_START)
        self.night_end = self._parse_time(night_end or NIGHT_END)
        self.amount_threshold = amount_threshold or IMPULSE_AMOUNT_THRESHOLD

        self._protection_enabled = True
        self._temporary_bypass_until = None

        logger.info(
            f"ImpulseGuard inicializado: "
            f"proteção {self.night_start.strftime('%H:%M')}-{self.night_end.strftime('%H:%M')}, "
            f"limite R${self.amount_threshold:.2f}"
        )

    def _parse_time(self, time_str: str) -> time:
        """Converte string HH:MM para objeto time"""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def is_night_period(self, check_time: datetime = None) -> bool:
        """
        Verifica se o horário está no período noturno

        Args:
            check_time: Horário a verificar (usa atual se não especificado)

        Returns:
            True se estiver no período noturno
        """
        check_time = check_time or datetime.now()
        current = check_time.time()

        # Caso especial: período cruza meia-noite
        if self.night_start > self.night_end:
            return current >= self.night_start or current <= self.night_end
        else:
            return self.night_start <= current <= self.night_end

    def get_hour_risk_multiplier(self, check_time: datetime = None) -> float:
        """
        Retorna multiplicador de risco baseado na hora

        Args:
            check_time: Horário a verificar

        Returns:
            Multiplicador de risco (1.0 = normal)
        """
        check_time = check_time or datetime.now()
        hour = check_time.hour
        return self.HOUR_RISK_MULTIPLIERS.get(hour, 1.0)

    def calculate_risk_score(
        self,
        amount: float,
        category: str,
        transaction_time: datetime = None,
        recent_transactions_count: int = 0
    ) -> Dict[str, Any]:
        """
        Calcula score de risco de uma transação

        Args:
            amount: Valor da transação
            category: Categoria da transação
            transaction_time: Horário da transação
            recent_transactions_count: Número de transações recentes (última hora)

        Returns:
            Dicionário com score de risco e detalhes
        """
        transaction_time = transaction_time or datetime.now()
        risk_factors = []
        base_score = 0

        # 1. Verificar período noturno
        is_night = self.is_night_period(transaction_time)
        if is_night:
            hour_multiplier = self.get_hour_risk_multiplier(transaction_time)
            base_score += 30 * hour_multiplier
            risk_factors.append({
                "factor": "horário_noturno",
                "description": f"Transação às {transaction_time.strftime('%H:%M')}",
                "score": 30 * hour_multiplier
            })

        # 2. Verificar valor
        if amount >= self.amount_threshold:
            amount_factor = min(amount / self.amount_threshold, 3.0)
            amount_score = 25 * amount_factor
            base_score += amount_score
            risk_factors.append({
                "factor": "valor_alto",
                "description": f"R${amount:.2f} acima do limite de R${self.amount_threshold:.2f}",
                "score": amount_score
            })

        # 3. Verificar categoria de risco
        if category and category.lower() in self.HIGH_RISK_CATEGORIES:
            category_score = 20
            if is_night:
                category_score *= 1.5  # Categoria de risco + noite = mais perigoso
            base_score += category_score
            risk_factors.append({
                "factor": "categoria_risco",
                "description": f"Categoria '{category}' é considerada de alto risco",
                "score": category_score
            })

        # 4. Verificar frequência
        if recent_transactions_count >= 3:
            frequency_score = min(recent_transactions_count * 5, 20)
            base_score += frequency_score
            risk_factors.append({
                "factor": "frequência_alta",
                "description": f"{recent_transactions_count} transações na última hora",
                "score": frequency_score
            })

        # Normalizar para 0-100
        final_score = min(int(base_score), 100)

        return {
            "score": final_score,
            "is_high_risk": final_score >= IMPULSE_RISK_SCORE_THRESHOLD,
            "is_night": is_night,
            "risk_factors": risk_factors,
            "recommendation": self._get_recommendation(final_score, risk_factors)
        }

    def _get_recommendation(self, score: int, risk_factors: List[Dict]) -> Dict[str, Any]:
        """Gera recomendação baseada no score de risco"""
        if score < 30:
            return {
                "level": "low",
                "message": "Transação dentro do padrão normal.",
                "action": "proceed",
                "delay_minutes": 0
            }
        elif score < 50:
            return {
                "level": "medium",
                "message": "Considere se esta compra é realmente necessária.",
                "action": "confirm",
                "delay_minutes": 0,
                "questions": [
                    "Esta compra estava planejada?",
                    "Você pode esperar até amanhã para decidir?"
                ]
            }
        elif score < 70:
            return {
                "level": "high",
                "message": "Alerta: Este gasto pode ser por impulso.",
                "action": "delay",
                "delay_minutes": DELAY_MINUTES_HIGH_RISK,
                "questions": [
                    "Por que você quer fazer esta compra agora?",
                    "Como vai se sentir amanhã com esta decisão?",
                    "Esta compra te aproxima de suas metas financeiras?"
                ]
            }
        else:
            return {
                "level": "critical",
                "message": "⚠️ ATENÇÃO: Alto risco de compra por impulso!",
                "action": "block",
                "delay_minutes": DELAY_MINUTES_HIGH_RISK * 2,
                "questions": [
                    "Você está sob estresse ou ansiedade agora?",
                    "Já fez compras que se arrependeu em horários como este?",
                    "O que aconteceria se você não comprasse isso?"
                ]
            }

    def check_transaction(
        self,
        amount: float,
        category: str = None,
        description: str = None,
        transaction_time: datetime = None,
        recent_transactions_count: int = 0
    ) -> Dict[str, Any]:
        """
        Verifica uma transação e retorna análise completa

        Args:
            amount: Valor da transação
            category: Categoria da transação
            description: Descrição da transação
            transaction_time: Horário da transação
            recent_transactions_count: Transações recentes

        Returns:
            Dicionário com análise completa
        """
        # Verificar se proteção está ativa
        if not self._protection_enabled:
            return {
                "allowed": True,
                "score": 0,
                "message": "Proteção desativada",
                "protection_enabled": False
            }

        # Verificar bypass temporário
        if self._temporary_bypass_until and datetime.now() < self._temporary_bypass_until:
            return {
                "allowed": True,
                "score": 0,
                "message": f"Bypass ativo até {self._temporary_bypass_until.strftime('%H:%M')}",
                "protection_enabled": True,
                "bypass_active": True
            }

        # Calcular risco
        risk_analysis = self.calculate_risk_score(
            amount=amount,
            category=category,
            transaction_time=transaction_time,
            recent_transactions_count=recent_transactions_count
        )

        # Logar alerta se necessário
        if risk_analysis["is_high_risk"]:
            log_alert(
                logger,
                alert_type="impulse" if not risk_analysis["is_night"] else "night",
                message=f"Transação de R${amount:.2f} em {category or 'categoria desconhecida'}",
                risk_score=risk_analysis["score"]
            )

        return {
            "allowed": not risk_analysis["is_high_risk"],
            "protection_enabled": True,
            **risk_analysis
        }

    def enable_protection(self) -> None:
        """Ativa a proteção contra impulso"""
        self._protection_enabled = True
        self._temporary_bypass_until = None
        logger.info("Proteção contra impulso ATIVADA")

    def disable_protection(self) -> None:
        """Desativa a proteção contra impulso"""
        self._protection_enabled = False
        logger.warning("Proteção contra impulso DESATIVADA")

    def temporary_bypass(self, minutes: int = 30) -> datetime:
        """
        Cria um bypass temporário da proteção

        Args:
            minutes: Duração do bypass em minutos

        Returns:
            Datetime até quando o bypass estará ativo
        """
        from datetime import timedelta
        self._temporary_bypass_until = datetime.now() + timedelta(minutes=minutes)
        logger.warning(f"Bypass temporário ativado até {self._temporary_bypass_until.strftime('%H:%M')}")
        return self._temporary_bypass_until

    def get_protection_status(self) -> Dict[str, Any]:
        """Retorna status atual da proteção"""
        now = datetime.now()
        return {
            "enabled": self._protection_enabled,
            "is_night_period": self.is_night_period(now),
            "current_hour_risk": self.get_hour_risk_multiplier(now),
            "bypass_active": self._temporary_bypass_until and now < self._temporary_bypass_until,
            "bypass_until": self._temporary_bypass_until.isoformat() if self._temporary_bypass_until else None,
            "night_start": self.night_start.strftime("%H:%M"),
            "night_end": self.night_end.strftime("%H:%M"),
            "amount_threshold": self.amount_threshold
        }


# === Instância global e funções de conveniência ===

_guard = None


def get_impulse_guard() -> ImpulseGuard:
    """Retorna instância global do ImpulseGuard"""
    global _guard
    if _guard is None:
        _guard = ImpulseGuard()
    return _guard


def check_transaction_risk(
    amount: float,
    category: str = None,
    description: str = None,
    recent_transactions_count: int = 0
) -> Dict[str, Any]:
    """
    Verifica risco de uma transação (função de conveniência)

    Args:
        amount: Valor da transação
        category: Categoria da transação
        description: Descrição da transação
        recent_transactions_count: Transações recentes

    Returns:
        Dicionário com análise de risco
    """
    guard = get_impulse_guard()
    return guard.check_transaction(
        amount=amount,
        category=category,
        description=description,
        recent_transactions_count=recent_transactions_count
    )


def is_night_mode() -> bool:
    """Verifica se está no período noturno"""
    guard = get_impulse_guard()
    return guard.is_night_period()
