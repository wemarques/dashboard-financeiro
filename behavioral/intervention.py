"""
Sistema de Intervenções Comportamentais
Gera perguntas reflexivas, delays e intervenções personalizadas
para ajudar na tomada de decisão financeira
"""
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

try:
    from config import DELAY_MINUTES_HIGH_RISK, IMPULSE_RISK_SCORE_THRESHOLD
    from utils.logger import get_logger
except ImportError:
    DELAY_MINUTES_HIGH_RISK = 5
    IMPULSE_RISK_SCORE_THRESHOLD = 70
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class InterventionType(Enum):
    """Tipos de intervenção"""
    QUESTION = "question"           # Pergunta reflexiva
    DELAY = "delay"                 # Atraso na confirmação
    COMPARISON = "comparison"       # Comparação com metas
    VISUALIZATION = "visualization" # Visualização do impacto
    ALTERNATIVE = "alternative"     # Sugestão de alternativa
    BLOCK = "block"                 # Bloqueio temporário


class InterventionLevel(Enum):
    """Níveis de intensidade da intervenção"""
    GENTLE = "gentle"       # Sutil, não intrusivo
    MODERATE = "moderate"   # Moderado, requer atenção
    STRONG = "strong"       # Forte, requer ação
    CRITICAL = "critical"   # Crítico, bloqueio


class InterventionEngine:
    """
    Motor de intervenções comportamentais.

    Gera intervenções personalizadas baseadas no contexto da transação,
    histórico do usuário e perfil comportamental.
    """

    # Perguntas reflexivas por categoria
    REFLECTIVE_QUESTIONS = {
        'default': [
            "Esta compra estava no seu planejamento?",
            "Como você vai se sentir amanhã com esta decisão?",
            "Você realmente precisa disso agora?",
            "O que aconteceria se você esperasse uma semana?",
            "Esta compra te aproxima ou afasta das suas metas?"
        ],
        'delivery': [
            "Você tem comida em casa que poderia preparar?",
            "Quanto você já gastou com delivery esta semana?",
            "Cozinhar não seria mais saudável e econômico?",
            "Este pedido é por fome real ou por impulso?"
        ],
        'jogos': [
            "Quanto tempo e dinheiro você investiu em jogos este mês?",
            "Este gasto vai te trazer satisfação duradoura?",
            "Você está jogando para se divertir ou para escapar?",
            "O que mais você poderia fazer com esse valor?"
        ],
        'compras': [
            "Você pesquisou preços em outros lugares?",
            "Este item vai ser usado regularmente?",
            "Você tem algo similar que poderia usar?",
            "Por que você quer isso agora e não daqui a uma semana?"
        ],
        'lazer': [
            "Existem alternativas gratuitas para este programa?",
            "Quanto você já gastou com lazer este mês?",
            "Este momento de lazer será memorável?",
            "Você poderia fazer algo igualmente prazeroso gastando menos?"
        ],
        'noturno': [
            "Você está tomando esta decisão descansado?",
            "Compras de madrugada tendem a ser por impulso. É o caso?",
            "O que te levou a querer comprar isso agora, às Xh?",
            "Amanhã de manhã você ainda vai querer isso?"
        ],
        'alto_valor': [
            "Você tem reserva de emergência?",
            "Este valor representa quanto do seu salário?",
            "Você conversou com alguém sobre esta compra?",
            "Este gasto está no seu orçamento mensal?"
        ]
    }

    # Mensagens de visualização de impacto
    IMPACT_MESSAGES = {
        'daily': "Com esse valor, você poderia cobrir {days} dias de alimentação",
        'goal': "Esse gasto representa {percent}% da sua meta de poupança",
        'subscription': "Esse valor é equivalente a {months} meses de Netflix",
        'investment': "Investido a 10% ao ano, em 5 anos seria R$ {future_value}",
        'hourly': "Você trabalha {hours} horas para ganhar esse valor"
    }

    # Alternativas sugeridas por categoria
    ALTERNATIVES = {
        'delivery': [
            "Preparar uma refeição simples em casa",
            "Pedir algo mais barato no mesmo lugar",
            "Usar cupom de desconto se disponível",
            "Dividir o pedido com alguém"
        ],
        'compras': [
            "Adicionar à lista de desejos e esperar 48h",
            "Procurar produto similar usado",
            "Esperar uma promoção ou Black Friday",
            "Verificar se já não tem algo parecido"
        ],
        'lazer': [
            "Buscar eventos gratuitos na cidade",
            "Fazer um programa em casa",
            "Usar pontos ou milhas acumulados",
            "Combinar com amigos e dividir custos"
        ],
        'assinaturas': [
            "Cancelar assinaturas não utilizadas primeiro",
            "Buscar plano família ou estudante",
            "Usar período de teste gratuito",
            "Alternar entre serviços mensalmente"
        ]
    }

    def __init__(self):
        """Inicializa o motor de intervenções"""
        self.active_delays = {}  # Delays ativos por usuário
        self.intervention_history = []  # Histórico de intervenções

        logger.info("InterventionEngine inicializado")

    def generate_intervention(
        self,
        transaction: Dict[str, Any],
        risk_score: int,
        user_profile: Optional[Dict[str, Any]] = None,
        goals: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Gera intervenção apropriada para a transação.

        Args:
            transaction: Dados da transação
            risk_score: Score de risco (0-100)
            user_profile: Perfil comportamental do usuário
            goals: Metas financeiras do usuário

        Returns:
            Intervenção com tipo, conteúdo e ações
        """
        amount = transaction.get('amount', 0)
        category = transaction.get('category', 'outros').lower()
        hour = datetime.now().hour
        is_night = 0 <= hour <= 6

        # Determinar nível da intervenção
        level = self._determine_level(risk_score)

        # Gerar componentes da intervenção
        intervention = {
            'level': level.value,
            'risk_score': risk_score,
            'timestamp': datetime.now().isoformat(),
            'components': []
        }

        # 1. Perguntas reflexivas (sempre incluir pelo menos uma)
        questions = self._get_questions(category, amount, is_night)
        intervention['components'].append({
            'type': InterventionType.QUESTION.value,
            'content': questions
        })

        # 2. Visualização de impacto (para valores significativos)
        if amount >= 50:
            impact = self._generate_impact_visualization(amount, goals)
            intervention['components'].append({
                'type': InterventionType.VISUALIZATION.value,
                'content': impact
            })

        # 3. Comparação com metas (se houver metas)
        if goals:
            comparison = self._compare_with_goals(amount, goals)
            if comparison:
                intervention['components'].append({
                    'type': InterventionType.COMPARISON.value,
                    'content': comparison
                })

        # 4. Alternativas (para categorias específicas)
        if category in self.ALTERNATIVES:
            alternatives = random.sample(
                self.ALTERNATIVES[category],
                min(2, len(self.ALTERNATIVES[category]))
            )
            intervention['components'].append({
                'type': InterventionType.ALTERNATIVE.value,
                'content': alternatives
            })

        # 5. Delay obrigatório (para alto risco)
        if level in [InterventionLevel.STRONG, InterventionLevel.CRITICAL]:
            delay_minutes = DELAY_MINUTES_HIGH_RISK
            if level == InterventionLevel.CRITICAL:
                delay_minutes *= 2

            intervention['components'].append({
                'type': InterventionType.DELAY.value,
                'content': {
                    'minutes': delay_minutes,
                    'expires_at': (datetime.now() + timedelta(minutes=delay_minutes)).isoformat(),
                    'message': f"Aguarde {delay_minutes} minutos antes de confirmar"
                }
            })

        # 6. Bloqueio (para nível crítico)
        if level == InterventionLevel.CRITICAL:
            intervention['components'].append({
                'type': InterventionType.BLOCK.value,
                'content': {
                    'reason': self._get_block_reason(risk_score, is_night, amount),
                    'can_override': True,
                    'override_requires': 'confirmation_code'
                }
            })

        # Gerar mensagem principal
        intervention['main_message'] = self._generate_main_message(level, category, is_night)

        # Ações sugeridas
        intervention['actions'] = self._get_suggested_actions(level)

        # Registrar no histórico
        self.intervention_history.append({
            'transaction': transaction,
            'intervention': intervention,
            'timestamp': datetime.now().isoformat()
        })

        return intervention

    def _determine_level(self, risk_score: int) -> InterventionLevel:
        """Determina nível da intervenção baseado no score"""
        if risk_score < 30:
            return InterventionLevel.GENTLE
        elif risk_score < 50:
            return InterventionLevel.MODERATE
        elif risk_score < 75:
            return InterventionLevel.STRONG
        else:
            return InterventionLevel.CRITICAL

    def _get_questions(
        self,
        category: str,
        amount: float,
        is_night: bool
    ) -> List[str]:
        """Seleciona perguntas reflexivas apropriadas"""
        questions = []

        # Perguntas específicas da categoria
        if category in self.REFLECTIVE_QUESTIONS:
            questions.extend(
                random.sample(
                    self.REFLECTIVE_QUESTIONS[category],
                    min(2, len(self.REFLECTIVE_QUESTIONS[category]))
                )
            )

        # Perguntas para horário noturno
        if is_night:
            night_questions = self.REFLECTIVE_QUESTIONS['noturno']
            q = random.choice(night_questions)
            q = q.replace('Xh', f'{datetime.now().hour}h')
            questions.append(q)

        # Perguntas para alto valor
        if amount >= 200:
            questions.extend(
                random.sample(
                    self.REFLECTIVE_QUESTIONS['alto_valor'],
                    min(1, len(self.REFLECTIVE_QUESTIONS['alto_valor']))
                )
            )

        # Garantir pelo menos uma pergunta default
        if not questions:
            questions = random.sample(self.REFLECTIVE_QUESTIONS['default'], 2)

        return questions[:4]  # Máximo 4 perguntas

    def _generate_impact_visualization(
        self,
        amount: float,
        goals: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Gera visualização do impacto financeiro"""
        impact = {}

        # Dias de alimentação (considerando R$30/dia)
        days = amount / 30
        impact['daily_food'] = f"Com R$ {amount:.2f}, você poderia comer por {days:.1f} dias"

        # Equivalente em assinaturas
        netflix_months = amount / 45  # ~R$45/mês
        impact['subscriptions'] = f"Equivale a {netflix_months:.1f} meses de streaming"

        # Valor futuro (investimento)
        future_value = amount * (1.10 ** 5)  # 10% a.a. por 5 anos
        impact['investment'] = f"Investido, em 5 anos seria R$ {future_value:.2f}"

        # Horas de trabalho (salário mínimo ~R$15/hora)
        hours = amount / 15
        impact['work_hours'] = f"Você trabalha {hours:.1f} horas para ganhar isso"

        # Progresso da meta (se houver)
        if goals:
            active_goal = next((g for g in goals if g.get('status') == 'active'), None)
            if active_goal:
                target = active_goal.get('target_amount', 1)
                percent = (amount / target) * 100
                impact['goal_impact'] = f"Este valor é {percent:.1f}% da sua meta '{active_goal.get('name')}'"

        return impact

    def _compare_with_goals(
        self,
        amount: float,
        goals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Compara gasto com metas do usuário"""
        if not goals:
            return None

        active_goals = [g for g in goals if g.get('status') == 'active']

        if not active_goals:
            return None

        comparisons = []
        for goal in active_goals:
            remaining = goal.get('target_amount', 0) - goal.get('current_amount', 0)
            if remaining > 0:
                percent_of_remaining = (amount / remaining) * 100
                comparisons.append({
                    'goal_name': goal.get('name'),
                    'remaining': remaining,
                    'this_purchase_percent': round(percent_of_remaining, 1),
                    'message': f"Este gasto representa {percent_of_remaining:.1f}% do que falta para '{goal.get('name')}'"
                })

        return {
            'comparisons': comparisons,
            'summary': f"Você tem {len(active_goals)} meta(s) ativa(s)"
        }

    def _get_block_reason(
        self,
        risk_score: int,
        is_night: bool,
        amount: float
    ) -> str:
        """Gera razão para bloqueio"""
        reasons = []

        if is_night:
            reasons.append("transação em horário de alto risco (madrugada)")
        if risk_score >= 80:
            reasons.append("padrão de compra por impulso detectado")
        if amount >= 500:
            reasons.append("valor significativo requer reflexão")

        if not reasons:
            reasons.append("múltiplos fatores de risco identificados")

        return "Bloqueio ativado devido a: " + ", ".join(reasons)

    def _generate_main_message(
        self,
        level: InterventionLevel,
        category: str,
        is_night: bool
    ) -> str:
        """Gera mensagem principal da intervenção"""
        messages = {
            InterventionLevel.GENTLE: [
                "Tudo bem prosseguir, mas considere estas reflexões:",
                "Antes de confirmar, uma rápida reflexão:"
            ],
            InterventionLevel.MODERATE: [
                "⚠️ Atenção! Esta compra merece uma reflexão.",
                "⚠️ Pause um momento e considere:"
            ],
            InterventionLevel.STRONG: [
                "🚨 Alerta: Alto risco de compra por impulso!",
                "🚨 Cuidado! Este padrão indica possível impulso."
            ],
            InterventionLevel.CRITICAL: [
                "🛑 ATENÇÃO MÁXIMA: Bloqueio de segurança ativado!",
                "🛑 PARE! Esta compra foi temporariamente bloqueada."
            ]
        }

        base_message = random.choice(messages[level])

        if is_night:
            base_message += " [Modo Noturno Ativo]"

        return base_message

    def _get_suggested_actions(self, level: InterventionLevel) -> List[Dict[str, str]]:
        """Retorna ações sugeridas para o usuário"""
        actions = {
            InterventionLevel.GENTLE: [
                {'action': 'proceed', 'label': 'Prosseguir', 'style': 'primary'},
                {'action': 'cancel', 'label': 'Cancelar', 'style': 'secondary'}
            ],
            InterventionLevel.MODERATE: [
                {'action': 'reflect', 'label': 'Refletir mais', 'style': 'primary'},
                {'action': 'proceed', 'label': 'Prosseguir mesmo assim', 'style': 'secondary'},
                {'action': 'cancel', 'label': 'Cancelar', 'style': 'secondary'}
            ],
            InterventionLevel.STRONG: [
                {'action': 'wait', 'label': 'Aguardar período de reflexão', 'style': 'primary'},
                {'action': 'add_to_wishlist', 'label': 'Adicionar à lista de desejos', 'style': 'secondary'},
                {'action': 'cancel', 'label': 'Cancelar', 'style': 'danger'}
            ],
            InterventionLevel.CRITICAL: [
                {'action': 'cancel', 'label': 'Cancelar (Recomendado)', 'style': 'danger'},
                {'action': 'override', 'label': 'Desbloquear (requer confirmação)', 'style': 'warning'}
            ]
        }

        return actions.get(level, actions[InterventionLevel.GENTLE])

    def check_delay_status(self, user_id: str, transaction_id: str) -> Dict[str, Any]:
        """
        Verifica status de um delay ativo.

        Args:
            user_id: ID do usuário
            transaction_id: ID da transação

        Returns:
            Status do delay
        """
        key = f"{user_id}_{transaction_id}"

        if key not in self.active_delays:
            return {'active': False}

        delay_info = self.active_delays[key]
        expires_at = datetime.fromisoformat(delay_info['expires_at'])

        if datetime.now() >= expires_at:
            del self.active_delays[key]
            return {
                'active': False,
                'expired': True,
                'can_proceed': True
            }

        remaining = (expires_at - datetime.now()).total_seconds()

        return {
            'active': True,
            'remaining_seconds': int(remaining),
            'remaining_minutes': round(remaining / 60, 1),
            'expires_at': expires_at.isoformat(),
            'can_proceed': False
        }

    def set_delay(
        self,
        user_id: str,
        transaction_id: str,
        minutes: int
    ) -> Dict[str, Any]:
        """
        Define um delay para transação.

        Args:
            user_id: ID do usuário
            transaction_id: ID da transação
            minutes: Minutos de delay

        Returns:
            Informações do delay criado
        """
        key = f"{user_id}_{transaction_id}"
        expires_at = datetime.now() + timedelta(minutes=minutes)

        self.active_delays[key] = {
            'created_at': datetime.now().isoformat(),
            'expires_at': expires_at.isoformat(),
            'minutes': minutes
        }

        logger.info(f"Delay de {minutes}min criado para {key}")

        return {
            'success': True,
            'expires_at': expires_at.isoformat(),
            'minutes': minutes
        }

    def get_intervention_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das intervenções"""
        if not self.intervention_history:
            return {'total': 0}

        total = len(self.intervention_history)
        by_level = {}

        for item in self.intervention_history:
            level = item['intervention'].get('level', 'unknown')
            by_level[level] = by_level.get(level, 0) + 1

        return {
            'total': total,
            'by_level': by_level,
            'active_delays': len(self.active_delays)
        }


# === Funções de conveniência ===

_engine = None


def get_intervention_engine() -> InterventionEngine:
    """Retorna instância global do motor de intervenções"""
    global _engine
    if _engine is None:
        _engine = InterventionEngine()
    return _engine


def generate_intervention(
    transaction: Dict[str, Any],
    risk_score: int,
    user_profile: Optional[Dict[str, Any]] = None,
    goals: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Gera intervenção para transação (função de conveniência)"""
    engine = get_intervention_engine()
    return engine.generate_intervention(transaction, risk_score, user_profile, goals)


def get_reflective_questions(category: str, count: int = 3) -> List[str]:
    """Retorna perguntas reflexivas para categoria"""
    engine = get_intervention_engine()
    questions = engine.REFLECTIVE_QUESTIONS.get(
        category,
        engine.REFLECTIVE_QUESTIONS['default']
    )
    return random.sample(questions, min(count, len(questions)))
