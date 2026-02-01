"""
Testes para o módulo de intervenções comportamentais
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInterventionEngine:
    """Testes para o motor de intervenções"""

    def test_import_module(self):
        """Testa importação do módulo"""
        from behavioral.intervention import InterventionEngine
        assert InterventionEngine is not None

    def test_engine_initialization(self):
        """Testa inicialização do motor"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()
        assert engine is not None
        assert engine.active_delays == {}

    def test_generate_intervention_low_risk(self):
        """Testa geração de intervenção para baixo risco"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()

        transaction = {
            'amount': 50,
            'category': 'alimentação',
            'merchant': 'Supermercado'
        }

        result = engine.generate_intervention(transaction, risk_score=20)

        assert 'level' in result
        assert result['level'] == 'gentle'
        assert 'components' in result

    def test_generate_intervention_high_risk(self):
        """Testa geração de intervenção para alto risco"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()

        transaction = {
            'amount': 500,
            'category': 'jogos',
            'merchant': 'Bet Online'
        }

        result = engine.generate_intervention(transaction, risk_score=80)

        assert 'level' in result
        assert result['level'] in ['strong', 'critical']
        assert 'components' in result

        # Deve ter delay ou bloqueio
        component_types = [c['type'] for c in result['components']]
        assert 'delay' in component_types or 'block' in component_types

    def test_questions_for_category(self):
        """Testa perguntas específicas por categoria"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()

        questions = engine._get_questions('delivery', 100, is_night=False)

        assert len(questions) > 0
        # Deve ter pergunta relacionada a delivery
        assert any('comida' in q.lower() or 'delivery' in q.lower() or 'cozinhar' in q.lower()
                   for q in questions)

    def test_questions_for_night(self):
        """Testa perguntas para horário noturno"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()

        questions = engine._get_questions('compras', 200, is_night=True)

        assert len(questions) > 0
        # Deve ter pergunta sobre horário
        assert any('madrugada' in q.lower() or 'descansado' in q.lower() or 'amanhã' in q.lower()
                   for q in questions)

    def test_impact_visualization(self):
        """Testa geração de visualização de impacto"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()

        impact = engine._generate_impact_visualization(amount=200)

        assert 'daily_food' in impact
        assert 'subscriptions' in impact
        assert 'investment' in impact
        assert 'work_hours' in impact

    def test_delay_management(self):
        """Testa gerenciamento de delays"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()

        # Definir delay
        result = engine.set_delay('user1', 'trans1', minutes=5)
        assert result['success'] is True

        # Verificar status
        status = engine.check_delay_status('user1', 'trans1')
        assert status['active'] is True
        assert status['can_proceed'] is False

    def test_intervention_levels(self):
        """Testa determinação de níveis de intervenção"""
        from behavioral.intervention import InterventionEngine, InterventionLevel

        engine = InterventionEngine()

        assert engine._determine_level(10) == InterventionLevel.GENTLE
        assert engine._determine_level(40) == InterventionLevel.MODERATE
        assert engine._determine_level(60) == InterventionLevel.STRONG
        assert engine._determine_level(85) == InterventionLevel.CRITICAL

    def test_suggested_actions(self):
        """Testa ações sugeridas por nível"""
        from behavioral.intervention import InterventionEngine, InterventionLevel

        engine = InterventionEngine()

        actions_gentle = engine._get_suggested_actions(InterventionLevel.GENTLE)
        assert any(a['action'] == 'proceed' for a in actions_gentle)

        actions_critical = engine._get_suggested_actions(InterventionLevel.CRITICAL)
        assert any(a['action'] == 'cancel' for a in actions_critical)


class TestConvenienceFunctions:
    """Testes para funções de conveniência"""

    def test_generate_intervention_function(self):
        """Testa função de conveniência"""
        from behavioral.intervention import generate_intervention

        transaction = {'amount': 100, 'category': 'compras'}
        result = generate_intervention(transaction, risk_score=50)

        assert 'level' in result
        assert 'components' in result

    def test_get_reflective_questions(self):
        """Testa obtenção de perguntas reflexivas"""
        from behavioral.intervention import get_reflective_questions

        questions = get_reflective_questions('delivery', count=3)

        assert len(questions) <= 3
        assert all(isinstance(q, str) for q in questions)


class TestInterventionWithGoals:
    """Testes de intervenção com metas"""

    def test_compare_with_goals(self):
        """Testa comparação com metas"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()

        goals = [
            {
                'name': 'Reserva de Emergência',
                'target_amount': 10000,
                'current_amount': 5000,
                'status': 'active'
            }
        ]

        result = engine._compare_with_goals(amount=500, goals=goals)

        assert result is not None
        assert 'comparisons' in result
        assert len(result['comparisons']) > 0

    def test_intervention_with_goals(self):
        """Testa intervenção considerando metas"""
        from behavioral.intervention import InterventionEngine

        engine = InterventionEngine()

        transaction = {'amount': 300, 'category': 'compras'}
        goals = [
            {
                'name': 'Viagem',
                'target_amount': 5000,
                'current_amount': 2000,
                'status': 'active'
            }
        ]

        result = engine.generate_intervention(
            transaction,
            risk_score=60,
            goals=goals
        )

        # Deve ter componente de comparação com metas
        component_types = [c['type'] for c in result['components']]
        assert 'comparison' in component_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
