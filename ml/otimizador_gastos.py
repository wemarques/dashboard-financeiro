"""
Otimizador de Gastos com Machine Learning
Inclui previsão de gastos, clustering de padrões e sugestões personalizadas
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from config import BEHAVIORAL_PROFILES, CATEGORIES
    from utils.logger import get_logger
except ImportError:
    BEHAVIORAL_PROFILES = {}
    CATEGORIES = []
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class GastosAnalyzer:
    """
    Analisador de gastos com técnicas de ML.

    Funcionalidades:
    - Previsão de gastos futuros (séries temporais)
    - Clustering de padrões de consumo (K-Means)
    - Identificação de perfil comportamental
    - Sugestões personalizadas de economia
    """

    # Perfis comportamentais predefinidos
    PROFILE_DEFINITIONS = {
        'notívago': {
            'description': 'Tendência a gastar mais durante a madrugada',
            'risk_level': 'alto',
            'characteristics': ['gastos entre 00h-06h', 'compras por impulso noturnas'],
            'suggestions': [
                'Evite ter apps de compras no celular',
                'Configure bloqueio de cartão à noite',
                'Deixe dinheiro separado para emergências'
            ]
        },
        'impulsivo': {
            'description': 'Compras frequentes não planejadas',
            'risk_level': 'alto',
            'characteristics': ['alta frequência de transações', 'valores variados'],
            'suggestions': [
                'Use a regra das 48h antes de comprar',
                'Faça lista antes de ir às compras',
                'Defina um limite diário de gastos'
            ]
        },
        'social': {
            'description': 'Gastos concentrados em lazer e alimentação fora',
            'risk_level': 'médio',
            'characteristics': ['delivery frequente', 'restaurantes', 'bares'],
            'suggestions': [
                'Cozinhe mais em casa',
                'Limite saídas a X vezes por semana',
                'Busque programas gratuitos de lazer'
            ]
        },
        'controlado': {
            'description': 'Padrão de gastos estável e previsível',
            'risk_level': 'baixo',
            'characteristics': ['gastos regulares', 'pouca variação'],
            'suggestions': [
                'Continue assim!',
                'Considere aumentar investimentos',
                'Explore otimizações em gastos fixos'
            ]
        },
        'sazonal': {
            'description': 'Gastos variam muito ao longo do mês',
            'risk_level': 'médio',
            'characteristics': ['picos em datas específicas', 'variação alta'],
            'suggestions': [
                'Distribua gastos ao longo do mês',
                'Antecipe compras de alto valor',
                'Crie reserva para picos sazonais'
            ]
        }
    }

    def __init__(self):
        """Inicializa o analisador de gastos"""
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.kmeans = None
        self.is_fitted = False
        self.historical_stats = {}

        logger.info("GastosAnalyzer inicializado")

    def analyze_spending_patterns(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analisa padrões de gastos nas transações.

        Args:
            transactions: Lista de transações

        Returns:
            Análise detalhada dos padrões
        """
        if not transactions:
            return {'error': 'Sem transações para analisar'}

        df = pd.DataFrame(transactions)

        # Garantir coluna de data
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
        else:
            return {'error': 'Dados sem informação de data'}

        # Análise por categoria
        category_analysis = self._analyze_by_category(df)

        # Análise temporal
        temporal_analysis = self._analyze_temporal_patterns(df)

        # Análise de tendências
        trend_analysis = self._analyze_trends(df)

        return {
            'summary': {
                'total_transactions': len(df),
                'total_spent': df['amount'].sum(),
                'average_transaction': df['amount'].mean(),
                'median_transaction': df['amount'].median(),
                'std_transaction': df['amount'].std()
            },
            'by_category': category_analysis,
            'temporal': temporal_analysis,
            'trends': trend_analysis
        }

    def _analyze_by_category(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa gastos por categoria"""
        if 'category' not in df.columns:
            return {}

        category_stats = df.groupby('category').agg({
            'amount': ['sum', 'mean', 'count', 'std']
        }).round(2)

        category_stats.columns = ['total', 'media', 'count', 'desvio']
        category_stats = category_stats.sort_values('total', ascending=False)

        total = df['amount'].sum()
        category_stats['percentual'] = (category_stats['total'] / total * 100).round(1)

        return category_stats.to_dict('index')

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões temporais"""
        result = {}

        # Por dia da semana
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        day_names = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        by_dow = df.groupby('day_of_week')['amount'].agg(['sum', 'mean', 'count'])
        by_dow.index = [day_names[i] for i in by_dow.index]
        result['by_day_of_week'] = by_dow.to_dict('index')

        # Por período do dia (se tiver timestamp)
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['period'] = pd.cut(
                df['hour'],
                bins=[-1, 6, 12, 18, 24],
                labels=['Madrugada', 'Manhã', 'Tarde', 'Noite']
            )
            by_period = df.groupby('period')['amount'].agg(['sum', 'count'])
            result['by_period'] = by_period.to_dict('index')

        # Por semana do mês
        df['week_of_month'] = (pd.to_datetime(df['date']).dt.day - 1) // 7 + 1
        by_week = df.groupby('week_of_month')['amount'].sum()
        result['by_week'] = by_week.to_dict()

        return result

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa tendências de gastos"""
        df['date'] = pd.to_datetime(df['date'])
        daily = df.groupby('date')['amount'].sum().sort_index()

        if len(daily) < 7:
            return {'message': 'Dados insuficientes para análise de tendência'}

        # Média móvel de 7 dias
        rolling_mean = daily.rolling(window=7).mean()

        # Tendência geral (regressão linear simples)
        x = np.arange(len(daily))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily.values)

        trend_direction = 'crescente' if slope > 0 else 'decrescente'
        trend_strength = abs(r_value)

        return {
            'direction': trend_direction,
            'slope': round(slope, 2),
            'r_squared': round(r_value ** 2, 3),
            'strength': 'forte' if trend_strength > 0.7 else 'moderada' if trend_strength > 0.4 else 'fraca',
            'daily_average': round(daily.mean(), 2),
            'projection_30_days': round(daily.mean() * 30, 2)
        }

    def predict_spending(
        self,
        transactions: List[Dict[str, Any]],
        days_ahead: int = 30
    ) -> Dict[str, Any]:
        """
        Prevê gastos futuros usando séries temporais.

        Args:
            transactions: Histórico de transações
            days_ahead: Dias para prever

        Returns:
            Previsão com intervalos de confiança
        """
        if not STATSMODELS_AVAILABLE:
            return self._predict_simple(transactions, days_ahead)

        df = pd.DataFrame(transactions)

        if 'date' not in df.columns:
            return {'error': 'Dados sem coluna de data'}

        df['date'] = pd.to_datetime(df['date'])
        daily = df.groupby('date')['amount'].sum()

        # Preencher dias sem transações com 0
        date_range = pd.date_range(daily.index.min(), daily.index.max())
        daily = daily.reindex(date_range, fill_value=0)

        if len(daily) < 14:
            return self._predict_simple(transactions, days_ahead)

        try:
            # Modelo Holt-Winters (suavização exponencial com sazonalidade)
            model = ExponentialSmoothing(
                daily.values,
                trend='add',
                seasonal='add' if len(daily) >= 14 else None,
                seasonal_periods=7 if len(daily) >= 14 else None
            )
            fitted = model.fit()

            # Previsão
            forecast = fitted.forecast(days_ahead)

            # Intervalo de confiança (aproximado)
            residuals = daily.values - fitted.fittedvalues
            std_residual = np.std(residuals)

            lower_bound = forecast - 1.96 * std_residual
            upper_bound = forecast + 1.96 * std_residual

            # Datas futuras
            last_date = daily.index[-1]
            future_dates = pd.date_range(last_date + timedelta(days=1), periods=days_ahead)

            predictions = []
            for i, date in enumerate(future_dates):
                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted': max(0, round(forecast[i], 2)),
                    'lower': max(0, round(lower_bound[i], 2)),
                    'upper': max(0, round(upper_bound[i], 2))
                })

            return {
                'method': 'Holt-Winters',
                'predictions': predictions,
                'total_predicted': round(sum(max(0, f) for f in forecast), 2),
                'average_daily': round(np.mean(forecast), 2),
                'confidence_level': 0.95
            }

        except Exception as e:
            logger.warning(f"Erro na previsão avançada: {e}")
            return self._predict_simple(transactions, days_ahead)

    def _predict_simple(
        self,
        transactions: List[Dict[str, Any]],
        days_ahead: int
    ) -> Dict[str, Any]:
        """Previsão simples baseada em médias"""
        df = pd.DataFrame(transactions)

        if 'amount' not in df.columns:
            return {'error': 'Dados sem coluna de valor'}

        daily_avg = df['amount'].sum() / max(len(set(df.get('date', []))), 1)
        std = df['amount'].std()

        total_predicted = daily_avg * days_ahead

        return {
            'method': 'Média histórica',
            'total_predicted': round(total_predicted, 2),
            'average_daily': round(daily_avg, 2),
            'range': {
                'lower': round(total_predicted - 1.96 * std * np.sqrt(days_ahead), 2),
                'upper': round(total_predicted + 1.96 * std * np.sqrt(days_ahead), 2)
            },
            'confidence_level': 0.95
        }

    def identify_behavioral_profile(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Identifica o perfil comportamental do usuário.

        Args:
            transactions: Lista de transações

        Returns:
            Perfil identificado com características
        """
        if not transactions:
            return {'profile': 'indefinido', 'confidence': 0}

        df = pd.DataFrame(transactions)
        scores = defaultdict(float)

        # Calcular indicadores

        # 1. Indicador noturno
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            night_ratio = len(df[(df['hour'] >= 0) & (df['hour'] <= 6)]) / len(df)
            if night_ratio > 0.15:
                scores['notívago'] += night_ratio * 100

        # 2. Indicador de impulso (frequência alta)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            unique_days = df['date'].nunique()
            transactions_per_day = len(df) / max(unique_days, 1)
            if transactions_per_day > 3:
                scores['impulsivo'] += min(transactions_per_day * 10, 50)

        # 3. Indicador social (categorias de lazer)
        if 'category' in df.columns:
            social_cats = ['lazer', 'delivery', 'restaurante', 'bar']
            social_ratio = len(df[df['category'].str.lower().isin(social_cats)]) / len(df)
            if social_ratio > 0.3:
                scores['social'] += social_ratio * 100

        # 4. Indicador sazonal (alta variação)
        if len(df) > 7:
            cv = df['amount'].std() / df['amount'].mean()  # Coeficiente de variação
            if cv > 1.5:
                scores['sazonal'] += cv * 20

        # 5. Indicador controlado (baixa variação, regularidade)
        if len(df) > 7:
            cv = df['amount'].std() / df['amount'].mean()
            if cv < 0.5:
                scores['controlado'] += (1 - cv) * 100

        # Determinar perfil dominante
        if not scores:
            return {
                'profile': 'controlado',
                'confidence': 50,
                'details': self.PROFILE_DEFINITIONS['controlado']
            }

        dominant_profile = max(scores, key=scores.get)
        confidence = min(scores[dominant_profile], 100)

        # Perfis secundários
        secondary_profiles = [
            {'profile': p, 'score': s}
            for p, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[1:3]
            if s > 20
        ]

        return {
            'profile': dominant_profile,
            'confidence': round(confidence, 1),
            'details': self.PROFILE_DEFINITIONS.get(dominant_profile, {}),
            'secondary_profiles': secondary_profiles,
            'all_scores': dict(scores)
        }

    def cluster_spending_patterns(
        self,
        transactions: List[Dict[str, Any]],
        n_clusters: int = 4
    ) -> Dict[str, Any]:
        """
        Agrupa transações em clusters de padrões similares.

        Args:
            transactions: Lista de transações
            n_clusters: Número de clusters

        Returns:
            Clusters identificados com características
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn não disponível'}

        if len(transactions) < n_clusters * 3:
            return {'error': 'Transações insuficientes para clustering'}

        df = pd.DataFrame(transactions)

        # Preparar features
        features = pd.DataFrame()
        features['amount'] = df['amount']

        if 'timestamp' in df.columns:
            features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        else:
            features['hour'] = 12

        if 'date' in df.columns:
            features['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        else:
            features['day_of_week'] = 0

        # Normalizar
        features_scaled = self.scaler.fit_transform(features)

        # K-Means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(features_scaled)

        df['cluster'] = clusters

        # Analisar cada cluster
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_data = df[df['cluster'] == i]

            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(df) * 100, 1),
                'avg_amount': round(cluster_data['amount'].mean(), 2),
                'total_amount': round(cluster_data['amount'].sum(), 2),
                'characteristics': self._describe_cluster(cluster_data, features[df['cluster'] == i])
            }

        return {
            'n_clusters': n_clusters,
            'clusters': cluster_analysis,
            'inertia': round(self.kmeans.inertia_, 2)
        }

    def _describe_cluster(self, df: pd.DataFrame, features: pd.DataFrame) -> List[str]:
        """Descreve características de um cluster"""
        characteristics = []

        avg_amount = df['amount'].mean()
        if avg_amount > df['amount'].quantile(0.75):
            characteristics.append('Valores altos')
        elif avg_amount < df['amount'].quantile(0.25):
            characteristics.append('Valores baixos')

        if 'hour' in features.columns:
            avg_hour = features['hour'].mean()
            if 0 <= avg_hour <= 6:
                characteristics.append('Predominantemente noturno')
            elif 6 < avg_hour <= 12:
                characteristics.append('Predominantemente matutino')
            elif 12 < avg_hour <= 18:
                characteristics.append('Predominantemente vespertino')

        if 'category' in df.columns:
            top_cat = df['category'].mode()
            if len(top_cat) > 0:
                characteristics.append(f'Categoria principal: {top_cat.iloc[0]}')

        return characteristics if characteristics else ['Padrão misto']

    def generate_savings_suggestions(
        self,
        transactions: List[Dict[str, Any]],
        target_reduction: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Gera sugestões personalizadas de economia.

        Args:
            transactions: Lista de transações
            target_reduction: Meta de redução (0.1 = 10%)

        Returns:
            Lista de sugestões com potencial de economia
        """
        if not transactions:
            return []

        df = pd.DataFrame(transactions)
        total_spent = df['amount'].sum()
        target_savings = total_spent * target_reduction

        suggestions = []

        # 1. Análise por categoria
        if 'category' in df.columns:
            category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)

            # Categorias com maior potencial de redução
            reducible_categories = ['delivery', 'lazer', 'assinaturas', 'jogos', 'compras']

            for cat in reducible_categories:
                if cat in category_totals.index:
                    cat_total = category_totals[cat]
                    potential = cat_total * 0.3  # Potencial de 30% de redução

                    suggestions.append({
                        'type': 'category_reduction',
                        'category': cat,
                        'current_spending': round(cat_total, 2),
                        'potential_savings': round(potential, 2),
                        'suggestion': f'Reduza gastos com {cat} em 30%',
                        'action': self._get_category_action(cat),
                        'priority': 'alta' if potential > target_savings * 0.2 else 'média'
                    })

        # 2. Análise de gastos recorrentes
        if 'merchant' in df.columns or 'description' in df.columns:
            merchant_col = 'merchant' if 'merchant' in df.columns else 'description'
            recurring = df.groupby(merchant_col).agg({
                'amount': ['count', 'sum', 'mean']
            })
            recurring.columns = ['count', 'total', 'avg']
            recurring = recurring[recurring['count'] >= 3].sort_values('total', ascending=False)

            for merchant, row in recurring.head(5).iterrows():
                suggestions.append({
                    'type': 'recurring_expense',
                    'merchant': merchant,
                    'frequency': int(row['count']),
                    'total_spent': round(row['total'], 2),
                    'avg_transaction': round(row['avg'], 2),
                    'suggestion': f'Você tem gasto recorrente em {merchant}',
                    'potential_savings': round(row['total'] * 0.2, 2),
                    'priority': 'média'
                })

        # 3. Análise de horário
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            night_spending = df[(df['hour'] >= 0) & (df['hour'] <= 6)]['amount'].sum()

            if night_spending > total_spent * 0.1:
                suggestions.append({
                    'type': 'behavioral',
                    'category': 'gastos_noturnos',
                    'current_spending': round(night_spending, 2),
                    'potential_savings': round(night_spending * 0.5, 2),
                    'suggestion': 'Você gasta significativamente durante a madrugada',
                    'action': 'Ative o modo de proteção noturna',
                    'priority': 'alta'
                })

        # Ordenar por potencial de economia
        suggestions.sort(key=lambda x: x.get('potential_savings', 0), reverse=True)

        # Adicionar resumo
        total_potential = sum(s.get('potential_savings', 0) for s in suggestions)

        return {
            'suggestions': suggestions[:10],  # Top 10
            'total_potential_savings': round(total_potential, 2),
            'target_savings': round(target_savings, 2),
            'achievable': total_potential >= target_savings
        }

    def _get_category_action(self, category: str) -> str:
        """Retorna ação específica para categoria"""
        actions = {
            'delivery': 'Cozinhe mais em casa. Prepare marmitas no fim de semana.',
            'lazer': 'Busque alternativas gratuitas: parques, eventos públicos.',
            'assinaturas': 'Revise todas as assinaturas. Cancele as não utilizadas.',
            'jogos': 'Defina um orçamento mensal fixo para jogos.',
            'compras': 'Use a regra das 48h antes de comprar algo não essencial.',
            'transporte': 'Considere carona, transporte público ou bicicleta.',
            'alimentação': 'Faça lista de compras e evite ir ao mercado com fome.'
        }
        return actions.get(category, 'Analise se este gasto é realmente necessário.')


# === Funções de conveniência ===

_analyzer = None


def get_analyzer() -> GastosAnalyzer:
    """Retorna instância global do analisador"""
    global _analyzer
    if _analyzer is None:
        _analyzer = GastosAnalyzer()
    return _analyzer


def analyze_spending(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analisa padrões de gastos"""
    return get_analyzer().analyze_spending_patterns(transactions)


def predict_spending(
    transactions: List[Dict[str, Any]],
    days_ahead: int = 30
) -> Dict[str, Any]:
    """Prevê gastos futuros"""
    return get_analyzer().predict_spending(transactions, days_ahead)


def get_user_profile(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identifica perfil comportamental"""
    return get_analyzer().identify_behavioral_profile(transactions)


def get_savings_suggestions(
    transactions: List[Dict[str, Any]],
    target_reduction: float = 0.1
) -> Dict[str, Any]:
    """Gera sugestões de economia"""
    return get_analyzer().generate_savings_suggestions(transactions, target_reduction)
