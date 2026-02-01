"""
Categorizador automático de transações usando NLP e ML
"""
import re
from typing import Optional, List, Dict, Tuple
import pickle
from pathlib import Path

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
except ImportError:
    TfidfVectorizer = None
    MultinomialNB = None
    Pipeline = None

try:
    from config import CATEGORIES, DATA_DIR
    from utils.logger import get_logger
except ImportError:
    CATEGORIES = ["alimentação", "transporte", "moradia", "saúde", "lazer", "outros"]
    DATA_DIR = Path("data")
    from logger import get_logger

logger = get_logger(__name__)


# === Regras baseadas em palavras-chave ===

KEYWORD_RULES = {
    "alimentação": [
        "supermercado", "mercado", "açougue", "padaria", "hortifruti",
        "carrefour", "extra", "pao de acucar", "assai", "atacadao",
        "dia", "minuto pao", "swift"
    ],
    "delivery": [
        "ifood", "rappi", "uber eats", "zé delivery", "aiqfome",
        "delivery", "entrega", "pizzaria", "hamburgueria", "sushi"
    ],
    "transporte": [
        "uber", "99", "cabify", "taxi", "combustivel", "gasolina",
        "etanol", "posto", "shell", "ipiranga", "br", "pedagio",
        "estacionamento", "parking", "bilhete unico", "metro", "onibus"
    ],
    "moradia": [
        "aluguel", "condominio", "iptu", "luz", "energia", "enel",
        "cemig", "agua", "sabesp", "gas", "comgas", "internet",
        "telefone", "vivo", "claro", "tim", "oi"
    ],
    "saúde": [
        "farmacia", "drogaria", "droga", "raia", "pacheco", "drogasil",
        "hospital", "clinica", "medico", "consulta", "exame", "laboratorio",
        "plano de saude", "unimed", "amil", "bradesco saude"
    ],
    "lazer": [
        "cinema", "teatro", "show", "ingresso", "netflix", "spotify",
        "amazon prime", "disney", "hbo", "globoplay", "youtube premium",
        "bar", "restaurante", "lanchonete", "cafe", "starbucks"
    ],
    "jogos": [
        "steam", "playstation", "xbox", "nintendo", "epic games",
        "riot", "blizzard", "ea sports", "ubisoft", "game", "jogo",
        "bet", "aposta", "loteria", "mega sena"
    ],
    "assinaturas": [
        "assinatura", "mensalidade", "anual", "recorrente",
        "academia", "smart fit", "bio ritmo", "gympass"
    ],
    "educação": [
        "escola", "faculdade", "universidade", "curso", "udemy",
        "coursera", "alura", "livro", "livraria", "saraiva", "cultura"
    ],
    "compras": [
        "amazon", "mercado livre", "magalu", "magazine luiza",
        "americanas", "shopee", "aliexpress", "shein", "renner",
        "c&a", "riachuelo", "zara", "shopping"
    ],
    "transferência": [
        "pix", "ted", "doc", "transferencia", "deposito"
    ],
    "salário": [
        "salario", "pagamento", "folha", "remuneracao", "pro-labore"
    ],
    "investimentos": [
        "investimento", "aplicacao", "cdb", "tesouro", "acao",
        "fundo", "corretora", "xp", "rico", "clear", "nuinvest"
    ]
}


class Categorizer:
    """Categorizador de transações financeiras"""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa o categorizador

        Args:
            model_path: Caminho opcional para carregar modelo treinado
        """
        self.model_path = model_path or DATA_DIR / "categorizer_model.pkl"
        self.pipeline = None
        self.is_trained = False

        # Tentar carregar modelo existente
        if self.model_path.exists():
            self._load_model()

    def _preprocess_text(self, text: str) -> str:
        """Pré-processa texto para classificação"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def categorize_by_rules(self, description: str, merchant: str = "") -> Tuple[str, float]:
        """
        Categoriza usando regras baseadas em palavras-chave

        Args:
            description: Descrição da transação
            merchant: Nome do estabelecimento

        Returns:
            Tupla (categoria, confiança)
        """
        text = self._preprocess_text(f"{description} {merchant}")

        matches = {}
        for category, keywords in KEYWORD_RULES.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > 0:
                matches[category] = count

        if matches:
            best_category = max(matches, key=matches.get)
            # Confiança baseada na quantidade de matches
            confidence = min(matches[best_category] / 3, 1.0)
            return best_category, confidence

        return "outros", 0.3

    def train(self, descriptions: List[str], categories: List[str]) -> float:
        """
        Treina o modelo de classificação

        Args:
            descriptions: Lista de descrições de transações
            categories: Lista de categorias correspondentes

        Returns:
            Acurácia do modelo
        """
        if not Pipeline:
            logger.error("scikit-learn não instalado")
            return 0.0

        # Pré-processar
        processed = [self._preprocess_text(d) for d in descriptions]

        # Criar pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])

        # Treinar
        self.pipeline.fit(processed, categories)
        self.is_trained = True

        # Calcular acurácia
        accuracy = self.pipeline.score(processed, categories)
        logger.info(f"Modelo treinado com acurácia: {accuracy:.2%}")

        # Salvar modelo
        self._save_model()

        return accuracy

    def categorize_ml(self, description: str, merchant: str = "") -> Tuple[str, float]:
        """
        Categoriza usando modelo ML treinado

        Args:
            description: Descrição da transação
            merchant: Nome do estabelecimento

        Returns:
            Tupla (categoria, confiança)
        """
        if not self.is_trained or not self.pipeline:
            return self.categorize_by_rules(description, merchant)

        text = self._preprocess_text(f"{description} {merchant}")

        # Predição
        category = self.pipeline.predict([text])[0]

        # Probabilidade/confiança
        proba = self.pipeline.predict_proba([text])[0]
        confidence = max(proba)

        return category, confidence

    def categorize(self, description: str, merchant: str = "") -> Dict[str, any]:
        """
        Categoriza transação usando combinação de regras e ML

        Args:
            description: Descrição da transação
            merchant: Nome do estabelecimento

        Returns:
            Dicionário com categoria e metadados
        """
        # Primeiro, tentar regras (mais confiável para padrões conhecidos)
        rule_cat, rule_conf = self.categorize_by_rules(description, merchant)

        # Se confiança das regras for alta, usar
        if rule_conf >= 0.7:
            return {
                "category": rule_cat,
                "confidence": rule_conf,
                "method": "rules"
            }

        # Tentar ML se disponível
        if self.is_trained:
            ml_cat, ml_conf = self.categorize_ml(description, merchant)

            # Usar ML se confiança for maior
            if ml_conf > rule_conf:
                return {
                    "category": ml_cat,
                    "confidence": ml_conf,
                    "method": "ml"
                }

        # Fallback para regras
        return {
            "category": rule_cat,
            "confidence": rule_conf,
            "method": "rules"
        }

    def _save_model(self) -> None:
        """Salva modelo treinado"""
        if self.pipeline:
            self.model_path.parent.mkdir(exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.pipeline, f)
            logger.info(f"Modelo salvo em: {self.model_path}")

    def _load_model(self) -> None:
        """Carrega modelo treinado"""
        try:
            with open(self.model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            self.is_trained = True
            logger.info(f"Modelo carregado de: {self.model_path}")
        except Exception as e:
            logger.warning(f"Não foi possível carregar modelo: {e}")


# === Função de conveniência ===

_categorizer = None


def categorize_transaction(description: str, merchant: str = "") -> Dict[str, any]:
    """
    Categoriza uma transação (função de conveniência)

    Args:
        description: Descrição da transação
        merchant: Nome do estabelecimento

    Returns:
        Dicionário com categoria e metadados
    """
    global _categorizer
    if _categorizer is None:
        _categorizer = Categorizer()
    return _categorizer.categorize(description, merchant)


def get_category_suggestions(partial_text: str, limit: int = 5) -> List[str]:
    """
    Retorna sugestões de categorias baseadas em texto parcial

    Args:
        partial_text: Texto parcial para buscar sugestões
        limit: Número máximo de sugestões

    Returns:
        Lista de categorias sugeridas
    """
    text = partial_text.lower()
    suggestions = []

    for category, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if text in kw or kw in text:
                if category not in suggestions:
                    suggestions.append(category)
                    break

    return suggestions[:limit] if suggestions else ["outros"]
