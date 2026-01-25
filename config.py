"""
Configurações globais do Dashboard Financeiro
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# === Diretórios ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Criar diretórios se não existirem
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# === Banco de Dados (PostgreSQL) ===
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', '')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'dashboard_financeiro')}"
)

# === Mistral AI ===
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = "pixtral-large-latest"  # Modelo multimodal para OCR

# === Horário de Risco (Proteção Noturna) ===
NIGHT_START = os.getenv("NIGHT_START", "00:00")
NIGHT_END = os.getenv("NIGHT_END", "06:00")

# === Limiares de Impulso ===
IMPULSE_AMOUNT_THRESHOLD = float(os.getenv("IMPULSE_THRESHOLD", "100.0"))  # BRL
IMPULSE_RISK_SCORE_THRESHOLD = 70  # Percentual de risco
DELAY_MINUTES_HIGH_RISK = 5  # Minutos de delay para compras de alto risco

# === Categorias Padrão ===
CATEGORIES = [
    "alimentação",
    "transporte",
    "moradia",
    "saúde",
    "lazer",
    "educação",
    "delivery",
    "jogos",
    "assinaturas",
    "compras",
    "transferência",
    "salário",
    "investimentos",
    "outros"
]

# === Tipos de Transação ===
TRANSACTION_TYPES = ["credito", "debito"]

# === Perfis Comportamentais ===
BEHAVIORAL_PROFILES = {
    "notívago": {
        "description": "Gasta mais durante a madrugada",
        "peak_hours": (0, 6),
        "risk_multiplier": 1.5
    },
    "ansioso": {
        "description": "Alta frequência de pequenas compras",
        "frequency_threshold": 5,
        "risk_multiplier": 1.3
    },
    "social": {
        "description": "Gastos concentrados em lazer e delivery",
        "category_focus": ["lazer", "delivery"],
        "risk_multiplier": 1.2
    },
    "controlado": {
        "description": "Padrão de gastos estável",
        "risk_multiplier": 0.8
    }
}

# === Configurações de Aplicação ===
APP_ENV = os.getenv("APP_ENV", "development")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "app.log"

# === Limites de Upload ===
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg", "pdf", "ofx", "csv"]

# === Configurações de Autenticação ===
COOKIE_NAME = "dashboard_financeiro_auth"
COOKIE_EXPIRY_DAYS = 30
