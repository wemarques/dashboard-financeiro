"""
Sistema de logging estruturado para o Dashboard Financeiro
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Importar configurações
try:
    from config import LOG_LEVEL, LOG_FILE, LOGS_DIR
except ImportError:
    LOG_LEVEL = "INFO"
    LOG_FILE = Path("logs/app.log")
    LOGS_DIR = Path("logs")


class ColoredFormatter(logging.Formatter):
    """Formatter com cores para output no console"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Cria e retorna um logger configurado

    Args:
        name: Nome do logger (geralmente __name__)
        log_file: Arquivo de log opcional (usa padrão se não especificado)

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)

    # Evitar duplicação de handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Criar diretório de logs se necessário
    LOGS_DIR.mkdir(exist_ok=True)

    # Handler para console (com cores)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Handler para arquivo
    file_path = log_file or LOG_FILE
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def log_transaction(logger: logging.Logger, action: str, data: dict) -> None:
    """
    Loga uma transação financeira de forma estruturada

    Args:
        logger: Logger a ser usado
        action: Ação realizada (create, update, delete, import)
        data: Dados da transação
    """
    logger.info(
        f"TRANSACTION | action={action} | "
        f"amount={data.get('amount', 'N/A')} | "
        f"category={data.get('category', 'N/A')} | "
        f"merchant={data.get('merchant', 'N/A')}"
    )


def log_ocr_result(logger: logging.Logger, success: bool, filename: str, items_count: int = 0) -> None:
    """
    Loga resultado de processamento OCR

    Args:
        logger: Logger a ser usado
        success: Se o OCR foi bem sucedido
        filename: Nome do arquivo processado
        items_count: Quantidade de itens extraídos
    """
    status = "SUCCESS" if success else "FAILED"
    logger.info(
        f"OCR | status={status} | file={filename} | items_extracted={items_count}"
    )


def log_alert(logger: logging.Logger, alert_type: str, message: str, risk_score: int = 0) -> None:
    """
    Loga alertas do sistema

    Args:
        logger: Logger a ser usado
        alert_type: Tipo do alerta (night, anomaly, threshold, impulse)
        message: Mensagem do alerta
        risk_score: Score de risco associado
    """
    logger.warning(
        f"ALERT | type={alert_type} | risk_score={risk_score} | message={message}"
    )
