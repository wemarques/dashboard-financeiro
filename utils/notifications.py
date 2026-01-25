"""
Sistema de Notificações e Alertas
Gerencia alertas in-app, email e log de notificações
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum
import json

try:
    from config import DATA_DIR
    from utils.logger import get_logger
except ImportError:
    from pathlib import Path
    DATA_DIR = Path("data")
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class NotificationType(Enum):
    """Tipos de notificação"""
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    SUCCESS = "success"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Canais de notificação"""
    IN_APP = "in_app"       # Notificação no app
    EMAIL = "email"         # Email
    CONSOLE = "console"     # Log no console
    FILE = "file"           # Arquivo de log


class Notification:
    """Representa uma notificação"""

    def __init__(
        self,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        category: str = "general",
        data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        self.id = f"notif_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.title = title
        self.message = message
        self.type = notification_type
        self.category = category
        self.data = data or {}
        self.user_id = user_id
        self.created_at = datetime.now()
        self.read = False
        self.read_at = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'type': self.type.value,
            'category': self.category,
            'data': self.data,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'read': self.read,
            'read_at': self.read_at.isoformat() if self.read_at else None
        }

    def mark_as_read(self):
        self.read = True
        self.read_at = datetime.now()


class NotificationManager:
    """
    Gerenciador central de notificações.

    Responsável por:
    - Criar e armazenar notificações
    - Enviar por diferentes canais
    - Gerenciar histórico
    """

    # Templates de notificação
    TEMPLATES = {
        'night_alert': {
            'title': '🌙 Alerta Noturno',
            'message': 'Você está fazendo uma compra de R$ {amount:.2f} às {hour}h. Compras neste horário tendem a ser por impulso.'
        },
        'anomaly_detected': {
            'title': '⚠️ Gasto Atípico',
            'message': 'Detectamos um gasto incomum: R$ {amount:.2f} em {category}. Score de anomalia: {score}/100'
        },
        'budget_warning': {
            'title': '💰 Alerta de Orçamento',
            'message': 'Você já gastou {percent:.0f}% do seu orçamento de {category}.'
        },
        'goal_progress': {
            'title': '🎯 Progresso da Meta',
            'message': 'Você atingiu {percent:.0f}% da meta "{goal_name}"! Faltam R$ {remaining:.2f}'
        },
        'goal_achieved': {
            'title': '🎉 Meta Alcançada!',
            'message': 'Parabéns! Você alcançou a meta "{goal_name}"!'
        },
        'weekly_summary': {
            'title': '📊 Resumo Semanal',
            'message': 'Esta semana você gastou R$ {total:.2f}. {comparison}'
        },
        'impulse_blocked': {
            'title': '🛑 Compra Bloqueada',
            'message': 'Uma compra de R$ {amount:.2f} foi bloqueada por medida de proteção. Motivo: {reason}'
        },
        'savings_tip': {
            'title': '💡 Dica de Economia',
            'message': '{tip}'
        }
    }

    def __init__(self, storage_path: Optional[str] = None):
        """
        Inicializa o gerenciador de notificações.

        Args:
            storage_path: Caminho para armazenar notificações
        """
        self.storage_path = storage_path or str(DATA_DIR / "notifications.json")
        self.notifications: List[Notification] = []
        self.email_config = {
            'enabled': False,
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'username': os.getenv('EMAIL_USERNAME', ''),
            'password': os.getenv('EMAIL_PASSWORD', ''),
            'from_address': os.getenv('EMAIL_FROM', '')
        }

        # Carregar notificações existentes
        self._load_notifications()

        logger.info("NotificationManager inicializado")

    def _load_notifications(self):
        """Carrega notificações do arquivo"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    # Converter para objetos Notification (simplificado)
                    self.notifications = data.get('notifications', [])
        except Exception as e:
            logger.warning(f"Erro ao carregar notificações: {e}")
            self.notifications = []

    def _save_notifications(self):
        """Salva notificações no arquivo"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'notifications': [
                        n.to_dict() if isinstance(n, Notification) else n
                        for n in self.notifications[-1000:]  # Manter últimas 1000
                    ],
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Erro ao salvar notificações: {e}")

    def create_notification(
        self,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        category: str = "general",
        data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Notification:
        """
        Cria e envia uma notificação.

        Args:
            title: Título da notificação
            message: Mensagem
            notification_type: Tipo (info, warning, alert, etc)
            category: Categoria (night, anomaly, budget, etc)
            data: Dados adicionais
            user_id: ID do usuário
            channels: Canais para enviar

        Returns:
            Notificação criada
        """
        notification = Notification(
            title=title,
            message=message,
            notification_type=notification_type,
            category=category,
            data=data,
            user_id=user_id
        )

        # Armazenar
        self.notifications.append(notification)
        self._save_notifications()

        # Enviar pelos canais especificados
        channels = channels or [NotificationChannel.IN_APP, NotificationChannel.CONSOLE]

        for channel in channels:
            self._send_to_channel(notification, channel)

        return notification

    def create_from_template(
        self,
        template_name: str,
        template_data: Dict[str, Any],
        notification_type: Optional[NotificationType] = None,
        user_id: Optional[str] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Optional[Notification]:
        """
        Cria notificação a partir de template.

        Args:
            template_name: Nome do template
            template_data: Dados para preencher o template
            notification_type: Tipo da notificação
            user_id: ID do usuário
            channels: Canais de envio

        Returns:
            Notificação criada ou None se template não existir
        """
        if template_name not in self.TEMPLATES:
            logger.warning(f"Template não encontrado: {template_name}")
            return None

        template = self.TEMPLATES[template_name]

        try:
            title = template['title'].format(**template_data)
            message = template['message'].format(**template_data)
        except KeyError as e:
            logger.error(f"Dados faltando no template: {e}")
            return None

        # Determinar tipo baseado no template se não especificado
        if notification_type is None:
            type_mapping = {
                'night_alert': NotificationType.ALERT,
                'anomaly_detected': NotificationType.WARNING,
                'budget_warning': NotificationType.WARNING,
                'goal_progress': NotificationType.INFO,
                'goal_achieved': NotificationType.SUCCESS,
                'weekly_summary': NotificationType.INFO,
                'impulse_blocked': NotificationType.CRITICAL,
                'savings_tip': NotificationType.INFO
            }
            notification_type = type_mapping.get(template_name, NotificationType.INFO)

        return self.create_notification(
            title=title,
            message=message,
            notification_type=notification_type,
            category=template_name.split('_')[0],
            data=template_data,
            user_id=user_id,
            channels=channels
        )

    def _send_to_channel(self, notification: Notification, channel: NotificationChannel):
        """Envia notificação para um canal específico"""
        if channel == NotificationChannel.CONSOLE:
            self._send_to_console(notification)
        elif channel == NotificationChannel.EMAIL:
            self._send_to_email(notification)
        elif channel == NotificationChannel.FILE:
            self._send_to_file(notification)
        # IN_APP já está armazenado

    def _send_to_console(self, notification: Notification):
        """Envia para console/log"""
        level_map = {
            NotificationType.INFO: logger.info,
            NotificationType.WARNING: logger.warning,
            NotificationType.ALERT: logger.warning,
            NotificationType.SUCCESS: logger.info,
            NotificationType.CRITICAL: logger.error
        }

        log_func = level_map.get(notification.type, logger.info)
        log_func(f"[{notification.title}] {notification.message}")

    def _send_to_email(self, notification: Notification):
        """Envia por email"""
        if not self.email_config['enabled']:
            logger.debug("Email desabilitado, notificação não enviada")
            return

        if not notification.user_id:
            logger.warning("Notificação sem user_id, email não enviado")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_address']
            msg['To'] = notification.data.get('email', '')
            msg['Subject'] = notification.title

            body = f"""
            {notification.message}

            ---
            Dashboard Financeiro
            Enviado em: {notification.created_at.strftime('%d/%m/%Y %H:%M')}
            """

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['username'],
                    self.email_config['password']
                )
                server.send_message(msg)

            logger.info(f"Email enviado para {msg['To']}")

        except Exception as e:
            logger.error(f"Erro ao enviar email: {e}")

    def _send_to_file(self, notification: Notification):
        """Salva em arquivo de log dedicado"""
        log_file = DATA_DIR / "notifications.log"
        try:
            with open(log_file, 'a') as f:
                f.write(f"[{notification.created_at.isoformat()}] ")
                f.write(f"[{notification.type.value.upper()}] ")
                f.write(f"{notification.title}: {notification.message}\n")
        except Exception as e:
            logger.error(f"Erro ao escrever log: {e}")

    def get_notifications(
        self,
        user_id: Optional[str] = None,
        unread_only: bool = False,
        category: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Busca notificações com filtros.

        Args:
            user_id: Filtrar por usuário
            unread_only: Apenas não lidas
            category: Filtrar por categoria
            limit: Limite de resultados

        Returns:
            Lista de notificações
        """
        result = []

        for n in reversed(self.notifications):  # Mais recentes primeiro
            if isinstance(n, dict):
                notif = n
            else:
                notif = n.to_dict()

            if user_id and notif.get('user_id') != user_id:
                continue
            if unread_only and notif.get('read'):
                continue
            if category and notif.get('category') != category:
                continue

            result.append(notif)

            if len(result) >= limit:
                break

        return result

    def mark_as_read(self, notification_id: str) -> bool:
        """Marca notificação como lida"""
        for n in self.notifications:
            if isinstance(n, Notification):
                if n.id == notification_id:
                    n.mark_as_read()
                    self._save_notifications()
                    return True
            elif isinstance(n, dict):
                if n.get('id') == notification_id:
                    n['read'] = True
                    n['read_at'] = datetime.now().isoformat()
                    self._save_notifications()
                    return True
        return False

    def mark_all_as_read(self, user_id: Optional[str] = None):
        """Marca todas notificações como lidas"""
        for n in self.notifications:
            if isinstance(n, Notification):
                if user_id is None or n.user_id == user_id:
                    n.mark_as_read()
            elif isinstance(n, dict):
                if user_id is None or n.get('user_id') == user_id:
                    n['read'] = True
                    n['read_at'] = datetime.now().isoformat()

        self._save_notifications()

    def get_unread_count(self, user_id: Optional[str] = None) -> int:
        """Retorna contagem de não lidas"""
        count = 0
        for n in self.notifications:
            if isinstance(n, Notification):
                if not n.read and (user_id is None or n.user_id == user_id):
                    count += 1
            elif isinstance(n, dict):
                if not n.get('read') and (user_id is None or n.get('user_id') == user_id):
                    count += 1
        return count


# === Funções de conveniência ===

_manager = None


def get_notification_manager() -> NotificationManager:
    """Retorna instância global do gerenciador"""
    global _manager
    if _manager is None:
        _manager = NotificationManager()
    return _manager


def send_notification(
    title: str,
    message: str,
    notification_type: str = "info",
    category: str = "general",
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Envia notificação (função de conveniência)"""
    manager = get_notification_manager()

    type_map = {
        'info': NotificationType.INFO,
        'warning': NotificationType.WARNING,
        'alert': NotificationType.ALERT,
        'success': NotificationType.SUCCESS,
        'critical': NotificationType.CRITICAL
    }

    notif = manager.create_notification(
        title=title,
        message=message,
        notification_type=type_map.get(notification_type, NotificationType.INFO),
        category=category,
        user_id=user_id
    )

    return notif.to_dict()


def send_night_alert(amount: float, user_id: Optional[str] = None):
    """Envia alerta noturno"""
    manager = get_notification_manager()
    return manager.create_from_template(
        'night_alert',
        {'amount': amount, 'hour': datetime.now().hour},
        user_id=user_id
    )


def send_anomaly_alert(
    amount: float,
    category: str,
    score: int,
    user_id: Optional[str] = None
):
    """Envia alerta de anomalia"""
    manager = get_notification_manager()
    return manager.create_from_template(
        'anomaly_detected',
        {'amount': amount, 'category': category, 'score': score},
        user_id=user_id
    )


def send_goal_notification(
    goal_name: str,
    percent: float,
    remaining: float,
    achieved: bool = False,
    user_id: Optional[str] = None
):
    """Envia notificação de meta"""
    manager = get_notification_manager()
    template = 'goal_achieved' if achieved else 'goal_progress'
    return manager.create_from_template(
        template,
        {'goal_name': goal_name, 'percent': percent, 'remaining': remaining},
        user_id=user_id
    )


def get_user_notifications(
    user_id: str,
    unread_only: bool = False,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Busca notificações do usuário"""
    manager = get_notification_manager()
    return manager.get_notifications(user_id=user_id, unread_only=unread_only, limit=limit)
