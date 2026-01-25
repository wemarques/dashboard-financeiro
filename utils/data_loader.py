"""
Gerenciador de banco de dados PostgreSQL para o Dashboard Financeiro
"""
import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Date, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func

try:
    from config import DATABASE_URL
    from utils.logger import get_logger
except ImportError:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/dashboard_financeiro")
    from logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


# === Modelos do Banco de Dados ===

class Transaction(Base):
    """Modelo para transações financeiras"""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    amount = Column(Float, nullable=False)
    merchant = Column(String(255))
    description = Column(Text)
    category = Column(String(100))
    transaction_type = Column(String(20), default="debito")  # credito ou debito
    timestamp = Column(DateTime, default=func.now())
    is_night = Column(Boolean, default=False)
    risk_score = Column(Integer, default=0)
    source = Column(String(50), default="manual")  # manual, ocr, import
    installment = Column(String(20))  # Ex: "1/3"
    notes = Column(Text)

    # Relacionamentos
    alerts = relationship("Alert", back_populates="transaction")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "date": self.date.isoformat() if self.date else None,
            "amount": self.amount,
            "merchant": self.merchant,
            "description": self.description,
            "category": self.category,
            "transaction_type": self.transaction_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_night": self.is_night,
            "risk_score": self.risk_score,
            "source": self.source,
            "installment": self.installment,
            "notes": self.notes
        }


class Goal(Base):
    """Modelo para metas financeiras"""
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    target_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0)
    deadline = Column(Date)
    status = Column(String(50), default="active")  # active, completed, cancelled
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "target_amount": self.target_amount,
            "current_amount": self.current_amount,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status,
            "progress": (self.current_amount / self.target_amount * 100) if self.target_amount > 0 else 0
        }


class Alert(Base):
    """Modelo para alertas do sistema"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=True)
    alert_type = Column(String(50), nullable=False)  # night, anomaly, threshold, impulse
    message = Column(Text, nullable=False)
    risk_score = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)

    # Relacionamentos
    transaction = relationship("Transaction", back_populates="alerts")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "alert_type": self.alert_type,
            "message": self.message,
            "risk_score": self.risk_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "acknowledged": self.acknowledged
        }


class UserProfile(Base):
    """Modelo para perfil do usuário"""
    __tablename__ = "user_profile"

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    behavioral_type = Column(String(50))
    night_protection_enabled = Column(Boolean, default=True)
    daily_limit = Column(Float)
    monthly_limit = Column(Float)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "behavioral_type": self.behavioral_type,
            "night_protection_enabled": self.night_protection_enabled,
            "daily_limit": self.daily_limit,
            "monthly_limit": self.monthly_limit
        }


# === Gerenciador de Banco de Dados ===

class DatabaseManager:
    """Gerenciador de operações do banco de dados"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"DatabaseManager inicializado")

    def create_tables(self) -> None:
        """Cria todas as tabelas no banco de dados"""
        Base.metadata.create_all(self.engine)
        logger.info("Tabelas criadas com sucesso")

    def drop_tables(self) -> None:
        """Remove todas as tabelas (use com cuidado!)"""
        Base.metadata.drop_all(self.engine)
        logger.warning("Todas as tabelas foram removidas")

    @contextmanager
    def get_session(self) -> Session:
        """Context manager para sessões do banco"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erro na sessão do banco: {e}")
            raise
        finally:
            session.close()

    # === Operações de Transações ===

    def add_transaction(self, data: Dict[str, Any]) -> Transaction:
        """Adiciona uma nova transação"""
        with self.get_session() as session:
            transaction = Transaction(
                date=data.get("date", date.today()),
                amount=data["amount"],
                merchant=data.get("merchant"),
                description=data.get("description"),
                category=data.get("category"),
                transaction_type=data.get("transaction_type", "debito"),
                is_night=data.get("is_night", False),
                risk_score=data.get("risk_score", 0),
                source=data.get("source", "manual"),
                installment=data.get("installment"),
                notes=data.get("notes")
            )
            session.add(transaction)
            session.flush()
            logger.info(f"Transação adicionada: ID={transaction.id}, valor={transaction.amount}")
            return transaction

    def add_transactions_batch(self, transactions: List[Dict[str, Any]]) -> int:
        """Adiciona múltiplas transações em lote"""
        with self.get_session() as session:
            count = 0
            for data in transactions:
                transaction = Transaction(
                    date=data.get("date", date.today()),
                    amount=data["amount"],
                    merchant=data.get("merchant"),
                    description=data.get("description"),
                    category=data.get("category"),
                    transaction_type=data.get("transaction_type", "debito"),
                    is_night=data.get("is_night", False),
                    risk_score=data.get("risk_score", 0),
                    source=data.get("source", "import"),
                    installment=data.get("installment"),
                    notes=data.get("notes")
                )
                session.add(transaction)
                count += 1
            logger.info(f"{count} transações adicionadas em lote")
            return count

    def get_transactions(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        category: Optional[str] = None,
        transaction_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Busca transações com filtros opcionais"""
        with self.get_session() as session:
            query = session.query(Transaction)

            if start_date:
                query = query.filter(Transaction.date >= start_date)
            if end_date:
                query = query.filter(Transaction.date <= end_date)
            if category:
                query = query.filter(Transaction.category == category)
            if transaction_type:
                query = query.filter(Transaction.transaction_type == transaction_type)

            query = query.order_by(Transaction.date.desc()).limit(limit)

            return [t.to_dict() for t in query.all()]

    def get_transactions_summary(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Retorna resumo das transações"""
        with self.get_session() as session:
            query = session.query(Transaction)

            if start_date:
                query = query.filter(Transaction.date >= start_date)
            if end_date:
                query = query.filter(Transaction.date <= end_date)

            transactions = query.all()

            total_credito = sum(t.amount for t in transactions if t.transaction_type == "credito")
            total_debito = sum(t.amount for t in transactions if t.transaction_type == "debito")

            # Agrupar por categoria
            by_category = {}
            for t in transactions:
                cat = t.category or "outros"
                if cat not in by_category:
                    by_category[cat] = 0
                by_category[cat] += t.amount

            return {
                "total_credito": total_credito,
                "total_debito": total_debito,
                "saldo": total_credito - total_debito,
                "count": len(transactions),
                "by_category": by_category
            }

    # === Operações de Alertas ===

    def add_alert(self, data: Dict[str, Any]) -> Alert:
        """Adiciona um novo alerta"""
        with self.get_session() as session:
            alert = Alert(
                transaction_id=data.get("transaction_id"),
                alert_type=data["alert_type"],
                message=data["message"],
                risk_score=data.get("risk_score", 0)
            )
            session.add(alert)
            session.flush()
            logger.warning(f"Alerta criado: {alert.alert_type} - {alert.message}")
            return alert

    def get_unacknowledged_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Busca alertas não reconhecidos"""
        with self.get_session() as session:
            alerts = session.query(Alert)\
                .filter(Alert.acknowledged == False)\
                .order_by(Alert.created_at.desc())\
                .limit(limit)\
                .all()
            return [a.to_dict() for a in alerts]

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Marca um alerta como reconhecido"""
        with self.get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                return True
            return False

    # === Operações de Metas ===

    def add_goal(self, data: Dict[str, Any]) -> Goal:
        """Adiciona uma nova meta financeira"""
        with self.get_session() as session:
            goal = Goal(
                name=data["name"],
                target_amount=data["target_amount"],
                current_amount=data.get("current_amount", 0),
                deadline=data.get("deadline")
            )
            session.add(goal)
            session.flush()
            logger.info(f"Meta criada: {goal.name}")
            return goal

    def get_goals(self, status: str = "active") -> List[Dict[str, Any]]:
        """Busca metas por status"""
        with self.get_session() as session:
            goals = session.query(Goal)\
                .filter(Goal.status == status)\
                .order_by(Goal.deadline)\
                .all()
            return [g.to_dict() for g in goals]

    def update_goal_progress(self, goal_id: int, amount: float) -> bool:
        """Atualiza o progresso de uma meta"""
        with self.get_session() as session:
            goal = session.query(Goal).filter(Goal.id == goal_id).first()
            if goal:
                goal.current_amount = amount
                if goal.current_amount >= goal.target_amount:
                    goal.status = "completed"
                return True
            return False

    # === Operações de Perfil ===

    def get_or_create_profile(self, username: str) -> Dict[str, Any]:
        """Busca ou cria perfil do usuário"""
        with self.get_session() as session:
            profile = session.query(UserProfile)\
                .filter(UserProfile.username == username)\
                .first()

            if not profile:
                profile = UserProfile(username=username)
                session.add(profile)
                session.flush()
                logger.info(f"Perfil criado para: {username}")

            return profile.to_dict()

    def update_profile(self, username: str, data: Dict[str, Any]) -> bool:
        """Atualiza perfil do usuário"""
        with self.get_session() as session:
            profile = session.query(UserProfile)\
                .filter(UserProfile.username == username)\
                .first()

            if profile:
                for key, value in data.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                return True
            return False
