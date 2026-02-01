"""
Dashboard Financeiro Pessoal
Sistema inteligente de gestão financeira com OCR, ML e análise comportamental
Fase 2: Inteligência Comportamental
"""
import os
import tempfile
from datetime import datetime, date, timedelta
import random

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página (deve ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="Dashboard Financeiro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === CSS RESPONSIVO ===
st.markdown("""
<style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .stColumns > div {
            flex: 100% !important;
            margin-bottom: 1rem;
        }
        .stMetric {
            padding: 0.5rem !important;
        }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        .stButton > button {
            width: 100% !important;
        }
        .upload-section {
            padding: 1rem !important;
        }
    }

    /* Tablet */
    @media (min-width: 769px) and (max-width: 1024px) {
        .stColumns > div {
            flex: 50% !important;
        }
    }

    /* Touch-friendly buttons */
    .stButton > button {
        min-height: 48px;
        min-width: 48px;
        font-size: 1rem;
    }

    /* Alertas customizados */
    .alert-night {
        background-color: #1a1a2e;
        border-left: 4px solid #e94560;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
    }

    /* Esconder menu do Streamlit em mobile */
    @media (max-width: 768px) {
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    }
</style>
""", unsafe_allow_html=True)

# === IMPORTS LOCAIS (após configuração da página) ===
try:
    from config import (
        MISTRAL_API_KEY, DATABASE_URL, CATEGORIES,
        NIGHT_START, NIGHT_END, IMPULSE_AMOUNT_THRESHOLD
    )
    from utils.logger import get_logger
    from utils.ocr_mistral import OCRProcessor
    from utils.notifications import get_notification_manager, get_user_notifications
    from ml.categorizer import categorize_transaction, Categorizer
    from ml.anomaly_detector import detect_anomalies, get_anomaly_report
    from ml.otimizador_gastos import (
        analyze_spending, predict_spending, get_user_profile, get_savings_suggestions
    )
    from behavioral.impulse_guard import ImpulseGuard, check_transaction_risk, is_night_mode
    from behavioral.intervention import generate_intervention, get_reflective_questions

    CONFIG_LOADED = True
    ML_LOADED = True
except ImportError as e:
    CONFIG_LOADED = False
    ML_LOADED = False
    st.warning(f"Alguns módulos não foram carregados: {e}")

# Logger
if CONFIG_LOADED:
    logger = get_logger(__name__)

# === AUTENTICAÇÃO ===
def check_authentication():
    """Sistema de autenticação simples"""

    # Verificar se já está autenticado
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 Login - Dashboard Financeiro")

    # Formulário de login
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar", use_container_width=True)

        if submitted:
            # Verificar credenciais (em produção, usar hash e banco de dados)
            # Para demo, aceitar qualquer usuário com senha "demo123"
            valid_users = {
                "admin": "admin123",
                "demo": "demo123",
                "usuario": "senha123"
            }

            if username in valid_users and valid_users[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos")

    st.info("💡 Para teste, use: usuário `demo` e senha `demo123`")
    return False


def logout():
    """Realiza logout do usuário"""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.rerun()


# === FUNÇÕES DE DADOS ===
@st.cache_data
def carregar_dados_csv():
    """Carrega dados do CSV legado"""
    try:
        df = pd.read_csv("dados.csv")
        return df
    except Exception as e:
        return pd.DataFrame()


def get_sample_data():
    """Retorna dados de exemplo para demonstração"""
    return {
        "receitas": 20300.02,
        "pagamentos": 14881.46,
        "poupanca": 5418.56,
        "saldo_inicial": 39416.49,
        "decendio": pd.DataFrame({
            "Período": ["1 a 10", "11 a 20", "21 a 31"],
            "Receitas": [7200.00, 100.00, 13000.02],
            "Despesas": [5418.49, 5222.07, 4240.90]
        }),
        "categorias": {
            "Empresa": 9982.89,
            "Pessoais": 4823.55,
            "Financeiras": 75.02
        },
        "entradas_diarias": [0, 0, 0, 2000, 0, 0, 0, 0, 5000, 200,
                            0, 0, 0, 100, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 8000, 0.01, 0.01, 5000],
        "saidas_diarias": [0, 100, 552.5, 100, 0, 0, 365.34, 2752.53, 225.12, 1323,
                          0, 0, 0, 1962.91, 0, 255.62, 2991.54, 12, 0, 0,
                          112.5, 508, 758.61, 50, 0, 0, 0, 500, 284.53, 506, 1521.26]
    }


def get_sample_transactions():
    """Gera transações de exemplo para demonstração do ML"""
    categories = ['alimentação', 'delivery', 'transporte', 'lazer', 'saúde', 'compras', 'assinaturas']
    merchants = {
        'alimentação': ['Supermercado Extra', 'Carrefour', 'Pão de Açúcar'],
        'delivery': ['iFood', 'Rappi', 'Uber Eats'],
        'transporte': ['Uber', '99', 'Posto Shell'],
        'lazer': ['Netflix', 'Spotify', 'Cinema'],
        'saúde': ['Farmácia', 'Drogasil', 'Consulta médica'],
        'compras': ['Amazon', 'Magazine Luiza', 'Mercado Livre'],
        'assinaturas': ['Netflix', 'Spotify', 'Amazon Prime']
    }

    transactions = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(50):
        cat = random.choice(categories)
        merchant = random.choice(merchants[cat])

        # Criar variação de valores por categoria
        if cat == 'delivery':
            amount = random.uniform(25, 120)
        elif cat == 'alimentação':
            amount = random.uniform(50, 400)
        elif cat == 'transporte':
            amount = random.uniform(15, 80)
        elif cat == 'lazer':
            amount = random.uniform(30, 150)
        elif cat == 'saúde':
            amount = random.uniform(20, 300)
        elif cat == 'compras':
            amount = random.uniform(50, 500)
        else:
            amount = random.uniform(15, 50)

        # Algumas transações noturnas
        hour = random.randint(8, 23)
        if random.random() < 0.15:  # 15% de chance de ser noturna
            hour = random.randint(0, 5)

        trans_date = base_date + timedelta(days=random.randint(0, 30))
        timestamp = trans_date.replace(hour=hour, minute=random.randint(0, 59))

        transactions.append({
            'date': trans_date.strftime('%Y-%m-%d'),
            'timestamp': timestamp.isoformat(),
            'amount': round(amount, 2),
            'merchant': merchant,
            'category': cat,
            'description': f'{merchant} - {cat}'
        })

    # Adicionar algumas anomalias
    transactions.append({
        'date': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
        'timestamp': (datetime.now() - timedelta(days=5)).replace(hour=2).isoformat(),
        'amount': 850.00,
        'merchant': 'Compra Online',
        'category': 'compras',
        'description': 'Compra por impulso às 2h'
    })

    transactions.append({
        'date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
        'timestamp': (datetime.now() - timedelta(days=3)).replace(hour=3).isoformat(),
        'amount': 450.00,
        'merchant': 'Bet365',
        'category': 'jogos',
        'description': 'Aposta noturna'
    })

    return transactions


def get_sample_goals():
    """Retorna metas de exemplo"""
    return [
        {
            'id': 1,
            'name': 'Reserva de Emergência',
            'target_amount': 15000.00,
            'current_amount': 8500.00,
            'deadline': '2024-12-31',
            'status': 'active'
        },
        {
            'id': 2,
            'name': 'Viagem de Férias',
            'target_amount': 5000.00,
            'current_amount': 2300.00,
            'deadline': '2024-06-30',
            'status': 'active'
        }
    ]


# === COMPONENTES DA UI ===
def render_header():
    """Renderiza cabeçalho do dashboard"""
    col1, col2, col3 = st.columns([6, 2, 2])

    with col1:
        st.title("💰 Dashboard Financeiro")
        st.caption(f"Bem-vindo, {st.session_state.get('username', 'Usuário')}")

    with col2:
        # Indicador de modo noturno
        if CONFIG_LOADED and is_night_mode():
            st.markdown("""
                <div style='background-color: #1a1a2e; color: #e94560; padding: 0.5rem;
                            border-radius: 0.5rem; text-align: center;'>
                    🌙 Modo Noturno Ativo
                </div>
            """, unsafe_allow_html=True)

    with col3:
        if st.button("🚪 Sair", use_container_width=True):
            logout()


def render_kpis(data):
    """Renderiza cards de KPIs"""
    st.subheader("📌 Resumo Financeiro")

    receitas = data["receitas"]
    pagamentos = data["pagamentos"]
    poupanca = data["poupanca"]
    perc_poupanca = (poupanca / receitas) * 100 if receitas > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Entradas",
            f"R$ {receitas:,.2f}",
            help="Total de receitas do período"
        )

    with col2:
        st.metric(
            "💸 Saídas",
            f"R$ {pagamentos:,.2f}",
            delta=f"-{(pagamentos/receitas)*100:.1f}% das entradas",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "✅ Saldo Líquido",
            f"R$ {poupanca:,.2f}",
            delta=f"+{perc_poupanca:.1f}%"
        )

    with col4:
        st.metric(
            "📈 Taxa de Poupança",
            f"{perc_poupanca:.1f}%",
            help="Percentual das receitas que foi poupado"
        )


def render_charts(data):
    """Renderiza gráficos principais"""

    # Gráfico de Decêndio
    st.subheader("📈 Receitas e Despesas por Decêndio")

    fig_bar = px.bar(
        data["decendio"],
        x="Período",
        y=["Receitas", "Despesas"],
        barmode="group",
        color_discrete_map={"Receitas": "#2E8B57", "Despesas": "#D32F2F"}
    )
    fig_bar.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Duas colunas para os próximos gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Composição das Despesas")
        fig_pie = px.pie(
            names=list(data["categorias"].keys()),
            values=list(data["categorias"].values()),
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("📋 Despesas por Categoria")
        df_categorias = pd.DataFrame(
            list(data["categorias"].items()),
            columns=["Categoria", "Valor (R$)"]
        )
        df_categorias["%"] = (df_categorias["Valor (R$)"] / data["pagamentos"] * 100).round(1)
        st.dataframe(df_categorias, use_container_width=True, hide_index=True)

    # Gráfico de evolução do saldo
    st.subheader("📉 Evolução Diária do Saldo")

    saldo = [data["saldo_inicial"]]
    for i in range(31):
        saldo.append(saldo[-1] + data["entradas_diarias"][i] - data["saidas_diarias"][i])
    saldo = saldo[1:]

    df_saldo = pd.DataFrame({
        "Dia": list(range(1, 32)),
        "Saldo": saldo
    })

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_saldo["Dia"],
        y=df_saldo["Saldo"],
        mode='lines+markers',
        name='Saldo',
        line=dict(color='#1976D2', width=2),
        marker=dict(size=6)
    ))
    fig_line.update_layout(
        xaxis_title="Dia do Mês",
        yaxis_title="Saldo (R$)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_line, use_container_width=True)


def render_upload_section():
    """Renderiza seção de upload de documentos"""
    st.subheader("📤 Importar Documentos")

    # Verificar se Mistral API está configurada
    if CONFIG_LOADED and not MISTRAL_API_KEY:
        st.warning("⚠️ MISTRAL_API_KEY não configurada. Configure no arquivo .env para usar OCR.")
        return

    tipo_doc = st.radio(
        "Tipo de documento",
        ["Recibo/Cupom", "Fatura de Cartão", "Extrato Bancário"],
        horizontal=True
    )

    uploaded_file = st.file_uploader(
        "Envie o arquivo",
        type=['png', 'jpg', 'jpeg', 'pdf', 'ofx', 'csv'],
        help="Formatos aceitos: PNG, JPG, PDF, OFX, CSV"
    )

    if uploaded_file:
        # Mostrar preview para imagens
        if uploaded_file.type.startswith('image/'):
            st.image(uploaded_file, caption="Preview do documento", width=300)

        if st.button("🔍 Processar com IA", use_container_width=True, type="primary"):
            # Salvar arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                with st.spinner("🤖 Extraindo dados com Mistral AI..."):
                    if CONFIG_LOADED:
                        processor = OCRProcessor()

                        if tipo_doc == "Recibo/Cupom":
                            resultado = processor.extrair_recibo(tmp_path)
                            if resultado:
                                st.success("✅ Dados extraídos com sucesso!")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Valor Total", f"R$ {resultado.get('valor_total', 0):,.2f}")
                                with col2:
                                    st.metric("Estabelecimento", resultado.get('estabelecimento', 'N/A'))

                                st.json(resultado)

                        elif tipo_doc == "Fatura de Cartão":
                            resultado = processor.extrair_fatura_cartao(tmp_path)
                            if resultado and "transacoes" in resultado:
                                st.success(f"✅ {len(resultado['transacoes'])} transações encontradas!")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Banco/Cartão", resultado.get('banco', 'N/A'))
                                with col2:
                                    st.metric("Valor Total", f"R$ {resultado.get('valor_total', 0):,.2f}")
                                with col3:
                                    st.metric("Vencimento", resultado.get('vencimento', 'N/A'))

                                if resultado['transacoes']:
                                    df_transacoes = pd.DataFrame(resultado['transacoes'])
                                    st.dataframe(df_transacoes, use_container_width=True, hide_index=True)

                        elif tipo_doc == "Extrato Bancário":
                            resultado = processor.extrair_extrato_bancario(tmp_path)
                            if resultado and "movimentacoes" in resultado:
                                st.success(f"✅ {len(resultado['movimentacoes'])} movimentações encontradas!")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Saldo Inicial", f"R$ {resultado.get('saldo_inicial', 0):,.2f}")
                                with col2:
                                    st.metric("Saldo Final", f"R$ {resultado.get('saldo_final', 0):,.2f}")

                                if resultado['movimentacoes']:
                                    df_mov = pd.DataFrame(resultado['movimentacoes'])
                                    st.dataframe(df_mov, use_container_width=True, hide_index=True)
                    else:
                        st.info("Módulo de OCR não disponível. Verifique a instalação.")

            except Exception as e:
                st.error(f"Erro ao processar documento: {e}")

            finally:
                # Limpar arquivo temporário
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


def render_manual_entry():
    """Renderiza formulário de entrada manual"""
    st.subheader("✏️ Adicionar Transação Manual")

    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)

        with col1:
            data_transacao = st.date_input("Data", value=date.today())
            valor = st.number_input("Valor (R$)", min_value=0.01, step=0.01)
            tipo = st.selectbox("Tipo", ["Despesa", "Receita"])

        with col2:
            estabelecimento = st.text_input("Estabelecimento/Descrição")
            categoria = st.selectbox(
                "Categoria",
                CATEGORIES if CONFIG_LOADED else [
                    "alimentação", "transporte", "moradia", "saúde",
                    "lazer", "educação", "outros"
                ]
            )

        submitted = st.form_submit_button("💾 Salvar Transação", use_container_width=True)

        if submitted:
            # Verificar risco da transação
            if CONFIG_LOADED and tipo == "Despesa":
                risk_check = check_transaction_risk(
                    amount=valor,
                    category=categoria,
                    description=estabelecimento
                )

                if risk_check.get("is_high_risk"):
                    st.warning(f"""
                        ⚠️ **Alerta de Compra por Impulso**

                        Score de Risco: {risk_check['score']}/100

                        {risk_check['recommendation']['message']}
                    """)

                    # Mostrar perguntas reflexivas
                    if "questions" in risk_check.get("recommendation", {}):
                        st.info("🤔 Reflita sobre estas perguntas:")
                        for q in risk_check["recommendation"]["questions"]:
                            st.write(f"• {q}")

                    # Botão para confirmar mesmo assim
                    if st.button("Confirmar mesmo assim", type="secondary"):
                        st.success("✅ Transação registrada!")
                else:
                    st.success("✅ Transação registrada com sucesso!")
            else:
                st.success("✅ Transação registrada com sucesso!")


def render_protection_settings():
    """Renderiza configurações de proteção"""
    st.subheader("🛡️ Proteção contra Impulsos")

    if not CONFIG_LOADED:
        st.info("Módulo de proteção não disponível.")
        return

    guard = ImpulseGuard()
    status = guard.get_protection_status()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            **Status da Proteção**
            - Ativada: {'✅ Sim' if status['enabled'] else '❌ Não'}
            - Período Noturno: {status['night_start']} - {status['night_end']}
            - Limite de Alerta: R$ {status['amount_threshold']:.2f}
        """)

        if status['is_night_period']:
            st.markdown("""
                <div class='alert-night'>
                    🌙 <strong>Modo Noturno Ativo</strong><br>
                    Transações serão analisadas com mais rigor.
                </div>
            """, unsafe_allow_html=True)

    with col2:
        # Toggle de proteção
        protection_enabled = st.toggle(
            "Proteção Noturna",
            value=status['enabled'],
            help="Ativa alertas para compras durante a madrugada"
        )

        if protection_enabled != status['enabled']:
            if protection_enabled:
                guard.enable_protection()
                st.success("Proteção ativada!")
            else:
                guard.disable_protection()
                st.warning("Proteção desativada!")
            st.rerun()


# === NOVAS SEÇÕES - FASE 2: INTELIGÊNCIA COMPORTAMENTAL ===

def render_insights_page():
    """Renderiza página de Insights Inteligentes (ML)"""
    st.subheader("🧠 Insights Inteligentes")

    if not ML_LOADED:
        st.warning("Módulos de ML não disponíveis. Verifique a instalação.")
        return

    # Carregar dados de exemplo
    transactions = get_sample_transactions()
    goals = get_sample_goals()

    # Tabs para diferentes insights
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Perfil Comportamental",
        "⚠️ Anomalias",
        "📈 Previsões",
        "💡 Sugestões de Economia"
    ])

    with tab1:
        render_behavioral_profile(transactions)

    with tab2:
        render_anomalies(transactions)

    with tab3:
        render_predictions(transactions)

    with tab4:
        render_savings_suggestions(transactions, goals)


def render_behavioral_profile(transactions):
    """Renderiza análise de perfil comportamental"""
    st.markdown("### 👤 Seu Perfil Financeiro")

    with st.spinner("Analisando seu perfil..."):
        profile = get_user_profile(transactions)

    if not profile or 'profile' not in profile:
        st.info("Dados insuficientes para análise de perfil.")
        return

    # Card do perfil principal
    col1, col2 = st.columns([2, 1])

    with col1:
        profile_name = profile['profile'].title()
        confidence = profile.get('confidence', 0)

        # Cores por perfil
        profile_colors = {
            'controlado': '🟢',
            'notívago': '🔴',
            'impulsivo': '🟠',
            'social': '🟡',
            'sazonal': '🔵'
        }

        emoji = profile_colors.get(profile['profile'], '⚪')

        st.markdown(f"""
        ### {emoji} Perfil: **{profile_name}**

        **Confiança:** {confidence:.0f}%

        **Descrição:** {profile.get('details', {}).get('description', 'N/A')}

        **Nível de Risco:** {profile.get('details', {}).get('risk_level', 'N/A').title()}
        """)

        # Características
        characteristics = profile.get('details', {}).get('characteristics', [])
        if characteristics:
            st.markdown("**Características identificadas:**")
            for char in characteristics:
                st.markdown(f"- {char}")

    with col2:
        # Gráfico radar dos scores
        all_scores = profile.get('all_scores', {})
        if all_scores:
            fig = go.Figure(data=go.Scatterpolar(
                r=list(all_scores.values()),
                theta=list(all_scores.keys()),
                fill='toself',
                line_color='#1f77b4'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    # Sugestões personalizadas
    suggestions = profile.get('details', {}).get('suggestions', [])
    if suggestions:
        st.markdown("### 💡 Recomendações para você")
        for i, suggestion in enumerate(suggestions, 1):
            st.info(f"{i}. {suggestion}")


def render_anomalies(transactions):
    """Renderiza detecção de anomalias"""
    st.markdown("### ⚠️ Gastos Atípicos Detectados")

    with st.spinner("Analisando transações..."):
        report = get_anomaly_report(transactions)

    if report.get('total_anomalies', 0) == 0:
        st.success("✅ Nenhum gasto atípico detectado! Seu padrão está consistente.")
        return

    # Métricas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Anomalias Detectadas",
            report['total_anomalies'],
            delta=f"{report.get('percentage', 0):.1f}% do total"
        )

    with col2:
        st.metric(
            "Valor Total Anômalo",
            f"R$ {report.get('total_value', 0):,.2f}"
        )

    with col3:
        st.metric(
            "Score Médio",
            f"{report.get('average_score', 0):.0f}/100"
        )

    # Lista de anomalias
    top_anomalies = report.get('top_anomalies', [])
    if top_anomalies:
        st.markdown("#### 🔍 Principais Gastos Atípicos")

        for i, anomaly in enumerate(top_anomalies[:5], 1):
            with st.expander(
                f"#{i} - R$ {anomaly.get('amount', 0):,.2f} | {anomaly.get('category', 'N/A')} | Score: {anomaly.get('anomaly_score', 0):.0f}",
                expanded=(i == 1)
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    - **Data:** {anomaly.get('date', 'N/A')}
                    - **Estabelecimento:** {anomaly.get('merchant', 'N/A')}
                    - **Categoria:** {anomaly.get('category', 'N/A')}
                    """)
                with col2:
                    reasons = anomaly.get('anomaly_reasons', [])
                    if reasons:
                        st.markdown("**Motivos:**")
                        for reason in reasons:
                            st.warning(f"⚠️ {reason}")

    # Distribuição por categoria
    categories = report.get('categories', {})
    if categories:
        st.markdown("#### 📊 Anomalias por Categoria")
        df_cat = pd.DataFrame([
            {'Categoria': k, 'Quantidade': v['count'], 'Total': v['total']}
            for k, v in categories.items()
        ])
        fig = px.bar(df_cat, x='Categoria', y='Total', color='Quantidade',
                     title="Valor anômalo por categoria")
        st.plotly_chart(fig, use_container_width=True)


def render_predictions(transactions):
    """Renderiza previsões de gastos"""
    st.markdown("### 📈 Previsão de Gastos")

    # Selecionar período
    days_ahead = st.slider("Prever para os próximos:", 7, 90, 30, step=7)

    with st.spinner(f"Calculando previsão para {days_ahead} dias..."):
        prediction = predict_spending(transactions, days_ahead)

    if 'error' in prediction:
        st.warning(prediction['error'])
        return

    # Métricas de previsão
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            f"Previsão ({days_ahead} dias)",
            f"R$ {prediction.get('total_predicted', 0):,.2f}",
            help=f"Método: {prediction.get('method', 'N/A')}"
        )

    with col2:
        st.metric(
            "Média Diária Prevista",
            f"R$ {prediction.get('average_daily', 0):,.2f}"
        )

    with col3:
        confidence = prediction.get('confidence_level', 0.95) * 100
        st.metric(
            "Confiança",
            f"{confidence:.0f}%"
        )

    # Gráfico de previsão
    if 'predictions' in prediction:
        predictions_df = pd.DataFrame(prediction['predictions'])
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])

        fig = go.Figure()

        # Linha de previsão
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['predicted'],
            mode='lines',
            name='Previsão',
            line=dict(color='#1f77b4', width=2)
        ))

        # Intervalo de confiança
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['upper'],
            mode='lines',
            name='Limite Superior',
            line=dict(color='#1f77b4', width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['lower'],
            mode='lines',
            name='Limite Inferior',
            line=dict(color='#1f77b4', width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            showlegend=False
        ))

        fig.update_layout(
            title="Previsão de Gastos Diários",
            xaxis_title="Data",
            yaxis_title="Valor (R$)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Análise de tendência
    analysis = analyze_spending(transactions)
    trends = analysis.get('trends', {})

    if trends:
        st.markdown("#### 📉 Análise de Tendência")
        col1, col2 = st.columns(2)

        with col1:
            direction = trends.get('direction', 'N/A')
            direction_emoji = '📈' if direction == 'crescente' else '📉'
            st.info(f"""
            {direction_emoji} **Tendência:** {direction.title()}

            **Força:** {trends.get('strength', 'N/A').title()}

            **R²:** {trends.get('r_squared', 0):.3f}
            """)

        with col2:
            st.info(f"""
            📊 **Média Diária Atual:** R$ {trends.get('daily_average', 0):.2f}

            📅 **Projeção 30 dias:** R$ {trends.get('projection_30_days', 0):,.2f}
            """)


def render_savings_suggestions(transactions, goals):
    """Renderiza sugestões de economia"""
    st.markdown("### 💰 Sugestões de Economia")

    # Slider para meta de redução
    target_reduction = st.slider(
        "Meta de redução de gastos:",
        5, 30, 10, step=5,
        format="%d%%"
    ) / 100

    with st.spinner("Analisando oportunidades de economia..."):
        result = get_savings_suggestions(transactions, target_reduction)

    if not result or 'suggestions' not in result:
        st.info("Analisando seus dados para encontrar oportunidades...")
        return

    # Resumo
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Meta de Economia",
            f"R$ {result.get('target_savings', 0):,.2f}",
            delta=f"{target_reduction*100:.0f}% de redução"
        )

    with col2:
        st.metric(
            "Potencial Identificado",
            f"R$ {result.get('total_potential_savings', 0):,.2f}"
        )

    with col3:
        achievable = result.get('achievable', False)
        if achievable:
            st.success("✅ Meta alcançável!")
        else:
            st.warning("⚠️ Meta desafiadora")

    # Lista de sugestões
    suggestions = result.get('suggestions', [])

    st.markdown("#### 📋 Oportunidades de Economia")

    for i, suggestion in enumerate(suggestions[:8], 1):
        sug_type = suggestion.get('type', 'general')
        priority = suggestion.get('priority', 'média')
        priority_color = {'alta': '🔴', 'média': '🟡', 'baixa': '🟢'}.get(priority, '⚪')

        with st.expander(
            f"{priority_color} {suggestion.get('suggestion', 'Sugestão')} | Economia: R$ {suggestion.get('potential_savings', 0):,.2f}",
            expanded=(i <= 3)
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                if sug_type == 'category_reduction':
                    st.markdown(f"""
                    **Categoria:** {suggestion.get('category', 'N/A')}

                    **Gasto Atual:** R$ {suggestion.get('current_spending', 0):,.2f}

                    **Ação:** {suggestion.get('action', 'N/A')}
                    """)
                elif sug_type == 'recurring_expense':
                    st.markdown(f"""
                    **Estabelecimento:** {suggestion.get('merchant', 'N/A')}

                    **Frequência:** {suggestion.get('frequency', 0)} vezes

                    **Valor Médio:** R$ {suggestion.get('avg_transaction', 0):,.2f}
                    """)
                elif sug_type == 'behavioral':
                    st.markdown(f"""
                    **Comportamento:** {suggestion.get('category', 'N/A')}

                    **Gasto Atual:** R$ {suggestion.get('current_spending', 0):,.2f}

                    **Ação:** {suggestion.get('action', 'N/A')}
                    """)

            with col2:
                st.metric(
                    "Economia Potencial",
                    f"R$ {suggestion.get('potential_savings', 0):,.2f}"
                )

    # Comparação com metas
    if goals:
        st.markdown("#### 🎯 Impacto nas suas Metas")

        total_potential = result.get('total_potential_savings', 0)

        for goal in goals:
            remaining = goal['target_amount'] - goal['current_amount']
            months_to_goal = remaining / total_potential if total_potential > 0 else float('inf')

            progress = (goal['current_amount'] / goal['target_amount']) * 100

            st.markdown(f"""
            **{goal['name']}**

            Progresso atual: {progress:.1f}%

            Com a economia identificada, você alcançaria esta meta em **{months_to_goal:.1f} meses**
            """)

            st.progress(min(progress / 100, 1.0))


def render_notifications():
    """Renderiza central de notificações"""
    st.subheader("🔔 Notificações")

    if not CONFIG_LOADED:
        st.info("Sistema de notificações não disponível.")
        return

    user_id = st.session_state.get('username', 'demo')

    # Buscar notificações
    notifications = get_user_notifications(user_id, limit=20)

    if not notifications:
        st.info("Você não tem notificações no momento.")
        return

    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox(
            "Filtrar por tipo",
            ["Todas", "Info", "Alertas", "Críticas"],
            key="notif_filter"
        )
    with col2:
        unread_only = st.checkbox("Apenas não lidas", key="notif_unread")

    # Marcar todas como lidas
    if st.button("✅ Marcar todas como lidas"):
        manager = get_notification_manager()
        manager.mark_all_as_read(user_id)
        st.success("Notificações marcadas como lidas!")
        st.rerun()

    # Lista de notificações
    for notif in notifications:
        if unread_only and notif.get('read'):
            continue

        notif_type = notif.get('type', 'info')
        type_icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'alert': '🚨',
            'success': '✅',
            'critical': '🛑'
        }

        icon = type_icons.get(notif_type, 'ℹ️')
        read_status = "" if notif.get('read') else "🔵 "

        with st.expander(f"{read_status}{icon} {notif.get('title', 'Notificação')}"):
            st.markdown(notif.get('message', ''))
            st.caption(f"Recebido em: {notif.get('created_at', 'N/A')}")


def render_sidebar():
    """Renderiza sidebar com navegação"""
    with st.sidebar:
        st.title("📊 Menu")

        page = st.radio(
            "Navegação",
            [
                "Dashboard",
                "🧠 Insights (ML)",
                "Importar Documentos",
                "Nova Transação",
                "🔔 Notificações",
                "Configurações"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        # Info do período
        st.markdown(f"""
            **Período Atual**
            📅 {datetime.now().strftime('%B %Y')}
        """)

        # Contador de notificações não lidas
        if CONFIG_LOADED:
            try:
                manager = get_notification_manager()
                unread_count = manager.get_unread_count(st.session_state.get('username'))
                if unread_count > 0:
                    st.info(f"🔔 {unread_count} notificação(ões) não lida(s)")
            except Exception:
                pass

        # Alerta noturno no sidebar
        if CONFIG_LOADED and is_night_mode():
            st.warning("🌙 Modo noturno ativo")

        return page


# === MAIN APP ===
def main():
    """Função principal do aplicativo"""

    # Verificar autenticação
    if not check_authentication():
        return

    # Navegação
    page = render_sidebar()

    # Header
    render_header()

    st.divider()

    # Carregar dados
    data = get_sample_data()

    # Renderizar página selecionada
    if page == "Dashboard":
        render_kpis(data)
        st.divider()
        render_charts(data)

    elif page == "🧠 Insights (ML)":
        render_insights_page()

    elif page == "Importar Documentos":
        render_upload_section()

    elif page == "Nova Transação":
        render_manual_entry()

    elif page == "🔔 Notificações":
        render_notifications()

    elif page == "Configurações":
        render_protection_settings()

        st.divider()

        st.subheader("ℹ️ Sobre o Sistema")
        st.markdown("""
            **Dashboard Financeiro v2.1** - Fase 2: Inteligência Comportamental

            Funcionalidades:
            - 📊 Visualização de gastos e receitas
            - 📸 OCR de recibos e faturas (Mistral AI)
            - 🤖 Categorização automática (ML)
            - 🧠 **NOVO:** Detecção de anomalias (Isolation Forest)
            - 📈 **NOVO:** Previsão de gastos (Séries Temporais)
            - 👤 **NOVO:** Perfil comportamental (Clustering)
            - 💡 **NOVO:** Sugestões personalizadas de economia
            - 🛡️ Proteção contra compras por impulso
            - 🔔 **NOVO:** Central de notificações
            - 📱 Interface responsiva (mobile/tablet)

            Desenvolvido com Streamlit + Python + scikit-learn
        """)

    # Footer
    st.divider()
    st.caption("Dashboard Financeiro v2.1 | Feito com ❤️ usando Streamlit + ML")


if __name__ == "__main__":
    main()
