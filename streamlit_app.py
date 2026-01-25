"""
Dashboard Financeiro Pessoal
Sistema inteligente de gestão financeira com OCR e análise comportamental
"""
import os
import tempfile
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd
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
    from ml.categorizer import categorize_transaction, Categorizer
    from behavioral.impulse_guard import ImpulseGuard, check_transaction_risk, is_night_mode

    CONFIG_LOADED = True
except ImportError as e:
    CONFIG_LOADED = False
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


def render_sidebar():
    """Renderiza sidebar com navegação"""
    with st.sidebar:
        st.title("📊 Menu")

        page = st.radio(
            "Navegação",
            ["Dashboard", "Importar Documentos", "Nova Transação", "Configurações"],
            label_visibility="collapsed"
        )

        st.divider()

        # Info do período
        st.markdown(f"""
            **Período Atual**
            📅 {datetime.now().strftime('%B %Y')}
        """)

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

    elif page == "Importar Documentos":
        render_upload_section()

    elif page == "Nova Transação":
        render_manual_entry()

    elif page == "Configurações":
        render_protection_settings()

        st.divider()

        st.subheader("ℹ️ Sobre o Sistema")
        st.markdown("""
            **Dashboard Financeiro v2.0**

            Funcionalidades:
            - 📊 Visualização de gastos e receitas
            - 📸 OCR de recibos e faturas (Mistral AI)
            - 🤖 Categorização automática (ML)
            - 🛡️ Proteção contra compras por impulso
            - 📱 Interface responsiva (mobile/tablet)

            Desenvolvido com Streamlit + Python
        """)

    # Footer
    st.divider()
    st.caption("Dashboard Financeiro | Feito com ❤️ usando Streamlit")


if __name__ == "__main__":
    main()
