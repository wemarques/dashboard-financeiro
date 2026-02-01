# 💰 Dashboard Financeiro Pessoal

Sistema inteligente de gestão financeira pessoal com OCR, Machine Learning e análise comportamental.

## ✨ Funcionalidades

### 📊 Dashboard
- Visualização de receitas e despesas
- KPIs financeiros (entradas, saídas, saldo, taxa de poupança)
- Gráficos interativos por período e categoria
- Evolução diária do saldo

### 📸 OCR Inteligente (Mistral AI)
- Upload de recibos e cupons fiscais
- Importação de faturas de cartão de crédito
- Extração de extratos bancários (PDF, OFX, CSV)
- Categorização automática das transações

### 🤖 Machine Learning
- **Detecção de anomalias**: Identifica gastos suspeitos (Isolation Forest)
- **Categorização automática**: NLP + Classificador para categorizar transações
- **Previsão de gastos**: Séries temporais para projetar gastos futuros
- **Clustering**: Agrupa padrões de consumo (K-Means)
- **Sugestões personalizadas**: Recomendações de economia baseadas em ML

### 🛡️ Proteção Comportamental
- **Modo Noturno**: Alertas para compras entre 00h-06h
- **Detecção de Impulso**: Identifica compras por impulso
- **Perguntas Reflexivas**: Ajuda na tomada de decisão
- **Delay de Confirmação**: Tempo para reflexão em compras de alto risco

### 📱 Interface Responsiva
- Design mobile-first
- Funciona em smartphones, tablets e desktop
- Autenticação com senha

## 🚀 Instalação

### Pré-requisitos
- Python 3.10+
- PostgreSQL 14+
- Poppler (para PDF)

### Passos

1. **Clone o repositório**
```bash
git clone https://github.com/wemarques/dashboard-financeiro.git
cd dashboard-financeiro
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

5. **Configure o banco de dados PostgreSQL**
```bash
createdb dashboard_financeiro
```

6. **Execute a aplicação**
```bash
streamlit run streamlit_app.py
```

## 🔧 Configuração

### Variáveis de Ambiente (.env)

```env
# Mistral AI (OCR)
MISTRAL_API_KEY=sua_chave_aqui

# PostgreSQL
DATABASE_URL=postgresql://usuario:senha@localhost:5432/dashboard_financeiro

# Configurações
NIGHT_START=00:00
NIGHT_END=06:00
IMPULSE_THRESHOLD=100.0
```

### Credenciais de Teste

Para ambiente de desenvolvimento:
- Usuário: `demo`
- Senha: `demo123`

## 📁 Estrutura do Projeto

```
dashboard-financeiro/
├── .env.example              # Template de variáveis de ambiente
├── .gitignore
├── requirements.txt          # Dependências Python
├── config.py                 # Configurações globais
├── streamlit_app.py          # Aplicação principal
├── .streamlit/
│   └── config.toml           # Configurações do Streamlit
├── data/
│   └── dados.csv             # Dados de exemplo
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Sistema de logging
│   ├── data_loader.py        # Gerenciador PostgreSQL
│   └── ocr_mistral.py        # OCR com Mistral AI
├── ml/
│   ├── __init__.py
│   └── categorizer.py        # Categorização automática
├── behavioral/
│   ├── __init__.py
│   └── impulse_guard.py      # Proteção contra impulsos
└── tests/
    ├── __init__.py
    ├── test_categorizer.py
    ├── test_impulse_guard.py
    └── test_ocr.py
```

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar teste específico
pytest tests/test_categorizer.py -v

# Com cobertura
pytest tests/ --cov=. --cov-report=html
```

## 📊 Roadmap

### Fase 1: MVP ✅
- [x] Dashboard básico
- [x] OCR com Mistral AI
- [x] Categorização automática
- [x] Proteção noturna
- [x] Autenticação
- [x] Interface responsiva

### Fase 2: Inteligência (Em desenvolvimento)
- [ ] Detecção de anomalias
- [ ] Previsão de gastos
- [ ] Perfil comportamental
- [ ] Intervenções personalizadas

### Fase 3: Maturidade
- [ ] Relatórios automáticos
- [ ] Integração Open Banking
- [ ] Metas financeiras
- [ ] Exportação de dados

## 🤝 Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👤 Autor

**Wellington Marques**
- GitHub: [@wemarques](https://github.com/wemarques)

---

Feito com ❤️ usando Streamlit, Mistral AI e Python
