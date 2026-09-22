# JEV + YOLO Time Series Analytics

Pipeline de alta performance para validação e testes quantitativos da comunicação entre **Visão Computacional Local (YOLOv8/v11 + rastreamento de IDs e física espacial)** e o modelo de decisão estruturada **Jev (TypeSafe AI)** via endpoint da OpenRouter (`/api/alpha/decisions`).

O sistema alimenta um **Dashboard em tempo real de Séries Temporais** exibido no notebook, onde cada gráfico plota a evolução temporal contínua de **UMA probabilidade** calculada pelo Jev.

---

## 🏛️ Arquitetura do Sistema

```
[ Câmera / Telemetria YOLO ] ──▶ [ Motor Espacial Local (sitrep_engine.py) ]
                                            │
                                            ▼ Homografia 2D, Dwell Time, Vetores
                                  [ SITREP JSON Enxuto ]
                                            │
                                            ▼ OpenRouter /api/alpha/decisions
                                  [ Jev Decision Engine ]
                                            │
                                            ▼ 4 Probabilidades 'noul'
                                  [ Buffer Time Series + Filtro EMA ]
                                            │
                                            ▼
                                  [ Dashboard Streamlit + Plotly ]
```

---

## 📋 As Perguntas de Probabilidade (`noul`)

Todas as decisões utilizam a primitiva estrita **`noul`** ($P \in [0.0, 1.0]$) para auditoria executiva de público:

1. **`p_purchase_intent`**: Intenção real de compra / interesse comercial no estande.
2. **`p_stand_retention_engagement`**: Avaliação da retenção e engajamento do visitante no espaço do estande (rompimento do fluxo do corredor).
3. **`p_demo_attractiveness_peak`**: Pico de atração e retenção de público na tela interativa/totem.
4. **`p_airborne_transmission_risk`**: Risco de transmissão aérea respiratória sob aglomeração estática em ambiente fechado.
5. **`p_user_satisfaction_engagement`**: Nível de engajamento, contato visual e satisfação na interação com a demonstração.

---

## 📁 Estrutura de Arquivos

```
Jev_yolo/
├── .env.example              # Modelo de credenciais da OpenRouter e configurações
├── .gitignore                # Proteção de credenciais e caches
├── requirements.txt          # Dependências do projeto (httpx, streamlit, plotly, etc.)
├── README.md                 # Documentação técnica e guia operacional
├── PLAN_JEV_YOLO.md          # Especificação técnica e arquitetural de referência
├── CHANGELOG.md              # Histórico de alterações seguindo Keep a Changelog
├── config.py                 # Validação de credenciais e parâmetros globais
├── questions_config.py       # Definição das 4 perguntas noul e critérios
├── yolo_stream.py            # Contratos de detecção e simulador de cenários da feira
├── sitrep_engine.py          # Motor Espacial: Homografia 2D, cinemática e SITREP
├── jev_client.py             # Cliente HTTP assíncrono/síncrono para /api/alpha/decisions
├── history_tracker.py        # Ring Buffer e suavização temporal via filtro EMA
├── test_connection.py        # Suíte de 5 testes rigorosos de comunicação e robustez
└── dashboard.py              # Aplicação Streamlit com 4 gráficos de séries temporais
```

---

## 🚀 Instalação e Execução

### 1. Criar e Ativar Ambiente Virtual
```bash
python -m venv .venv
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1
# No Linux/macOS:
source .venv/bin/activate
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Configurar Variáveis de Ambiente
Copie o arquivo de exemplo `.env.example` para `.env`:
```bash
cp .env.example .env
```
Edite o arquivo `.env` inserindo sua chave da OpenRouter:
```env
OPENROUTER_API_KEY=sua_chave_openrouter_aqui
JEV_MODEL=typesafe/jev-1.13
```
> **Nota:** Caso execute sem chave configurada, o sistema entra automaticamente em **Modo Demonstração (Smart Dry-Run)**, calculando as probabilidades com base no motor analítico local sem interrupções.

### 4. Executar os Testes de Validação
```bash
python test_connection.py
```
A suíte valida:
1. Conformidade de Schema Estrito TypeSafe (OpenRouter `/api/alpha/decisions`).
2. Extração numérica das 4 probabilidades float $[0.0, 1.0]$.
3. Sensibilidade dinâmica a alterações no comportamento dos visitantes.
4. Amortecimento de ruído e oclusões transitórias pelo filtro EMA.
5. Medição de latência RTT em milissegundos.

### 5. Iniciar o Dashboard em Tempo Real
```bash
streamlit run dashboard.py
```
O dashboard será aberto no navegador exibindo os 4 gráficos de séries temporais, seletor de cenários, cartões KPI executivos e linha de limiar em 70%.
