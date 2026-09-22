# Changelog

Todas as alterações notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado no [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Adicionado (Added)
- **Infraestrutura e Configurações**:
  - Arquivo `.env.example` com template de credenciais e endpoint `/api/alpha/decisions` da OpenRouter.
  - Arquivo `.gitignore` configurado para isolar ambientes virtuais, caches e chaves sensíveis.
  - Arquivo `requirements.txt` com as bibliotecas essenciais (`httpx`, `streamlit`, `plotly`, `pandas`, `pydantic`, `python-dotenv`, `numpy`, `scipy`, `opencv-python`).
  - Módulo `config.py` centralizando parâmetros de calibração espacial, tolerâncias de engajamento e conexão com o modelo `typesafe/jev-1.13`.
  - Documentação completa no `README.md` detalhando arquitetura, guia de instalação e execução.
- **Visão Computacional e Física Espacial Local**:
  - Módulo `yolo_stream.py` com modelos Pydantic de telemetria de detecções por frame (`Detection`, `YoloFrame`) e gerador sintético de 4 cenários comportamentais da feira (*Lead Quente*, *Passante Rápido*, *Pico de Demonstração*, *Risco de Biossegurança*).
  - Motor espacial `sitrep_engine.py` implementando homografia 2D de solo, rastreamento cinético de velocidades vetoriais, acumulador de permanência (*Dwell Time*), vetor de alinhamento de atenção visual e compilação do SITREP em JSON enxuto.
- **Decisão e Séries Temporais**:
  - Configuração `questions_config.py` das 4 perguntas de probabilidade `noul` do Jev (`p_purchase_intent`, `p_sales_approach_urgency`, `p_demo_attractiveness_peak`, `p_airborne_transmission_risk`) com instruções e critérios estritos.
  - Cliente HTTP `jev_client.py` assíncrono e síncrono para o endpoint da OpenRouter com extração segura de probabilidades, medição de latência RTT e fallback inteligente (*Smart Dry-Run*).
  - Gerenciador temporal `history_tracker.py` com Ring Buffer cronológico e filtro de suavização temporal por Média Móvel Exponencial (EMA) para amortecimento de oclusões transitórias.
  - Suíte de testes `test_connection.py` cobrindo validação de schema estrito, integridade numérica, teste de sensibilidade dinâmica e absorção de ruído pelo filtro EMA.
  - Aplicação executiva `dashboard.py` construída em Streamlit + Plotly exibindo 4 gráficos contínuos de séries temporais, limiares em 70%, indicadores físicos do estande e controles interativos em tempo real.

### Modificado (Changed)
- Ajustada a chave de instrução de perguntas para `instructions` (plural) conforme especificação validada no endpoint `/api/alpha/decisions` da OpenRouter.
- Calibrados os limiares de sensibilidade dinâmica dos cenários sintéticos de teste.
- Adicionado modo de execução **Manual por Botão** como padrão no Dashboard para evitar consumo contínuo e involuntário de tokens da OpenRouter.
- Adicionado botão **Avançar 1 Frame Local (0 Tokens)** para permitir visualização da física espacial sem realizar requisições de rede.
- Adicionada suíte de testes de sensibilidade e auditoria de conta [run_sensitivity_suite.py](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/run_sensitivity_suite.py), registrando logs de saldo e respostas em `logs/sensitivity_test.log`.
- Expandida arquitetura para **Visão em Dupla Escala**:
  - Módulo [notebook_perception.py](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/notebook_perception.py) para rastreamento de contato visual, tempo de atenção e satisfação do usuário na câmera frontal do notebook.
  - Enriquecimento de `yolo_stream.py` com keypoints COCO-17 do YOLOv8-Pose e ângulo de rotação da cabeça (*Head Yaw*).
  - Adição da 5ª pergunta `p_user_satisfaction_engagement` no Jev para avaliação de experiência do usuário.
  - Diagnóstico executivo no Dashboard para detecção imediata de **Oportunidades Comerciais Perdidas** (leads de alto interesse não atendidos pela equipe de vendas).
- **Métrica Padrão de Feiras (CEIR / RetailNext)**:
  - Formulação e implementação do **QEI (Qualified Engagement Index)**:
    $$\text{QEI} = (\text{Taxa de Captura} \times \text{Taxa de Experimentação} \times \text{Fator de Qualidade \& Satisfação}) \times 100$$
    Projetado especificamente para avaliação e testes A/B de impacto de arranjos de estande e horários nobres de engajamento.
  - Expansão do Dashboard Streamlit para **6 Séries Temporais Contínuas (Grade 2x3)** em Plotly, integrando a curva temporal contínua do QEI com suavização por filtro EMA.
- **Validação de Sensibilidade do Jev + QEI (API Real OpenRouter)**:
  - Suíte controlada [run_sensitivity_suite.py](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/run_sensitivity_suite.py) auditando 4 cenários com log detalhado em `logs/sensitivity_test.log`:
    - *Passante Rápido*: RTT 379ms | $\text{QEI} = 0.0$ | $P(\text{Compra}) = 6.0\%$ | $P(\text{Urgência}) = 10.0\%$ | $P(\text{Satisfação}) = 2.0\%$
    - *Lead Quente sem Vendedor*: RTT 389ms | $\text{QEI} = 79.6$ | $P(\text{Compra}) = 69.0\%$ | $P(\text{Urgência}) = 67.0\%$ | $P(\text{Satisfação}) = 84.0\%$
    - *Pico de Demonstração*: RTT 413ms | $\text{QEI} = 72.7$ | $P(\text{Compra}) = 80.0\%$ | $P(\text{Urgência}) = 10.0\%$ | $P(\text{Satisfação}) = 91.0\%$
    - *Risco Biossegurança*: RTT 487ms | $\text{QEI} = 7.7$ | $P(\text{Compra}) = 41.0\%$ | $P(\text{Urgência}) = 50.0\%$ | $P(\text{Risco}) = 83.0\%$ | $P(\text{Satisfação}) = 4.0\%$

- **Planta Baixa 2D e Mapa de Calor de Retenção (Dwell Heatmap)**:
  - Módulo [spatial_heatmap.py](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/spatial_heatmap.py) implementando grade contínua de solo ($80 \times 60$ células de $10\text{ cm}$), decaimento térmico exponencial por resfriamento contínuo e convolução gaussiana 2D para visualização fluida.
  - Diferenciação determinística entre trânsito rápido no corredor e paradas prolongadas de retenção (*Dwell Time*) em frente ao totem/notebook com consumo de **zero tokens da API**.
  - Expansão do [dashboard.py](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/dashboard.py) com sistema de abas de tela cheia:
    - *Aba 1: Séries Temporais & QEI* (6 curvas contínuas com EMA, KPIs e alertas).
    - *Aba 2: Planta Baixa 2D & Mapa de Calor* (planta top-down do estande, corredor, totem, balcão, pedestres em tempo real com vetores de olhar e métricas de retenção).
- **Documentação de Metodologia para Replicação**:
  - Especificação formal no documento [METODOLOGIA_MAPA_CALOR_ESPACIAL.md](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/METODOLOGIA_MAPA_CALOR_ESPACIAL.md) cobrindo homografia projetiva 2D, formulações da dinâmica de calor, contratos de interface e guia de integração para modelos alternativos (YOLOv11, RT-DETR, MediaPipe, ByteTrack).
- **Refatoração para Auditoria Executiva de Feira para a Liderança**:
  - Substituição da métrica de abordagem comercial `p_sales_approach_urgency` por `p_stand_retention_engagement` (*Retenção e Engajamento no Estande*), mensurando a capacidade da fachada e mostruário em capturar a atenção de transeuntes.
  - Simulação de multidão realista de feira de tecnologia em [yolo_stream.py](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/yolo_stream.py), com fluxo contínuo de 14 a 18 pedestres bidirecionais no corredor e visitantes retidos no estande.
  - Implementação de métricas auditadas de público em [sitrep_engine.py](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/sitrep_engine.py): `corridor_passersby_count` e `booth_visitors_retained_count`.
  - Reorientação dos alertas do [dashboard.py](file:///c:/Users/andre/Documents/Sistemas/Jev_yolo/dashboard.py) para auditoria executiva de evento (*"ALTO ENGAJAMENTO AUDITADO NO ESTANDE"*).

### Removido (Removed)
- **Eliminação Completa da Suposição de Vendedor / Promotor Comercial**:
  - Removido o rastreamento arbitrário de vendedor (`track_id=999`) e representação de balcão de vendas na planta baixa.
  - Removidos os atributos `sales_rep_distance_meters`, `sales_rep_interacting` e `sales_rep_pos` de todos os módulos (`sitrep_engine.py`, `spatial_heatmap.py`, `test_connection.py`, `run_sensitivity_suite.py`), tratando todos os indivíduos captados pelo YOLO estritamente como público auditado.

### Corrigido (Fixed)
- Corrigido `AttributeError: 'HistoryTracker' object has no attribute '_last_filtered'` em `history_tracker.py` decorrente da omissão do dicionário de estado de filtro após inclusão dos atributos de QEI.
- Implementado mecanismo de autocura e compatibilidade defensiva em `HistoryTracker.add_point()` e `dashboard.py:initialize_session_state()` para restaurar atributos ausentes em instâncias pré-existentes no cache do navegador sem necessidade de reiniciar a sessão do usuário.
