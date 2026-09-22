# Plano de Implementação: JEV + YOLO Time Series Analytics (Feira de Tecnologia)

## 1. Visão Geral do Sistema e Arquitetura de Comunicação

O objetivo deste projeto é validar e testar de forma rigorosa e quantitativa a comunicação entre **visão computacional local (YOLOv8/v11 + rastreamento de IDs e física espacial)** e o modelo de decisão estruturada **Jev (TypeSafe AI)** via endpoint da OpenRouter (`https://openrouter.ai/api/alpha/decisions`).

A aplicação alimenta um **Dashboard em tempo real de Séries Temporais (Time Series)** exibido na tela de um notebook, onde **cada gráfico plota a evolução temporal contínua de UMA probabilidade** calculada pelo Jev.

### Pipeline Completo de Dados (Do Sensor à Decisão):
```
[ Câmera / Webcam do Notebook ]
              │
              ▼
[ YOLO (v8 / v11 / YOLO-World) + ByteTrack ]
   - Extrai por frame: [track_id, bbox_xyxy, confidence, class_name]
              │
              ▼
[ Motor Espacial e Cinético Local (sitrep_engine.py) ]
   - Projeção de solo via Homografia 2D (pixels -> metros no plano da feira)
   - Filtro de Kalman / Diferença Finita: velocidades vetoriais (Vx, Vy) e aceleração
   - Matriz de distâncias euclidianas interpessoais e com o mostruário/totem
   - Acumulador de permanência estática (Co-presence & Engagement Dwell Time)
   - Cálculo de entropia vetorial / dispersão de fluxo de caminhada
              │
              ▼
[ SITREP em JSON (Compilado para o `state` do Jev) ]
   - Objeto semântico e dimensional enxuto (sem coordenadas brutas para evitar a "jaggedness" do Jev)
              │
              ▼
[ Jev Decision Engine (OpenRouter API: /api/alpha/decisions) ]
   - Avaliação paralela das 4 perguntas `noul` em um único ciclo de inferência (~100-180ms)
              │
              ▼
[ Buffer de Série Temporal (history_tracker.py) ]
   - Ring Buffer cronológico indexado por timestamp_ms: [t, p1, p2, p3, p4]
              │
              ▼
[ Dashboard Streamlit + Plotly no Notebook (dashboard.py) ]
   - 4 Gráficos Time Series contínuos (1 por pergunta de probabilidade)
   - Indicadores executivos de oportunidade comercial e alertas ambientais
```

---

## 2. Contrato de Entrada do YOLO e Síntese da Física Local

### 2.1. Saída Bruta por Frame do YOLO (`yolo_stream.py`)
O detector e tracker local produzem a telemetria básica:
```json
{
  "frame_id": 4820,
  "timestamp_ms": 1727021400500,
  "detections": [
    {
      "track_id": 14,
      "class_name": "person",
      "confidence": 0.94,
      "bbox_xyxy": [380, 210, 490, 680],
      "center_foot_px": [435, 680],
      "head_bbox_xyxy": [410, 210, 460, 280]
    },
    {
      "track_id": 22,
      "class_name": "person",
      "confidence": 0.91,
      "bbox_xyxy": [620, 240, 710, 700],
      "center_foot_px": [665, 700],
      "head_bbox_xyxy": [645, 240, 685, 295]
    }
  ]
}
```

### 2.2. A Transformação Matemática em Métricas de Engenharia (`sitrep_engine.py`)
Para contornar a incapacidade do Jev de calcular geometria bruta (*jaggedness* do modelo), o `sitrep_engine.py` executa localmente:
1. **Homografia 2D de Solo**:
   - $P_{\text{metro}} = H \cdot P_{\text{pixel}}$, onde $H$ é a matriz de perspectiva 3x3 obtida na calibração de 4 pontos no chão do estande.
2. **Distância e Tempo de Permanência (Dwell Time)**:
   - Mede a distância entre o visitante $i$ e o display: $D_i(t) = \| P_i(t) - P_{\text{display}} \|$.
   - Se $D_i(t) < 1.5\text{m}$ e $\| v_i(t) \| < 0.2\text{m/s}$, incrementa $\text{dwell\_time}_i += \Delta t$.
3. **Vetor de Atenção / Gaze Proxy**:
   - Razão entre a posição da cabeça em relação ao tronco e a localização da tela/demonstração.
4. **Distância do Promotor de Vendas**:
   - Identificação do promotor (via credencial/posição base ou etiqueta) e medição da menor distância até os visitantes retidos.
5. **Aglomeração e Exposição Respiratória**:
   - Contagem de pares com distância interpessoal $< 1.0\text{m}$ sustentada por tempo contínuo $t > 45\text{s}$.

### 2.3. O JSON Final Enviado no `state` ao Jev
```json
{
  "zone_telemetry": {
    "location": "corredor_feira_tecnologia",
    "booth_capacity_nominal": 15,
    "ventilation_status": "enclosed_pavilion_low_turnover"
  },
  "metrics": {
    "active_visitors_count": 4,
    "max_dwell_time_seconds": 38.0,
    "gaze_alignment_with_demo_pct": 78.0,
    "walking_velocity_mps": 0.15,
    "sales_rep_distance_meters": 3.2,
    "sales_rep_interacting": false,
    "crowd_density_people_per_sqm": 2.2,
    "sustained_close_contact_seconds": 52.0
  }
}
```

---

## 3. As 4 Perguntas de Probabilidade no Jev (Todas tipo `noul`)

Todas as 4 perguntas utilizam rigorosamente a primitiva **`noul`**, garantindo retornos numéricos de probabilidade contínua ($P \in [0.0, 1.0]$) para plotagem nas séries temporais:

### 1. `p_purchase_intent`
* **Pergunta**: "Qual a probabilidade de o visitante em frente ao estande ter intenção real de compra ou interesse comercial?"
* **Tipo**: `noul`
* **Instrução**: "Avalie a probabilidade de o visitante em frente ao estande ter intenção real de compra ou interesse comercial."
* **Critérios**:
  - `false`: Visitante em trânsito rápido, olhando para o corredor geral ou sem foco no produto.
  - `true`: Visitante desacelerou ou parou em frente ao mostruário com corpo e olhar voltados para a demonstração.

### 2. `p_sales_approach_urgency`
* **Pergunta**: "Qual a probabilidade de a equipe de vendas precisar abordar o visitante imediatamente para não perder o contato?"
* **Tipo**: `noul`
* **Instrução**: "Avalie a probabilidade de a equipe de vendas precisar abordar o visitante imediatamente para não perder o contato."
* **Critérios**:
  - `false`: Visitante já atendido ou em deslocamento rápido sem engajamento.
  - `true`: Visitante retido e demonstrando interesse sem nenhum vendedor próximo para atendê-lo.

### 3. `p_demo_attractiveness_peak`
* **Pergunta**: "Qual a probabilidade de a demonstração do produto estar atingindo um pico de atração e retenção de público?"
* **Tipo**: `noul`
* **Instrução**: "Avalie a probabilidade de a demonstração do produto estar atingindo um pico de atração e retenção de público."
* **Critérios**:
  - `false`: Público disperso, ignorando a tela interativa ou totem.
  - `true`: Múltiplos visitantes convergindo e mantendo foco simultâneo na demonstração.

### 4. `p_airborne_transmission_risk`
* **Pergunta**: "Qual a probabilidade de haver risco aumentado de transmissão aérea de vírus respiratórios no local?"
* **Tipo**: `noul`
* **Instrução**: "Avalie a probabilidade de haver risco aumentado de transmissão aérea de vírus respiratórios no local."
* **Critérios**:
  - `false`: Ambiente ventilado ou pessoas em movimento com espaçamento adequado.
  - `true`: Aglomeração densa e estática por tempo prolongado sob ar estagnado.

---

## 4. Estratégia de Testes de Comunicação, Formato e Robustez (`format_tester.py` / `test_connection.py`)

Para comprovar a efetividade da comunicação do pipeline YOLO -> Jev, implementamos 4 testes rigorosos:

1. **Validação de Schema Estrito TypeSafe (OpenRouter /api/alpha/decisions)**:
   - Valida conformidade do modelo (`typesafe/jev-1.13` ou `~typesafe/jev-latest`).
   - Verifica se o `state` é um JSON serializável compacto (< 1.000 tokens).
   - Verifica se o mapa `questions` contém exatamente as 4 chaves `noul`, cada qual com `instructions` e o par `criteria: {"false": "...", "true": "..."}`.
2. **Teste de Extração Numérica para Time Series**:
   - Confirmação do parsing seguro das 4 probabilidades float retornadas pela OpenRouter:
     `answers -> { question_id -> { "noul": float } }`.
3. **Teste de Sensibilidade Dinâmica a Variações do YOLO**:
   - Injeção de cenário transitório (Estado A: transeunte a 1.2 m/s -> Estado B: transeunte para a 0.05 m/s por 30s) para certificar que o Jev eleva as probabilidades $P(\text{compra})$ e $P(\text{urgência})$ de forma responsiva.
4. **Filtro de Histerese Temporal e Amortecimento de Ruído (EMA)**:
   - Aplicação de Média Móvel Exponencial nas probabilidades:
     $P_{\text{filtrado}}(t) = \alpha \cdot P_{\text{jev}}(t) + (1 - \alpha) \cdot P_{\text{filtrado}}(t - 1)$
     Evita que falsas oclusões de 1 frame do YOLO façam as curvas no dashboard oscilarem descontroladamente.
5. **Benchmarking de Latência e Throughput**:
   - Medição do tempo de ida e volta (RTT) da chamada assíncrona HTTP em requisições recorrentes a cada 1 ou 2 segundos.

---

## 5. Dashboard de Séries Temporais no Notebook (`dashboard.py`)

Construído em **Streamlit + Plotly**, configurado para rodar localmente no notebook:
* **Gráfico 1**: Série Temporal de $P(\text{Intenção de Compra})$ [0% a 100%].
* **Gráfico 2**: Série Temporal de $P(\text{Urgência de Abordagem})$ [0% a 100%].
* **Gráfico 3**: Série Temporal de $P(\text{Pico de Atração da Demo})$ [0% a 100%].
* **Gráfico 4**: Série Temporal de $P(\text{Risco de Transmissão Aérea})$ [0% a 100%].
* **Linhas de Limiar (Thresholds)**: Marcação visual tracejada em 70% indicando zona de alerta/ação executiva.
* **Seletor de Modo**:
  - *Modo Câmera/Webcam*: Leitura direta de vídeo com processamento OpenCV/YOLO.
  - *Modo Cenários de Teste*: Injeção controlada de padrões comportamentais (*Passante Rápido*, *Lead Quente Curioso*, *Aglomeração em Demonstração*) para validação imediata da comunicação.

---

## 6. Escalabilidade Futura: Ponte para NVIDIA DeepStream / Metropolis

Embora a aplicação rode perfeitamente no notebook com Python puro (OpenCV + NumPy + Ultralytics), a arquitetura é 100% desacoplada:
* Para escalar para 50 câmeras IP (RTSP) em um pavilhão inteiro de convenções, substitui-se o loop de captura local pelo pipeline **NVIDIA DeepStream (NVDEC + TensorRT + nvdsanalytics)**.
* O contrato do SITREP em JSON e as 4 perguntas do Jev permanecem **rigorosamente idênticos**, garantindo portabilidade total do edge corporativo ao notebook.

---

## 7. Estrutura de Arquivos da Aplicação no Workspace

```
Jev_yolo/
├── .env.example              # OPENROUTER_API_KEY, JEV_MODEL=typesafe/jev-1.13
├── .gitignore                # Protege credenciais e venv
├── requirements.txt          # Dependências: httpx, streamlit, plotly, pandas, pydantic, python-dotenv, opencv-python, numpy, scipy
├── README.md                 # Guia de instalação, configuração do .env e execução
├── PLAN_JEV_YOLO.md          # Este plano detalhado gravado na raiz
├── config.py                 # Validação de credenciais e endpoint da OpenRouter
│
├── yolo_stream.py            # Parser de telemetria do YOLO e simulador de streams da feira
├── sitrep_engine.py          # Motor Espacial: Homografia 2D, distâncias euclidianas, dwell time e aproximação
├── questions_config.py       # As 4 perguntas de probabilidade noul com seus critérios
├── jev_client.py             # Cliente HTTP assíncrono para /api/alpha/decisions na OpenRouter
├── test_connection.py        # Suíte de teste da comunicação YOLO -> Jev (schema, latência e parsing)
├── history_tracker.py        # Ring buffer e séries temporais das 4 probabilidades com filtro EMA
└── dashboard.py              # Interface Streamlit com 4 gráficos Time Series dedicados
```
