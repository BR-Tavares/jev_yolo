# Metodologia de Planta Baixa 2D e Mapa de Calor de Retenção Espacial (Dwell Heatmap)

Esta metodologia documenta a arquitetura, formulação matemática e engenharia de software desenvolvida para mapear espacialmente o comportamento de público em corredores e estandes de feiras de negócios.

O método é **100% determinístico, de alta resolução espacial e roda localmente sem consumo de tokens de modelos de linguagem ou APIs pagas**.

---

## 1. Princípios Norteadores da Arquitetura

1. **Separação de Camadas (Visão Local vs. Decisão Cognitiva)**:
   - A visão computacional, homografia de solo e acúmulo de mapa térmico ocorrem inteiramente na máquina local (CPU ou GPU).
   - O modelo de decisão (Jev / TypeSafe AI) recebe apenas o **SITREP semântico agregado** (SITREP JSON enxuto), eliminando redundância e protegendo o saldo de tokens.
2. **Diferenciação Estrita: Trânsito vs. Retenção (*Dwell*)**:
   - Um mapa de calor tradicional de contagem apenas mede fluxo de passagem.
   - O **Dwell Heatmap** pondera o tempo que a pessoa permanece com velocidade nula ou reduzida em frente a um ativo (totem, tela interativa, mostruário).
3. **Agrupamento em Dupla Escala**:
   - **Macrovisão (Corredor)**: Câmera angular cobrindo o corredor geral e a fachada.
   - **Microvisão (Notebook/Totem)**: Câmera frontal medindo contato visual direto, expressão e distância de operação.

---

## 2. Pipeline Matemático e Algoritmo

```
[ Frame de Vídeo / Câmera ]
            │
            ▼
[ Detector de Pessoas / Poses (YOLO / RT-DETR) ]
            │
            ▼ Coordenadas dos pés no plano de imagem (u, v) [px]
[ Homografia Projetiva 2D (Matriz H 3x3) ]
            │
            ▼ Coordenadas no plano métrico de solo (X, Y) [metros]
[ Rastreador Cinético (Tracking ID + Velocidade + Dwell Time) ]
            │
            ▼
[ Matriz de Solo Discreta (Nx × Ny células de 10 cm) ]
  ├── 1. Decaimento Temporal Exponencial (Resfriamento)
  ├── 2. Injeção de Calor Ponderada por Velocidade e Dwell Time
  └── 3. Convolução Gaussiana 2D (Suavização Orgânica)
            │
            ▼
[ Renderização Arquitetural Plotly / Dashboard ]
```

---

## 3. Formulação Matemática

### 3.1. Projeção Planar por Homografia ($3 	imes 3$)
Para transformar os pixels da câmera de perspectiva em metros planos no chão do estande:

$$egin{bmatrix} x' \ y' \ w \end{bmatrix} = \mathbf{H} egin{bmatrix} u \ v \ 1 \end{bmatrix}, \quad X_w = rac{x'}{w}, \quad Y_w = rac{y'}{w}$$

Onde $\mathbf{H}$ é calibrada via 4 pontos correspondentes conhecidos entre a imagem e o piso da feira.

### 3.2. Dinâmica de Acúmulo Térmico de Solo

A cada passo de tempo $\Delta t$:

1. **Evaporação / Resfriamento Contínuo**:
   $$\mathbf{M}_{t}(x, y) = \mathbf{M}_{t-1}(x, y) 	imes \max(0, 1 - \lambda \cdot \Delta t)$$
   *Taxa padrão adotada:* $\lambda = 0.015	ext{ s}^{-1}$ (garante que após a pessoa sair, a mancha esfria suavemente em ~45 a 60 segundos).

2. **Injeção de Calor por Pedestre**:
   Para cada pedestre $i$ com velocidade $v_i$ e tempo de permanência $t_{	ext{dwell}, i}$:
   $$W_i = egin{cases}
   (1.5 + 0.15 	imes \min(t_{	ext{dwell}, i}, 40.0)) \cdot \Delta t & 	ext{se } v_i < 0.30	ext{ m/s (Retenção / Parado)} \
   0.60 \cdot \Delta t & 	ext{se } 0.30 \le v_i < 0.80	ext{ m/s (Curioso / Desacelerando)} \
   0.15 \cdot \Delta t & 	ext{se } v_i \ge 0.80	ext{ m/s (Passante rápido)}
   \end{cases}$$

3. **Convolução Gaussiana Contínua**:
   Para transformar a grade discreta em superfícies térmicas suaves e legíveis:
   $$\mathbf{S} = \mathbf{M} * \mathbf{K}_{\sigma}, \quad 	ext{onde } \sigma = 1.8 	ext{ células (18 cm)}$$

---

## 4. Guia de Replicação para Outros Modelos

Para replicar este componente com outro detector visual ou framework:

### 4.1. Entrada Mínima Necessária do Detector
Qualquer modelo (YOLOv11, YOLOv8, MediaPipe, ByteTrack, RT-DETR) deve fornecer para cada pedestre ativo no frame:
1. `track_id`: Inteiro identificador único persistente entre frames.
2. `foot_coords`: Ponto $(u, v)$ no plano de imagem (ponto central inferior da bounding box `[x_center, y_max]`).
3. `head_yaw_deg` (opcional mas recomendado): Ângulo de rotação da cabeça em graus para traçar a seta do vetor de atenção visual.

### 4.2. Classe de Interface Padrão em Python
```python
from spatial_heatmap import SpatialHeatmapEngine

# 1. Instanciação única no início da aplicação
heatmap = SpatialHeatmapEngine(
    width_m=8.0,        # Largura total observada (metros)
    depth_m=6.0,        # Profundidade (metros)
    resolution_m=0.10,  # Resolução da grade (10 cm por célula)
    decay_rate=0.015,   # Resfriamento gradual
)

# 2. Em cada frame de vídeo ou tick temporal:
pedestrians_data = [
    {
        "track_id": 101,
        "pos_m": (0.2, 2.3),      # Coordenadas em metros calculadas pela homografia
        "velocity_mps": 0.05,     # Velocidade em m/s
        "dwell_s": 35.0,          # Segundos parado
        "yaw_deg": 12.0,          # Rotação da cabeça
        "is_sales_rep": False,
    }
]

heatmap.update(pedestrians_data, dt=0.5)

# 3. Obtenção do gráfico para Streamlit, Dash ou Web UI:
fig = heatmap.create_floorplan_figure()
```

---

## 5. Aplicações Práticas em Feiras de Negócios

1. **Testes A/B de Layout do Estande**:
   - Compare a mancha de calor no primeiro dia (totem centralizado) com o segundo dia (totem na lateral).
   - O gráfico comprova matematicamente qual arranjo atrai e retém mais leads qualificados.
2. **Auditoria de Oportunidades Perdidas**:
   - Cruza a zona de calor do totem com a posição do vendedor: se a mancha estiver vermelha e o vendedor distante, dispara alerta em tempo real.
3. **Cálculo do QEI (Qualified Engagement Index)**:
   - Alimenta o fator de experimentação e captura do índice padrão da indústria (CEIR / RetailNext).
