"""
Dashboard em Tempo Real de Séries Temporais: JEV + YOLO Analytics (dashboard.py).
Construído com Streamlit e Plotly para exibição local no notebook durante feiras de tecnologia.
Plota a evolução temporal contínua das 4 probabilidades geradas pelo Jev com limiares e alertas.
"""

import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from config import settings
from questions_config import JEV_QUESTIONS, get_question_ids
from yolo_stream import YoloScenarioSimulator
from sitrep_engine import SitrepEngine
from jev_client import JevClient
from history_tracker import HistoryTracker
from notebook_perception import NotebookPerceptionSimulator
from spatial_heatmap import SpatialHeatmapEngine

# Configuração da página Streamlit
st.set_page_config(
    page_title="Jev + YOLO Time Series Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS personalizada para tema executivo escuro
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .badge-alert {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-normal {
        background-color: #064E3B;
        color: #6EE7B7;
    }
    .badge-warning {
        background-color: #78350F;
        color: #FCD34D;
    }
    .badge-danger {
        background-color: #7F1D1D;
        color: #FCA5A5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Inicializa as instâncias duradouras de estado no Streamlit."""
    if "tracker" not in st.session_state:
        st.session_state.tracker = HistoryTracker(
            maxlen=settings.HISTORY_MAX_POINTS, alpha=settings.EMA_ALPHA
        )
    else:
        # Recuperação automática de instâncias legadas mantidas no cache do navegador
        t = st.session_state.tracker
        if not hasattr(t, "_last_filtered") or t._last_filtered is None:
            t._last_filtered = {qid: None for qid in t.question_ids}
        if not hasattr(t, "qei_raw"):
            from collections import deque
            t.qei_raw = deque(maxlen=t.maxlen)
            t.qei_filtered = deque(maxlen=t.maxlen)
            t._last_qei_filt = None

    if "engine" not in st.session_state:
        st.session_state.engine = SitrepEngine()
    if "simulator" not in st.session_state:
        st.session_state.simulator = YoloScenarioSimulator()
    if "nb_simulator" not in st.session_state:
        st.session_state.nb_simulator = NotebookPerceptionSimulator()
    if "heatmap_engine" not in st.session_state:
        st.session_state.heatmap_engine = SpatialHeatmapEngine()
    if "step_count" not in st.session_state:
        st.session_state.step_count = 0
    if "auto_running" not in st.session_state:
        st.session_state.auto_running = False
    if "last_sitrep" not in st.session_state:
        st.session_state.last_sitrep = {}
    if "trigger_step" not in st.session_state:
        st.session_state.trigger_step = False


initialize_session_state()

# ==============================================================================
# Barra Lateral (Controles e Conexão)
# ==============================================================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Reverse.svg",
        width=180,
    )
    st.title("⚙️ Painel de Controle")
    st.caption("Visão Computacional + Decisão Estruturada Jev")

    st.markdown("---")
    st.subheader("🛑 Modo de Execução & Economia de Tokens")

    exec_mode = st.radio(
        "Modo de Operação",
        options=[
            "🎯 Manual por Botão (Econômico - 1 token por clique)",
            "🧪 Simulação Analítica Local (0 Tokens OpenRouter)",
            "⚡ Automático Contínuo (Consome Tokens em Loop)",
        ],
        index=0,
        help="Escolha 'Manual por Botão' para chamar a API apenas quando você clicar, ou 'Simulação Analítica' para testar sem gastar nenhum token.",
    )

    st.markdown("---")
    st.subheader("🌐 Conexão OpenRouter / Jev")

    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=settings.OPENROUTER_API_KEY,
        type="password",
        help="Informe sua chave de API para inferência real no modelo Jev.",
    )

    model_input = st.selectbox(
        "Modelo Jev",
        options=["typesafe/jev-1.13", "typesafe/jev-latest"],
        index=0,
    )

    @st.cache_data(ttl=60)
    def fetch_account_credits(key: str):
        try:
            import httpx
            r = httpx.get(
                "https://openrouter.ai/api/v1/credits",
                headers={"Authorization": f"Bearer {key}"},
                timeout=4.0,
            )
            if r.status_code == 200:
                d = r.json().get("data", {})
                tot = float(d.get("total_credits", 0.0))
                usg = float(d.get("total_usage", 0.0))
                return round(tot - usg, 2)
        except Exception:
            pass
        return None

    if "Simulação" in exec_mode:
        st.info("🧪 Simulação Local Ativa (Zero chamadas à OpenRouter)")
    elif api_key_input and not api_key_input.startswith("your_"):
        st.success("🟢 Chave configurada (Chamada Real à OpenRouter)")
        bal = fetch_account_credits(api_key_input)
        if bal is not None:
            st.caption(f"💰 Saldo Restante na Conta: **${bal:.2f} USD**")
    else:
        st.info("🟡 Sem Chave: Operando via Simulação Analítica")

    st.markdown("---")
    st.subheader("📹 Câmeras da Feira (Dupla Escala)")
    st.caption("Visão 1: Corredor (YOLO-Pose) | Visão 2: Notebook (Frontal)")

    data_source = st.radio(
        "Origem dos Dados",
        options=["Cenários de Teste (Feira)", "Webcam / Câmera Local"],
        index=0,
    )

    selected_scenario = "lead_quente_curioso"
    if data_source == "Cenários de Teste (Feira)":
        selected_scenario = st.selectbox(
            "Câmera 1 (Corredor / YOLO-Pose)",
            options=[
                ("lead_quente_curioso", "1. Lead Quente Curioso (Parado no Totem)"),
                ("passante_rapido", "2. Passante Rápido (Trânsito Contínuo)"),
                ("aglomeracao_demo_pico", "3. Pico de Demonstração (Grupo Focado)"),
                ("risco_respiratorio_elevado", "4. Risco Biossegurança (Aglomeração Fechada)"),
            ],
            format_func=lambda x: x[1],
            index=0,
        )[0]

    selected_nb_scenario = st.selectbox(
        "Câmera 2 (Notebook / Engajamento Facial)",
        options=[
            ("visitante_encantado", "😍 Visitante Encantado (Foco 90%, Sorrindo)"),
            ("visitante_focado_tecnico", "🧐 Focado e Analítico (Testando o Sistema)"),
            ("visitante_distraido", "🥱 Distraído (Olhando ao redor)"),
            ("notebook_livre", "👤 Notebook Livre (Sem ninguém na tela)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    st.markdown("---")
    st.subheader("⚡ Parâmetros do Filtro EMA")

    ema_alpha = st.slider(
        "Fator de Suavização EMA (α)",
        min_value=0.05,
        max_value=1.0,
        value=settings.EMA_ALPHA,
        step=0.05,
        help="Valores menores suavizam ruídos de oclusão; valores maiores respondem instantaneamente.",
    )
    st.session_state.tracker.set_alpha(ema_alpha)

    refresh_interval = 2.0
    if "Automático" in exec_mode:
        refresh_interval = st.slider(
            "Intervalo entre Chamadas (segundos)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=1.0,
        )

    if st.button("🔄 Limpar Histórico de Séries"):
        st.session_state.tracker.clear()
        st.session_state.step_count = 0
        st.rerun()


# ==============================================================================
# Execução do Ciclo de Inferência (Sob Demanda)
# ==============================================================================
# Determina se devemos executar a inferência no ciclo atual
should_infer = False

if "Automático" in exec_mode:
    should_infer = True
elif "Simulação" in exec_mode:
    # Em simulação pura, roda o passo
    should_infer = True
elif st.session_state.trigger_step:
    # Usuário clicou no botão manual
    should_infer = True
    st.session_state.trigger_step = False

# Se estiver em modo de simulação, forçamos chave vazia no cliente para não gastar tokens
active_api_key = "" if "Simulação" in exec_mode else api_key_input
client = JevClient(api_key=active_api_key, model=model_input)

if should_infer:
    st.session_state.step_count += 1
    # 1. Obtenção do Frame de Telemetria do Corredor (YOLO-Pose Macro)
    frame = st.session_state.simulator.generate_frame(
        scenario=selected_scenario, step=st.session_state.step_count
    )

    # 2. Obtenção da Percepção Facial no Notebook (Microvisão)
    nb_data = st.session_state.nb_simulator.generate_perception(
        scenario=selected_nb_scenario, step=st.session_state.step_count
    )

    # 3. Motor Espacial e Cinético Local (SITREP Unificado)
    sitrep = st.session_state.engine.process_frame(frame, notebook_perception=nb_data)
    st.session_state.last_sitrep = sitrep

    # 4. Cálculo do Qualified Engagement Index (QEI - CEIR / RetailNext)
    current_qei = st.session_state.engine.calculate_qei(sitrep)

    # 5. Decisão do Jev
    try:
        probabilities, latency_ms, raw_resp = client.decide_sync(sitrep)
    except Exception as err:
        st.error(f"Erro na inferência do Jev: {err}")
        probabilities = {qid: 0.0 for qid in get_question_ids()}
        latency_ms = 0.0

    # 6. Ring Buffer e Filtro EMA com QEI
    st.session_state.tracker.add_point(probabilities, latency_ms=latency_ms, qei=current_qei)

    # 7. Atualização determinística da Matriz de Calor de Solo (Dwell Heatmap)
    st.session_state.heatmap_engine.update(
        getattr(st.session_state.engine, "last_pedestrians_spatial_state", []),
        dt=0.5,
    )


# ==============================================================================
# Cabeçalho Principal e Indicadores Executivos
# ==============================================================================
st.title("🎯 JEV + YOLO Time Series Analytics")
st.markdown(
    "Monitoramento contínuo em tempo real de **probabilidades de decisão comportamental** via IA Simbólica e Física Espacial."
)

# Barra de Ação Manual e Economia de Tokens
col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
with col_act1:
    if "Manual" in exec_mode:
        if st.button("⚡ Executar Inferência no Jev (Consome 1 Requisição)", type="primary", use_container_width=True):
            st.session_state.trigger_step = True
            st.rerun()
    elif "Simulação" in exec_mode:
        st.info("🧪 **Modo Simulação Local**: Zero tokens consumidos da OpenRouter.")
    else:
        st.warning("⚠️ **Modo Automático**: Chamando Jev em loop a cada ciclo.")

with col_act2:
    if st.button("➕ Avançar 1 Frame Local (0 Tokens)", use_container_width=True, help="Calcula a física e projeta probabilidades analiticamente sem chamar a OpenRouter."):
        st.session_state.step_count += 1
        f = st.session_state.simulator.generate_frame(scenario=selected_scenario, step=st.session_state.step_count)
        nb_f = st.session_state.nb_simulator.generate_perception(scenario=selected_nb_scenario, step=st.session_state.step_count)
        s = st.session_state.engine.process_frame(f, notebook_perception=nb_f)
        st.session_state.last_sitrep = s
        local_qei = st.session_state.engine.calculate_qei(s)
        p = client._simulate_smart_fallback(s)
        st.session_state.tracker.add_point(p, latency_ms=15.0, qei=local_qei)
        st.session_state.heatmap_engine.update(
            getattr(st.session_state.engine, "last_pedestrians_spatial_state", []),
            dt=0.5,
        )
        st.rerun()

with col_act3:
    if st.button("🗑️ Limpar Gráficos", use_container_width=True):
        st.session_state.tracker.clear()
        st.session_state.heatmap_engine.reset()
        st.session_state.step_count = 0
        st.rerun()

latest_values = st.session_state.tracker.get_latest_values()
latest_latency = st.session_state.tracker.get_latest_latency()
current_metrics = st.session_state.last_sitrep.get("metrics", {})
nb_metrics = st.session_state.last_sitrep.get("notebook_interaction", {})

# Alerta Executivo de Auditoria da Feira: Picos de Engajamento e Retenção de Público
p_buy = latest_values.get("p_purchase_intent", {}).get("filtered", 0.0)
p_ret = latest_values.get("p_stand_retention_engagement", {}).get("filtered", 0.0)
p_sat = latest_values.get("p_user_satisfaction_engagement", {}).get("filtered", 0.0)

if p_buy >= 0.60 or p_ret >= 0.60 or p_sat >= 0.60:
    st.success(
        f"🎯 **ALTO ENGAJAMENTO AUDITADO NO ESTANDE:** Visitante absorvido pela demonstração "
        f"(Intenção Comercial: {p_buy*100:.0f}% | Retenção Física: {p_ret*100:.0f}% | Satisfação na Tela: {p_sat*100:.0f}%)."
    )


def create_time_series_figure(qid: str, df: pd.DataFrame) -> go.Figure:
    """Gera o gráfico Plotly escuro e executivo para a probabilidade especificada."""
    q_meta = JEV_QUESTIONS[qid]
    fig = go.Figure()

    if not df.empty and len(df) > 0:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[f"{qid}_raw"] * 100.0,
                mode="markers",
                name="Inferência Bruta Jev",
                marker=dict(color=q_meta["color"], size=5, opacity=0.35),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[f"{qid}_filt"] * 100.0,
                mode="lines+markers",
                name="Série Suavizada (EMA)",
                line=dict(color=q_meta["color"], width=3),
                marker=dict(size=6),
            )
        )
        fig.add_hline(
            y=q_meta["threshold"] * 100.0,
            line_dash="dash",
            line_color="#F87171",
            annotation_text=f"Limiar de Ação ({int(q_meta['threshold'] * 100)}%)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#F87171"),
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{q_meta['display_name']}</b>",
            font=dict(size=13, color="#F8FAFC"),
        ),
        xaxis=dict(title="Horário", showgrid=True, gridcolor="#334155", tickfont=dict(size=10, color="#94A3B8")),
        yaxis=dict(title="Probabilidade (%)", range=[0, 105], showgrid=True, gridcolor="#334155", tickfont=dict(size=10, color="#94A3B8")),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=35, r=20, t=35, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9, color="#94A3B8")),
        height=300,
    )
    return fig


def create_qei_time_series_figure(df: pd.DataFrame) -> go.Figure:
    """Gera o gráfico de Série Temporal da métrica padrão QEI (CEIR / RetailNext)."""
    fig = go.Figure()
    if not df.empty and "qei_filt" in df.columns and len(df) > 0:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["qei_raw"],
                mode="markers",
                name="QEI Bruto",
                marker=dict(color="#F59E0B", size=5, opacity=0.35),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["qei_filt"],
                mode="lines+markers",
                name="QEI Suavizado (EMA)",
                line=dict(color="#F59E0B", width=3),
                marker=dict(size=6),
            )
        )
        fig.add_hline(
            y=50.0,
            line_dash="dash",
            line_color="#10B981",
            annotation_text="Alta Performance (50 pts)",
            annotation_position="top right",
            annotation_font=dict(size=9, color="#10B981"),
        )
        fig.add_hline(
            y=25.0,
            line_dash="dot",
            line_color="#FCD34D",
            annotation_text="Média de Feira (25 pts)",
            annotation_position="bottom right",
            annotation_font=dict(size=9, color="#FCD34D"),
        )

    fig.update_layout(
        title=dict(
            text="<b>Qualified Engagement Index (QEI) — CEIR / RetailNext</b>",
            font=dict(size=13, color="#F59E0B"),
        ),
        xaxis=dict(title="Horário", showgrid=True, gridcolor="#334155", tickfont=dict(size=10, color="#94A3B8")),
        yaxis=dict(title="Índice QEI (0-100 pts)", range=[0, 105], showgrid=True, gridcolor="#334155", tickfont=dict(size=10, color="#94A3B8")),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=35, r=20, t=35, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9, color="#94A3B8")),
        height=300,
    )
    return fig


tab_series, tab_heatmap = st.tabs([
    "📈 Séries Temporais & QEI (Jev + Visão Dual)",
    "🗺️ Planta Baixa 2D & Mapa de Calor (Visão Espacial)",
])

with tab_series:
    # Banner Executivo: Métrica Padrão Internacional QEI (CEIR / RetailNext)
    latest_qei = st.session_state.tracker.get_latest_qei()
    qei_val = latest_qei["filtered"]
    qei_delta = latest_qei["delta"]
    qei_badge_color = "#064E3B" if qei_val >= 50 else ("#78350F" if qei_val >= 25 else "#7F1D1D")
    qei_badge_label = "🟢 ALTA PERFORMANCE" if qei_val >= 50 else ("🟡 MÉDIA DE FEIRA" if qei_val >= 25 else "🔴 SUB-UTILIZADO")

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 2px solid #F59E0B; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.8rem; color: #F59E0B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                        🏆 Métrica Padrão da Indústria: Qualified Engagement Index (QEI) — Padrão CEIR / RetailNext
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">
                        {qei_val:.1f} <span style="font-size: 1.0rem; color: #94A3B8; font-weight: 400;">/ 100 pts</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
                        Equação Oficial: <i>(Capture Rate × Demo Trial Rate × Gaze & Sentiment Quality) × 100</i> | Variação: <b>{qei_delta:+.1f} pts</b>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: {qei_badge_color}; color: #F8FAFC; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                        {qei_badge_label}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Linha de 5 Cartões KPI com as Probabilidades em Tempo Real
    q_ids = get_question_ids()
    kpi_cols = st.columns(len(q_ids))

    for idx, qid in enumerate(q_ids):
        q_meta = JEV_QUESTIONS[qid]
        val_data = latest_values.get(qid, {"filtered": 0.0, "raw": 0.0, "delta": 0.0})
        val_pct = val_data["filtered"] * 100.0
        delta_pct = val_data["delta"] * 100.0
        is_alert = val_data["filtered"] >= q_meta["threshold"]

        with kpi_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 5px solid {q_meta['color']};">
                    <div class="metric-title">{q_meta['display_name']}</div>
                    <div class="metric-value" style="color: {q_meta['color']}; font-size: 1.5rem;">{val_pct:.1f}%</div>
                    <div style="font-size: 0.75rem; margin-top: 4px;">
                        <span class="{'badge-danger' if is_alert else 'badge-normal'} badge-alert">
                            {'⚠️ ALERTA ACIMA DO LIMIAR' if is_alert else '✓ ESTÁVEL / NOMINAL'}
                        </span>
                        <span style="color: #94A3B8; margin-left: 6px;">Var: {delta_pct:+.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Painel de Telemetria Física (SITREP) e Latência
    with st.expander("📍 Telemetria Física de Dupla Escala (Corredor + Notebook)", expanded=False):
        st.markdown("**Câmera 1: Corredor da Feira (YOLOv8-Pose)**")
        t_col1, t_col2, t_col3, t_col4, t_col5, t_col6 = st.columns(6)
        t_col1.metric("Público Total", current_metrics.get("active_visitors_count", 0))
        t_col2.metric("Fluxo Corredor", f"{current_metrics.get('corridor_passersby_count', 0)} pessoas")
        t_col3.metric("Retidos no Estande", f"{current_metrics.get('booth_visitors_retained_count', 0)} pessoas")
        t_col4.metric("Dwell Time Máx.", f"{current_metrics.get('max_dwell_time_seconds', 0.0)} s")
        t_col5.metric("Alinhamento do Olhar", f"{current_metrics.get('gaze_alignment_with_demo_pct', 0.0)} %")
        t_col6.metric("Latência RTT Jev", f"{latest_latency:.0f} ms")

        st.markdown("**Câmera 2: Câmera Frontal do Notebook**")
        nb_col1, nb_col2, nb_col3, nb_col4 = st.columns(4)
        nb_col1.metric("Usuário Presente na Tela", "Sim" if nb_metrics.get("user_present", False) else "Não")
        nb_col2.metric("Contato Visual com Tela", f"{nb_metrics.get('screen_gaze_contact_pct', 0.0)} %")
        nb_col3.metric("Score de Satisfação Facial", f"{nb_metrics.get('satisfaction_score_pct', 0.0)} %")
        nb_col4.metric("Distância da Tela", f"{nb_metrics.get('face_distance_cm', 0.0)} cm")

    # ==============================================================================
    # Os 6 Gráficos de Séries Temporais (Plotly em Grade 2x3)
    # ==============================================================================
    df_history = st.session_state.tracker.to_dataframe()

    chart_r1_c1, chart_r1_c2, chart_r1_c3 = st.columns(3)
    chart_r2_c1, chart_r2_c2, chart_r2_c3 = st.columns(3)

    grid_positions = [
        (chart_r1_c1, "p_purchase_intent"),
        (chart_r1_c2, "p_stand_retention_engagement"),
        (chart_r1_c3, "p_user_satisfaction_engagement"),
        (chart_r2_c1, "p_demo_attractiveness_peak"),
        (chart_r2_c2, "p_airborne_transmission_risk"),
    ]

    for col_container, qid in grid_positions:
        with col_container:
            fig = create_time_series_figure(qid, df_history)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{qid}")

    with chart_r2_c3:
        fig_qei = create_qei_time_series_figure(df_history)
        st.plotly_chart(fig_qei, use_container_width=True, key="chart_qei_ceir")

    # Alertas Ativos no Rodapé
    active_alerts = []
    for qid in get_question_ids():
        val = latest_values.get(qid, {}).get("filtered", 0.0)
        if val >= JEV_QUESTIONS[qid]["threshold"]:
            active_alerts.append(f"**{JEV_QUESTIONS[qid]['display_name']}**: {JEV_QUESTIONS[qid]['alert_text']}")

    if active_alerts:
        st.error("🚨 **AÇÕES RECOMENDADAS PELO SISTEMA:**\n\n" + "\n\n".join(active_alerts))

with tab_heatmap:
    st.markdown("### 🗺️ Planta Baixa 2D do Estande & Mapa de Calor de Retenção (*Dwell Heatmap*)")
    st.caption("Mapeamento determinístico em tempo real do corredor e estande em coordenadas métricas reais. Cores quentes indicam áreas onde os pedestres desaceleram e permanecem retidos no totem/notebook. Zero tokens consumidos da API.")

    col_h_info, col_h_btn = st.columns([5, 1])
    with col_h_btn:
        if st.button("🔥 Zerar Mapa Térmico", use_container_width=True, help="Reinicia a matriz térmica para testar um novo arranjo físico ou horário."):
            st.session_state.heatmap_engine.reset()
            st.rerun()

    # Planta Baixa em Tela Cheia
    fig_floorplan = st.session_state.heatmap_engine.create_floorplan_figure()
    st.plotly_chart(fig_floorplan, use_container_width=True, key="floorplan_map_full")

    # Métricas de Análise Espacial
    peds = getattr(st.session_state.engine, "last_pedestrians_spatial_state", [])
    retained_peds = [p for p in peds if p.get("dwell_s", 0.0) >= 3.0]
    fast_peds = [p for p in peds if p.get("velocity_mps", 0.0) >= 0.8]
    max_d = max([p.get("dwell_s", 0.0) for p in peds] + [0.0])

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Público Total Mapeado", len(peds))
    m_col2.metric("Visitantes Retidos no Totem", f"{len(retained_peds)} pessoas", delta="Zona Quente" if retained_peds else None)
    m_col3.metric("Fluxo Rápido no Corredor", f"{len(fast_peds)} pessoas")
    m_col4.metric("Dwell Time Máximo Registrado", f"{max_d:.1f} s")


# Loop de auto-refresh no Streamlit (APENAS se explicitamente selecionado Automático ou Simulação)
if "Automático" in exec_mode or "Simulação" in exec_mode:
    time.sleep(refresh_interval)
    st.rerun()
