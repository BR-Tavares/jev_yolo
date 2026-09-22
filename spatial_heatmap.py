"""
Motor de Planta Baixa 2D e Mapa de Calor de Retenção (spatial_heatmap.py).
Implementa a matriz determinística de solo, acumulação ponderada por Dwell Time,
decaimento temporal contínuo e renderização da planta baixa em Plotly.
Zero chamadas de API / Zero consumo de tokens.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from config import settings


class SpatialHeatmapEngine:
    """
    Mantém uma matriz 2D contínua do plano de solo da feira.
    Mapeia onde os visitantes param em frente ao totem/notebook
    e diferencia passantes rápidos de pessoas retidas.
    """

    def __init__(
        self,
        width_m: float = 8.0,
        depth_m: float = 6.0,
        resolution_m: float = 0.10,
        decay_rate: float = 0.015,
        gaussian_sigma: float = 1.8,
    ):
        self.width_m = width_m
        self.depth_m = depth_m
        self.resolution_m = resolution_m
        self.decay_rate = decay_rate
        self.gaussian_sigma = gaussian_sigma

        self.nx = int(math.ceil(width_m / resolution_m))
        self.ny = int(math.ceil(depth_m / resolution_m))

        self.heatmap_matrix = np.zeros((self.ny, self.nx), dtype=np.float32)

        self.x_coords = np.linspace(-width_m / 2.0, width_m / 2.0, self.nx)
        self.y_coords = np.linspace(0.0, depth_m, self.ny)

        self.recent_positions: List[Dict[str, Any]] = []

    def reset(self):
        """Zera o mapa de calor para iniciar nova medição de arranjo ou horário."""
        self.heatmap_matrix.fill(0.0)
        self.recent_positions.clear()

    def update(
        self,
        pedestrians: List[Dict[str, Any]],
        dt: float = 0.5,
    ):
        """
        Atualiza a matriz de calor de solo com as posições dos pedestres.
        """
        decay_factor = max(0.0, 1.0 - (self.decay_rate * dt))
        self.heatmap_matrix *= decay_factor

        self.recent_positions = pedestrians

        half_w = self.width_m / 2.0

        for ped in pedestrians:
            pos = ped.get("pos_m", (0.0, 0.0))
            px, py = pos[0], pos[1]
            vel = ped.get("velocity_mps", 1.0)
            dwell = ped.get("dwell_s", 0.0)

            gx = int((px + half_w) / self.resolution_m)
            gy = int(py / self.resolution_m)

            if 0 <= gx < self.nx and 0 <= gy < self.ny:
                # Se estiver no corredor público (py < 2.0m) e em trânsito, quase não dissipa calor
                if py < 2.0:
                    if vel < 0.6:
                        weight = 0.3 * dt  # Transeunte reduzindo o passo para olhar
                    else:
                        weight = 0.02 * dt  # Passante fluido não gera calor de retenção
                else:
                    # No interior do estande: aquecimento forte proporcional ao dwell time na bancada
                    if dwell >= 1.0 or vel < settings.ENGAGEMENT_VELOCITY_THRESHOLD_MPS:
                        weight = (2.0 + min(dwell, 40.0) * 0.25) * dt
                    else:
                        weight = 0.4 * dt

                self.heatmap_matrix[gy, gx] += weight

        self.heatmap_matrix = np.clip(self.heatmap_matrix, 0.0, 100.0)

    def get_smoothed_heatmap(self) -> np.ndarray:
        """Aplica convolução gaussiana para gerar gradientes térmicos suaves."""
        if np.max(self.heatmap_matrix) < 1e-4:
            return self.heatmap_matrix
        return gaussian_filter(self.heatmap_matrix, sigma=self.gaussian_sigma)

    def create_floorplan_figure(self) -> go.Figure:
        """
        Renderiza a planta baixa arquitetural 2D do estande sobreposta com o mapa térmico de solo.
        """
        fig = go.Figure()
        half_w = self.width_m / 2.0

        # 1. Mapa de Calor (Gradiente Térmico Contínuo)
        smooth_heat = self.get_smoothed_heatmap()
        x_coords = np.linspace(-half_w, half_w, self.nx)
        y_coords = np.linspace(0.0, self.depth_m, self.ny)

        fig.add_trace(
            go.Heatmap(
                x=x_coords,
                y=y_coords,
                z=smooth_heat,
                colorscale="Hot",
                reversescale=True,
                zmin=0.0,
                zmax=max(5.0, float(np.max(smooth_heat))),
                opacity=0.68,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Retenção (Dwell)", font=dict(color="#F8FAFC", size=11)),
                    tickfont=dict(color="#94A3B8"),
                    len=0.75,
                ),
                hoverinfo="none",
            )
        )

        # 2. Desenho Arquitetural do Estande e Corredor
        # Perímetro do estande (2m a 6.2m)
        fig.add_shape(
            type="rect",
            x0=-half_w, y0=2.0, x1=half_w, y1=self.depth_m,
            line=dict(color="#475569", width=2),
            fillcolor="rgba(241, 245, 249, 0.06)",
        )

        # Linha da Fachada / Entrada do Estande (aberta em Y=2.0m)
        fig.add_shape(
            type="line",
            x0=-half_w, y0=2.0, x1=half_w, y1=2.0,
            line=dict(color="#94A3B8", width=2.5, dash="dash"),
        )

        # Corredor público da feira (0m a 2m)
        fig.add_shape(
            type="rect",
            x0=-half_w, y0=0.0, x1=half_w, y1=2.0,
            fillcolor="rgba(30, 41, 59, 0.35)",
            line=dict(width=0),
            layer="below",
        )

        # Bancada de Demonstração com Notebook no Interior do Estande (Y=4.2m)
        bancada_x, bancada_y = settings.BOOTH_DISPLAY_POS_M
        bancada_w = 2.2
        bancada_d = 0.9
        fig.add_shape(
            type="rect",
            x0=bancada_x - bancada_w / 2, y0=bancada_y - bancada_d / 2,
            x1=bancada_x + bancada_w / 2, y1=bancada_y + bancada_d / 2,
            fillcolor="#1E293B",
            line=dict(color="#38BDF8", width=2.5),
        )

        fig.add_annotation(
            x=0.0, y=0.5,
            text="🚶 FLUXO PÚBLICO DO CORREDOR DA FEIRA",
            showarrow=False,
            font=dict(size=11, color="#64748B"),
        )
        fig.add_annotation(
            x=-half_w + 1.4, y=2.25,
            text="🚪 ENTRADA LIVRE DO ESTANDE",
            showarrow=False,
            font=dict(size=10, color="#94A3B8"),
        )
        fig.add_annotation(
            x=bancada_x, y=bancada_y,
            text="💻 BANCADA DE DEMONSTRAÇÃO (NOTEBOOK)",
            showarrow=False,
            font=dict(size=11, color="#38BDF8"),
        )
        fig.add_annotation(
            x=0.0, y=5.6,
            text="ÁREA INTERNA DO ESTANDE",
            showarrow=False,
            font=dict(size=11, color="#475569"),
        )

        # 3. Pedestres com Classificação Semântica Estrita por Zona
        for ped in self.recent_positions:
            pos = ped.get("pos_m", (0.0, 0.0))
            px, py = pos[0], pos[1]
            vel = ped.get("velocity_mps", 1.2)
            dwell = ped.get("dwell_s", 0.0)
            yaw = ped.get("yaw_deg", 0.0)

            # Classificação semântica:
            if py < 2.0:
                # CORREDOR: Transeuntes públicos da feira
                # NUNCA podem ser vermelhos ou rotulados como "Lead Retido"!
                if vel < 0.6:
                    p_color = "#F59E0B"  # Amarelo (Curioso desacelerando)
                    p_symbol = "circle"
                    p_size = 12
                    label = f"Curioso ({vel:.1f} m/s)"
                else:
                    p_color = "#10B981"  # Verde (Passante em trânsito)
                    p_symbol = "circle"
                    p_size = 11
                    label = f"Passante ({vel:.1f} m/s)"
            else:
                # DENTRO DO ESTANDE:
                if dwell >= 1.0 or vel < settings.ENGAGEMENT_VELOCITY_THRESHOLD_MPS:
                    p_color = "#EF4444"  # Vermelho (Lead Retido na Bancada do Notebook)
                    p_symbol = "circle"
                    p_size = 16
                    label = f"Lead Retido: {dwell:.0f}s"
                else:
                    p_color = "#3B82F6"  # Azul (Visitante circulando pelo estande)
                    p_symbol = "circle"
                    p_size = 13
                    label = f"Visitante ({vel:.1f} m/s)"

            fig.add_trace(
                go.Scatter(
                    x=[px],
                    y=[py],
                    mode="markers+text",
                    marker=dict(size=p_size, color=p_color, symbol=p_symbol, line=dict(color="#F8FAFC", width=1.5)),
                    text=[label],
                    textposition="top center",
                    textfont=dict(size=10, color="#F8FAFC"),
                    hoverinfo="text",
                    showlegend=False,
                )
            )

            yaw_rad = math.radians(yaw)
            arrow_len = 0.55
            dx = arrow_len * math.sin(yaw_rad)
            dy = arrow_len * math.cos(yaw_rad)

            fig.add_annotation(
                x=px + dx,
                y=py + dy,
                ax=px,
                ay=py,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=p_color,
            )

        fig.update_layout(
            title=dict(
                text="<b>Planta Baixa 2D do Estande & Mapa de Calor de Retenção (Dwell Heatmap)</b>",
                font=dict(size=16, color="#F8FAFC"),
                x=0.02,
                y=0.98,
            ),
            xaxis=dict(
                title="Largura da Fachada (Metros)",
                range=[-half_w, half_w],
                showgrid=True,
                gridcolor="#334155",
                zerolinecolor="#475569",
                tickfont=dict(color="#94A3B8"),
            ),
            yaxis=dict(
                title="Profundidade do Estande (Metros)",
                range=[-0.2, self.depth_m],
                showgrid=True,
                gridcolor="#334155",
                zerolinecolor="#475569",
                tickfont=dict(color="#94A3B8"),
                scaleanchor="x",
                scaleratio=1,
            ),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            margin=dict(l=40, r=40, t=50, b=40),
            height=660,
        )
        return fig
