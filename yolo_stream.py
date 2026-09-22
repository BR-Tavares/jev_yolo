"""
Módulo de Telemetria e Stream do YOLO (yolo_stream.py).
Define os contratos de dados de visão computacional (bounding boxes, pés, cabeça, rastreamento)
e provê geradores sintéticos de alta fidelidade para os cenários da Feira de Tecnologia,
além de suporte a webcam local.
"""

import time
import math
import random
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Representa a detecção de uma pessoa com atributos espaciais e esqueleto de pose por frame."""
    track_id: int
    class_name: str = "person"
    confidence: float
    bbox_xyxy: List[int] = Field(..., description="[x1, y1, x2, y2] no plano da imagem")
    center_foot_px: List[int] = Field(..., description="[x, y] ponto de contato com o solo")
    head_bbox_xyxy: List[int] = Field(..., description="[x1, y1, x2, y2] caixa da cabeça")
    # Atributos de YOLOv8-Pose (17 Keypoints anatômicos COCO)
    keypoints_17: Optional[List[List[float]]] = Field(
        default=None,
        description="17 pontos [x, y, conf]: 0-nariz, 1/2-olhos, 3/4-orelhas, 5/6-ombros...",
    )
    head_yaw_deg: float = Field(
        default=0.0,
        description="Ângulo de rotação da cabeça em graus (0° = olhando direto para o estande; >45° = olhando para o corredor)",
    )


class YoloFrame(BaseModel):
    """Contrato da telemetria bruta emitida a cada frame do detector."""
    frame_id: int
    timestamp_ms: int
    detections: List[Detection]


class YoloScenarioSimulator:
    """
    Simulador determinístico de telemetria YOLO para testes e demonstrações em feiras.
    Emite coordenadas e comportamentos cinéticos coerentes com cada padrão comportamental.
    """

    def __init__(self):
        self._frame_count = 0
        self._start_time = time.time()

    def generate_frame(self, scenario: str = "lead_quente_curioso", step: int = 0) -> YoloFrame:
        """
        Gera um frame contendo detecções de acordo com o cenário selecionado:
        - passante_rapido: transeunte atravessando o campo visual a ~1.2 m/s sem parar.
        - lead_quente_curioso: visitante se aproxima do mostruário, desacelera e foca na tela.
        - aglomeracao_demo_pico: múltiplos visitantes reunidos e concentrados na apresentação.
        - risco_respiratorio_elevado: alta densidade de pessoas estáticas e próximas em ar fechado.
        """
        self._frame_count += 1
        now_ms = int(time.time() * 1000)

        detections: List[Detection] = []

        # ----------------------------------------------------------------------
        # 1. População de Fundo do Corredor da Feira (14 a 18 Pedestres em Trânsito)
        # Fluxo contínuo bidirecional: pessoas andando pela passarela pública da feira
        # ----------------------------------------------------------------------
        corridor_crowd_size = 14
        for i in range(corridor_crowd_size):
            p_id = 50 + i
            # Direção de caminhada: pares vão para a direita (+X), ímpares para a esquerda (-X)
            direction = 1 if (i % 2 == 0) else -1
            base_speed = 18.0 + (i % 5) * 3.5  # pixels por frame
            
            # Posição horizontal contínua com wrap-around periódico no corredor
            cycle_width = 1100
            x_raw = (i * 75 + step * base_speed * direction) % cycle_width
            x_pos = int(50 + x_raw)
            # Faixa de profundidade do corredor (Y entre 460 e 540 px)
            y_pos = int(470 + (i % 4) * 20 + math.sin(step * 0.1 + i) * 5)
            
            yaw_dir = 82.0 if direction > 0 else -82.0

            detections.append(
                Detection(
                    track_id=p_id,
                    confidence=0.88 + (i % 10) * 0.01,
                    bbox_xyxy=[x_pos - 35, y_pos - 280, x_pos + 35, y_pos],
                    center_foot_px=[x_pos, y_pos],
                    head_bbox_xyxy=[x_pos - 18, y_pos - 280, x_pos + 18, y_pos - 225],
                    head_yaw_deg=yaw_dir,
                )
            )

        # ----------------------------------------------------------------------
        # 2. Interações Específicas do Estande (Sem Vendedor - Foco no Visitante)
        # ----------------------------------------------------------------------
        if scenario == "passante_rapido":
            # Apenas o trânsito normal e fluido do corredor, ninguém parando na fachada
            pass

        elif scenario == "lead_quente_curioso":
            # Visitante que rompeu o fluxo do corredor e parou no totem interativo
            noise_x = int(math.sin(step * 0.3) * 3)
            noise_y = int(math.cos(step * 0.2) * 2)
            base_x = 520 + noise_x
            base_y = 660 + noise_y

            # Cabeça e corpo diretamente alinhados com o totem (yaw ~ 4° - foco na tela)
            detections.append(
                Detection(
                    track_id=205,
                    confidence=0.97,
                    bbox_xyxy=[base_x - 55, base_y - 390, base_x + 55, base_y],
                    center_foot_px=[base_x, base_y],
                    head_bbox_xyxy=[base_x - 22, base_y - 390, base_x + 22, base_y - 320],
                    head_yaw_deg=4.0,  # Olhar direto para a demonstração
                )
            )

        elif scenario == "aglomeracao_demo_pico":
            # Grupo de 5 visitantes em leque retidos simultaneamente em frente à tela
            anchor_points = [
                (450, 640, -12.0),
                (520, 670, 2.0),
                (590, 650, 14.0),
                (400, 610, -22.0),
                (640, 620, 20.0),
            ]
            for idx, (px, py, yaw) in enumerate(anchor_points):
                jitter_x = int(math.sin(step * 0.2 + idx) * 3)
                jitter_y = int(math.cos(step * 0.2 + idx) * 2)
                x = px + jitter_x
                y = py + jitter_y
                detections.append(
                    Detection(
                        track_id=300 + idx,
                        confidence=0.92,
                        bbox_xyxy=[x - 50, y - 360, x + 50, y],
                        center_foot_px=[x, y],
                        head_bbox_xyxy=[x - 20, y - 360, x + 20, y - 295],
                        head_yaw_deg=yaw,
                    )
                )

        elif scenario == "risco_respiratorio_elevado":
            # Aglomeração pesada e estagnada de público
            base_cluster = [
                (480, 630), (510, 645), (460, 660), (535, 620),
                (550, 670), (430, 635), (500, 690), (470, 600),
            ]
            for idx, (px, py) in enumerate(base_cluster):
                jitter_x = int(math.sin(step * 0.1 + idx) * 2)
                jitter_y = int(math.cos(step * 0.1 + idx) * 2)
                x = px + jitter_x
                y = py + jitter_y
                detections.append(
                    Detection(
                        track_id=400 + idx,
                        confidence=0.90,
                        bbox_xyxy=[x - 45, y - 340, x + 45, y],
                        center_foot_px=[x, y],
                        head_bbox_xyxy=[x - 18, y - 340, x + 18, y - 280],
                        head_yaw_deg=random.uniform(-30.0, 30.0),
                    )
                )

        return YoloFrame(
            frame_id=self._frame_count,
            timestamp_ms=now_ms,
            detections=detections,
        )


def capture_webcam_frame_mock(frame_id: int) -> YoloFrame:
    """
    Fallback para leitura de webcam real ou mock quando OpenCV não possuir câmera conectada.
    """
    simulator = YoloScenarioSimulator()
    return simulator.generate_frame("lead_quente_curioso", step=frame_id)
