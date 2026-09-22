"""
Módulo de Percepção e Engajamento da Câmera do Notebook (notebook_perception.py).
Mede em escala micro (face a tela) a atenção, proximidade e satisfação de quem interage
diretamente com a demonstração no estande da feira.
"""

import time
import random
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class NotebookPerceptionData(BaseModel):
    """Telemetria da câmera frontal do notebook focada no usuário atual."""
    user_present: bool = Field(..., description="Se há alguém na frente da tela do notebook")
    screen_gaze_contact_pct: float = Field(..., description="Porcentagem de tempo olhando diretamente para a tela")
    satisfaction_score_pct: float = Field(..., description="Score de expressão positiva / engajamento facial (0 a 100%)")
    attention_dwell_time_seconds: float = Field(..., description="Segundos ininterruptos focado no notebook")
    face_distance_cm: float = Field(..., description="Distância estimada do rosto até a tela em centímetros")


class NotebookPerceptionSimulator:
    """
    Simulador de percepção da webcam local do notebook para cenários da feira.
    Permite demonstrar reações de visitantes testando o software.
    """

    def __init__(self):
        self._dwell_accum_s: float = 0.0
        self._last_time: float = time.time()

    def generate_perception(self, scenario: str = "visitante_encantado", step: int = 0) -> NotebookPerceptionData:
        """
        Gera telemetria facial de acordo com o padrão do visitante no notebook:
        - visitante_encantado: alto foco na tela, expressão sorridente/curiosa, proximidade ideal (50cm).
        - visitante_focado_tecnico: muito focado e sério, testando o sistema com atenção contínua.
        - visitante_distraido: olhando ao redor, mexendo no celular, baixo contato visual.
        - notebook_livre: ninguém na frente do notebook no momento.
        """
        now = time.time()
        dt = max(0.1, min(now - self._last_time, 2.0))
        self._last_time = now

        if scenario == "notebook_livre":
            self._dwell_accum_s = 0.0
            return NotebookPerceptionData(
                user_present=False,
                screen_gaze_contact_pct=0.0,
                satisfaction_score_pct=0.0,
                attention_dwell_time_seconds=0.0,
                face_distance_cm=0.0,
            )

        self._dwell_accum_s += dt

        if scenario == "visitante_encantado":
            gaze = round(min(100.0, 88.0 + random.uniform(-4, 6)), 1)
            satisfaction = round(min(100.0, 85.0 + random.uniform(-5, 8)), 1)
            distance = round(52.0 + random.uniform(-3, 3), 1)
            dwell = round(self._dwell_accum_s + 20.0, 1)

        elif scenario == "visitante_focado_tecnico":
            gaze = round(min(100.0, 94.0 + random.uniform(-2, 4)), 1)
            satisfaction = round(min(100.0, 68.0 + random.uniform(-4, 5)), 1)  # Mais sério/analítico
            distance = round(48.0 + random.uniform(-2, 2), 1)
            dwell = round(self._dwell_accum_s + 35.0, 1)

        elif scenario == "visitante_distraido":
            gaze = round(max(5.0, 22.0 + random.uniform(-10, 10)), 1)
            satisfaction = round(max(10.0, 35.0 + random.uniform(-8, 8)), 1)
            distance = round(75.0 + random.uniform(-5, 10), 1)
            dwell = round(min(self._dwell_accum_s, 6.0), 1)

        else:
            gaze = 70.0
            satisfaction = 50.0
            distance = 55.0
            dwell = round(self._dwell_accum_s, 1)

        return NotebookPerceptionData(
            user_present=True,
            screen_gaze_contact_pct=gaze,
            satisfaction_score_pct=satisfaction,
            attention_dwell_time_seconds=dwell,
            face_distance_cm=distance,
        )
