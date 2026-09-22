"""
Motor Espacial e Cinético Local (sitrep_engine.py).
Executa homografia 2D de solo, cálculo de velocidades vetoriais, acumulador de permanência (dwell time),
distâncias euclidianas interpessoais e de promotores de vendas, gerando o SITREP semântico enxuto
para o payload de decisão do Jev.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from config import settings
from yolo_stream import YoloFrame, Detection


class TrackHistory:
    """Mantém o histórico cinético e temporal de um indivíduo rastreado."""
    def __init__(self, track_id: int, initial_pos_m: Tuple[float, float], timestamp_s: float, initial_velocity: float = 1.2):
        self.track_id = track_id
        self.last_pos_m = initial_pos_m
        self.last_timestamp_s = timestamp_s
        self.velocity_mps = initial_velocity
        self.dwell_time_s = 0.0
        self.is_engaged = False


class SitrepEngine:
    """
    Motor local de fusão espacial e cinética.
    Transforma coordenadas em pixels do YOLO em métricas físicas do estande (metros, segundos, densidade).
    """

    def __init__(self, booth_area_sqm: float = 12.0):
        self.booth_area_sqm = booth_area_sqm
        self.tracks: Dict[int, TrackHistory] = {}
        self.close_contact_accum_s: float = 0.0
        self.last_frame_timestamp_s: Optional[float] = None
        self.last_pedestrians_spatial_state: List[Dict[str, Any]] = []

        # Matriz de Homografia padrão (Pixels da câmera -> Metros no chão do estande)
        # Calibrada para resolução 1280x720 com totem centralizado em (0.0, 2.5m)
        self.H = self._compute_default_homography()

    def _compute_default_homography(self) -> np.ndarray:
        """Gera matriz de perspectiva 3x3 para transformar pontos da imagem em coordenadas métricas de solo."""
        # 4 pontos no plano da imagem (px) -> 4 pontos no mundo real (metros)
        src_pts = np.float32([
            [300, 700],   # inferior esquerdo
            [980, 700],   # inferior direito
            [480, 400],   # superior esquerdo
            [800, 400],   # superior direito
        ])
        dst_pts = np.float32([
            [-1.5, 0.5],
            [1.5, 0.5],
            [-1.5, 4.0],
            [1.5, 4.0],
        ])
        # cv2.getPerspectiveTransform ou via álgebra linear direta
        # Usamos cv2 se disponível, senão resolução de sistema linear
        try:
            import cv2
            return cv2.getPerspectiveTransform(src_pts, dst_pts)
        except Exception:
            # Fallback direto via modelo afim/projetivo aproximado
            return np.array([
                [0.0035, 0.0000, -1.8],
                [0.0000, 0.0105, -3.2],
                [0.0000, 0.0008,  1.0]
            ], dtype=np.float32)

    def pixel_to_ground_meters(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """Projeta coordenadas de contato dos pés (pixels) para o plano métrico do solo."""
        vec = np.array([px_x, px_y, 1.0], dtype=np.float32)
        projected = np.dot(self.H, vec)
        w = projected[2] if abs(projected[2]) > 1e-6 else 1.0
        return float(projected[0] / w), float(projected[1] / w)

    def process_frame(
        self,
        frame: YoloFrame,
        notebook_perception: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Processa um frame de telemetria do corredor e dados da webcam do notebook,
        sintetizando o SITREP unificado para o Jev.
        """
        now_s = frame.timestamp_ms / 1000.0
        dt = (now_s - self.last_frame_timestamp_s) if self.last_frame_timestamp_s else 0.1
        dt = max(0.01, min(dt, 1.0))
        self.last_frame_timestamp_s = now_s

        visitors_positions: List[Tuple[float, float]] = []
        velocities: List[float] = []
        dwell_times: List[float] = []
        gaze_alignments: List[float] = []
        head_yaws: List[float] = []

        active_track_ids = set()

        # Histórico espacial consolidado de pedestres para a planta baixa e mapa de calor
        current_spatial_list: List[Dict[str, Any]] = []

        corridor_count = 0
        booth_count = 0

        for det in frame.detections:
            track_id = det.track_id
            active_track_ids.add(track_id)

            foot_px = det.center_foot_px
            pos_m = self.pixel_to_ground_meters(foot_px[0], foot_px[1])
            det_yaw = getattr(det, "head_yaw_deg", 0.0)

            # Rotação da cabeça do YOLO-Pose (yaw em graus)
            if hasattr(det, "head_yaw_deg"):
                head_yaws.append(det.head_yaw_deg)

            visitors_positions.append(pos_m)

            # Contabilização de zonas da feira (Y < 2m = Corredor; Y >= 2m = Estande)
            if pos_m[1] < 2.0:
                corridor_count += 1
            else:
                booth_count += 1

            # Atualização do rastreamento cinético
            default_init_v = 1.25 if pos_m[1] < 2.0 else 0.05
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackHistory(track_id, pos_m, now_s, initial_velocity=default_init_v)

            tracker = self.tracks[track_id]
            dx = pos_m[0] - tracker.last_pos_m[0]
            dy = pos_m[1] - tracker.last_pos_m[1]
            dist_moved = math.sqrt(dx * dx + dy * dy)

            # Se a detecção possui speed_mps explícito (YOLO-Pose / Tracker cinético), respeita fielmente
            if hasattr(det, "speed_mps") and det.speed_mps is not None:
                tracker.velocity_mps = det.speed_mps
            else:
                current_velocity = dist_moved / dt
                tracker.velocity_mps = 0.6 * tracker.velocity_mps + 0.4 * current_velocity

            tracker.last_pos_m = pos_m
            velocities.append(tracker.velocity_mps)

            # Distância até o Totem/Display
            display_pos = settings.BOOTH_DISPLAY_POS_M
            dist_to_display = math.sqrt(
                (pos_m[0] - display_pos[0]) ** 2 + (pos_m[1] - display_pos[1]) ** 2
            )

            # Critério de Dwell Time: próximo ao mostruário e baixa velocidade
            if dist_to_display <= settings.ENGAGEMENT_DISTANCE_THRESHOLD_M and tracker.velocity_mps <= settings.ENGAGEMENT_VELOCITY_THRESHOLD_MPS:
                tracker.dwell_time_s += dt
                tracker.is_engaged = True
            else:
                tracker.is_engaged = False

            dwell_times.append(tracker.dwell_time_s)

            current_spatial_list.append({
                "track_id": track_id,
                "pos_m": pos_m,
                "velocity_mps": tracker.velocity_mps,
                "dwell_s": tracker.dwell_time_s,
                "yaw_deg": det_yaw,
                "is_sales_rep": False,
            })

            # Se temos o ângulo de cabeça (YOLO-Pose), usamos alinhamento angular real!
            if hasattr(det, "head_yaw_deg") and det.head_yaw_deg is not None:
                # 0° = 100% alinhamento; 90° = 0% alinhamento
                yaw_err = abs(det.head_yaw_deg)
                pose_alignment = max(5.0, min(100.0, 100.0 - (yaw_err / 90.0) * 100.0))
                gaze_alignments.append(pose_alignment)
            else:
                # Fallback Proxy geométrico cabeça vs tronco
                head_center_x = (det.head_bbox_xyxy[0] + det.head_bbox_xyxy[2]) / 2.0
                body_center_x = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2.0
                torso_head_offset = abs(head_center_x - body_center_x)
                if dist_to_display <= settings.ENGAGEMENT_DISTANCE_THRESHOLD_M:
                    alignment = max(50.0, 100.0 - torso_head_offset * 1.5)
                else:
                    alignment = max(10.0, 40.0 - torso_head_offset * 2.0)
                gaze_alignments.append(alignment)

        # Atualiza a lista espacial de pedestres para consumo direto pelo heatmap
        self.last_pedestrians_spatial_state = current_spatial_list

        # Remove tracks extintos há muito tempo
        self.tracks = {tid: t for tid, t in self.tracks.items() if tid in active_track_ids}

        active_visitors_count = len(visitors_positions)
        max_dwell = max(dwell_times) if dwell_times else 0.0
        avg_velocity = (sum(velocities) / len(velocities)) if velocities else 0.0
        avg_gaze = (sum(gaze_alignments) / len(gaze_alignments)) if gaze_alignments else 0.0
        avg_yaw = (sum(head_yaws) / len(head_yaws)) if head_yaws else 15.0

        avg_yaw = (sum(head_yaws) / len(head_yaws)) if head_yaws else 15.0

        crowd_density = round(active_visitors_count / self.booth_area_sqm, 2)

        close_pairs = 0
        for i in range(len(visitors_positions)):
            for j in range(i + 1, len(visitors_positions)):
                p1 = visitors_positions[i]
                p2 = visitors_positions[j]
                pair_dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if pair_dist < 1.0:
                    close_pairs += 1

        if close_pairs >= 2 and avg_velocity < 0.2:
            self.close_contact_accum_s += dt
        else:
            self.close_contact_accum_s = max(0.0, self.close_contact_accum_s - dt * 0.5)

        # Dados da câmera local do notebook
        nb_dict = {
            "user_present": False,
            "screen_gaze_contact_pct": 0.0,
            "satisfaction_score_pct": 0.0,
            "attention_dwell_time_seconds": 0.0,
            "face_distance_cm": 0.0,
        }
        if notebook_perception is not None:
            if hasattr(notebook_perception, "dict"):
                nb_dict = notebook_perception.dict()
            elif isinstance(notebook_perception, dict):
                nb_dict = notebook_perception

        sitrep_payload = {
            "zone_telemetry": {
                "location": "corredor_feira_tecnologia",
                "booth_capacity_nominal": settings.BOOTH_NOMINAL_CAPACITY,
                "ventilation_status": "enclosed_pavilion_low_turnover",
            },
            "metrics": {
                "active_visitors_count": active_visitors_count,
                "corridor_passersby_count": corridor_count,
                "booth_visitors_retained_count": booth_count,
                "max_dwell_time_seconds": round(max_dwell, 1),
                "gaze_alignment_with_demo_pct": round(avg_gaze, 1),
                "head_yaw_angle_deg": round(avg_yaw, 1),
                "walking_velocity_mps": round(avg_velocity, 2),
                "crowd_density_people_per_sqm": crowd_density,
                "sustained_close_contact_seconds": round(self.close_contact_accum_s, 1),
            },
            "notebook_interaction": nb_dict,
        }

        return sitrep_payload

    def calculate_qei(self, sitrep_payload: Dict[str, Any]) -> float:
        """
        Calcula o Qualified Engagement Index (QEI) no padrão internacional CEIR / RetailNext:
        QEI = (Capture Rate * Demo Trial Rate * Sentiment & Dwell Factor) * 100
        Métrica padrão normalizada [0.0 a 100.0] para teste A/B de arranjos e horários nobres.
        """
        metrics = sitrep_payload.get("metrics", {})
        nb = sitrep_payload.get("notebook_interaction", {})

        active_count = metrics.get("active_visitors_count", 0)
        yaw = metrics.get("head_yaw_angle_deg", 45.0)
        vel = metrics.get("walking_velocity_mps", 1.0)
        dwell = metrics.get("max_dwell_time_seconds", 0.0)

        # 1. Capture Rate: fração de passantes atraídos pela fachada
        yaw_factor = max(0.0, 1.0 - (abs(yaw) / 75.0))
        vel_factor = max(0.0, 1.0 - (vel / 1.3))
        stop_power = 0.5 * yaw_factor + 0.5 * vel_factor
        capture_rate = stop_power if active_count > 0 else 0.05

        # 2. Demo Trial Rate: conversão de presença no estande em teste no notebook
        user_present = nb.get("user_present", False)
        trial_rate = 1.0 if user_present else (0.35 if dwell > 15.0 else 0.08)

        # 3. Quality & Sentiment Factor (Gaze x Satisfação x Dwell na tela)
        if user_present:
            gaze = nb.get("screen_gaze_contact_pct", 0.0) / 100.0
            sat = nb.get("satisfaction_score_pct", 0.0) / 100.0
            dwell_nb = min(1.0, nb.get("attention_dwell_time_seconds", 0.0) / 45.0)
            quality_factor = 0.35 * gaze + 0.45 * sat + 0.20 * dwell_nb
        else:
            quality_factor = min(0.3, dwell / 60.0)

        qei = (capture_rate * trial_rate * quality_factor) * 100.0
        return round(max(0.0, min(100.0, qei)), 1)
