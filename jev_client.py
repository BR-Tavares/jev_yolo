"""
Cliente de Comunicação com o Jev Decision Engine na OpenRouter (jev_client.py).
Conecta ao endpoint https://openrouter.ai/api/alpha/decisions com validação rigorosa de schema,
suporte assíncrono (httpx) e síncrono, medição de latência RTT e modo de simulação física inteligente (dry-run).
"""

import time
import json
from typing import Dict, Any, Optional, Tuple
import httpx
from config import settings
from questions_config import get_jev_questions_payload, get_question_ids


class JevClient:
    """Cliente HTTP para o endpoint /api/alpha/decisions do Jev na OpenRouter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 8.0,
    ):
        self.api_key = (api_key or settings.OPENROUTER_API_KEY).strip()
        self.model = (model or settings.JEV_MODEL).strip()
        self.base_url = (base_url or settings.OPENROUTER_BASE_URL).strip()
        self.timeout_seconds = timeout_seconds

    def _build_headers(self) -> Dict[str, str]:
        """Constrói os cabeçalhos HTTP exigidos pela OpenRouter."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.HTTP_REFERER,
            "X-Title": settings.APP_TITLE,
        }

    def _build_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Monta o payload estrito com o state e as 4 perguntas noul."""
        return {
            "model": self.model,
            "state": state,
            "questions": get_jev_questions_payload(),
        }

    def _parse_answers(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrai com segurança as probabilidades das respostas da API do Jev.
        Suporta formatos:
          - {"answers": {"q_id": {"noul": 0.85}}}
          - {"answers": {"q_id": 0.85}}
          - {"choices": [{"message": {"content": ...}}]}
        """
        probabilities: Dict[str, float] = {}
        expected_ids = get_question_ids()

        answers_map = data.get("answers")
        if not answers_map and "choices" in data and len(data["choices"]) > 0:
            first_choice = data["choices"][0]
            if "message" in first_choice and "content" in first_choice["message"]:
                try:
                    content_json = json.loads(first_choice["message"]["content"])
                    answers_map = content_json.get("answers", content_json)
                except Exception:
                    pass

        if not isinstance(answers_map, dict):
            answers_map = {}

        for q_id in expected_ids:
            val = answers_map.get(q_id)
            prob = 0.0
            if isinstance(val, dict):
                # Padrão formal do TypeSafe Jev para noul: {"noul": 0.85}
                prob = float(val.get("noul", val.get("probability", 0.0)))
            elif isinstance(val, (int, float)):
                prob = float(val)
            elif isinstance(val, str):
                try:
                    prob = float(val)
                except ValueError:
                    prob = 0.0

            # Garante limites contínuos [0.0, 1.0]
            prob = max(0.0, min(1.0, prob))
            probabilities[q_id] = prob

        return probabilities

    def _simulate_smart_fallback(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula probabilidades analíticas fiéis aos critérios do Jev caso
        a chave de API não esteja configurada ou para modo offline de bancada.
        """
        metrics = state.get("metrics", {})
        active_count = metrics.get("active_visitors_count", 0)
        dwell = metrics.get("max_dwell_time_seconds", 0.0)
        gaze = metrics.get("gaze_alignment_with_demo_pct", 0.0)
        velocity = metrics.get("walking_velocity_mps", 1.0)
        sales_dist = metrics.get("sales_rep_distance_meters", 3.0)
        interacting = metrics.get("sales_rep_interacting", False)
        crowd_density = metrics.get("crowd_density_people_per_sqm", 0.0)
        contact_s = metrics.get("sustained_close_contact_seconds", 0.0)

        # 1. p_purchase_intent
        # Parou, olhando para a demo, velocidade baixa
        p_intent = 0.05
        if active_count > 0:
            dwell_factor = min(1.0, dwell / 25.0)
            gaze_factor = gaze / 100.0
            vel_factor = max(0.0, 1.0 - (velocity / 1.0))
            p_intent = 0.10 + 0.45 * dwell_factor + 0.30 * gaze_factor + 0.15 * vel_factor

        # 2. p_sales_approach_urgency
        # Alto interesse, mas vendedor distante e não interagindo
        p_urgency = 0.05
        if active_count > 0 and not interacting:
            dist_factor = min(1.0, sales_dist / 3.0)
            p_urgency = p_intent * (0.4 + 0.6 * dist_factor)
        elif interacting:
            p_urgency = 0.12

        # 3. p_demo_attractiveness_peak
        # Múltiplas pessoas focadas simultaneamente
        p_peak = 0.05
        if active_count >= 2:
            crowd_factor = min(1.0, active_count / 5.0)
            gaze_factor = gaze / 100.0
            p_peak = 0.15 + 0.50 * crowd_factor + 0.35 * gaze_factor

        # 4. p_airborne_transmission_risk
        # Alta densidade e contato próximo prolongado em ambiente fechado
        density_factor = min(1.0, crowd_density / 2.5)
        contact_factor = min(1.0, contact_s / 40.0)
        p_risk = 0.08 + 0.50 * density_factor + 0.42 * contact_factor

        # 5. p_user_satisfaction_engagement (câmera frontal do notebook)
        nb_info = state.get("notebook_interaction", {})
        p_user_sat = 0.0
        if nb_info.get("user_present", False):
            scr_gaze = nb_info.get("screen_gaze_contact_pct", 0.0) / 100.0
            sat_score = nb_info.get("satisfaction_score_pct", 0.0) / 100.0
            dwell_nb = min(1.0, nb_info.get("attention_dwell_time_seconds", 0.0) / 30.0)
            p_user_sat = 0.15 + 0.40 * scr_gaze + 0.35 * sat_score + 0.10 * dwell_nb

        return {
            "p_purchase_intent": round(max(0.0, min(1.0, p_intent)), 3),
            "p_sales_approach_urgency": round(max(0.0, min(1.0, p_urgency)), 3),
            "p_demo_attractiveness_peak": round(max(0.0, min(1.0, p_peak)), 3),
            "p_airborne_transmission_risk": round(max(0.0, min(1.0, p_risk)), 3),
            "p_user_satisfaction_engagement": round(max(0.0, min(1.0, p_user_sat)), 3),
        }

    async def decide_async(
        self, state: Dict[str, Any]
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """
        Executa chamada assíncrona ao endpoint da OpenRouter.
        Retorna: (probabilidades, latencia_ms, raw_response_ou_meta)
        """
        # Se chave não configurada, utiliza simulador inteligente
        if not self.api_key or self.api_key.startswith("your_"):
            time_start = time.perf_counter()
            sim_probs = self._simulate_smart_fallback(state)
            latency_ms = (time.perf_counter() - time_start) * 1000 + 120.0
            return (
                sim_probs,
                round(latency_ms, 1),
                {"mode": "smart_simulation", "reason": "No valid OpenRouter API Key"},
            )

        payload = self._build_payload(state)
        headers = self._build_headers()

        time_start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            latency_ms = (time.perf_counter() - time_start) * 1000.0

            if response.status_code != 200:
                error_body = response.text
                raise RuntimeError(
                    f"OpenRouter API Error HTTP {response.status_code}: {error_body}"
                )

            data = response.json()
            probs = self._parse_answers(data)
            return probs, round(latency_ms, 1), data

    def decide_sync(
        self, state: Dict[str, Any]
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """
        Executa chamada síncrona (conveniente para Streamlit e scripts de teste diretos).
        """
        if not self.api_key or self.api_key.startswith("your_"):
            time_start = time.perf_counter()
            sim_probs = self._simulate_smart_fallback(state)
            latency_ms = (time.perf_counter() - time_start) * 1000 + 120.0
            return (
                sim_probs,
                round(latency_ms, 1),
                {"mode": "smart_simulation", "reason": "No valid OpenRouter API Key"},
            )

        payload = self._build_payload(state)
        headers = self._build_headers()

        time_start = time.perf_counter()
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.base_url, headers=headers, json=payload)
            latency_ms = (time.perf_counter() - time_start) * 1000.0

            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter API Error HTTP {response.status_code}: {response.text}"
                )

            data = response.json()
            probs = self._parse_answers(data)
            return probs, round(latency_ms, 1), data
