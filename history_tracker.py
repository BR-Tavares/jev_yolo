"""
Gerenciador de Séries Temporais e Filtro EMA (history_tracker.py).
Implementa o Ring Buffer cronológico e o filtro de Média Móvel Exponencial (EMA)
para amortecimento de ruído e oclusões transitórias do detector visual.
"""

from collections import deque
import time
from typing import Dict, List, Optional
import pandas as pd
from questions_config import get_question_ids


class HistoryTracker:
    """
    Mantém uma janela deslizante (Ring Buffer) de pontos no tempo
    armazenando probabilidades brutas e suavizadas pelo filtro EMA.
    """

    def __init__(self, maxlen: int = 120, alpha: float = 0.35):
        self.maxlen = maxlen
        self.alpha = alpha
        self.timestamps: deque[float] = deque(maxlen=maxlen)
        self.formatted_times: deque[str] = deque(maxlen=maxlen)
        self.latencies: deque[float] = deque(maxlen=maxlen)

        self.question_ids = get_question_ids()

        # Buffers de probabilidades brutas e filtradas
        self.raw_history: Dict[str, deque[float]] = {
            qid: deque(maxlen=maxlen) for qid in self.question_ids
        }
        self.filtered_history: Dict[str, deque[float]] = {
            qid: deque(maxlen=maxlen) for qid in self.question_ids
        }

        # Últimos valores filtrados para o cálculo recursivo do EMA
        self._last_filtered: Dict[str, Optional[float]] = {
            qid: None for qid in self.question_ids
        }

        # Buffers do Qualified Engagement Index (QEI - Padrão CEIR / RetailNext)
        self.qei_raw: deque[float] = deque(maxlen=maxlen)
        self.qei_filtered: deque[float] = deque(maxlen=maxlen)
        self._last_qei_filt: Optional[float] = None

    def set_alpha(self, new_alpha: float):
        """Ajusta dinamicamente a constante de suavização do filtro EMA [0.05 a 1.0]."""
        self.alpha = max(0.01, min(1.0, new_alpha))

    def add_point(
        self,
        probabilities: Dict[str, float],
        latency_ms: float = 0.0,
        timestamp_s: Optional[float] = None,
        qei: float = 0.0,
    ):
        """
        Adiciona um novo vetor de probabilidades de inferência do Jev e o QEI no buffer temporal.
        Calcula recursivamente o filtro EMA.
        """
        # Proteção defensiva para instâncias persistentes na sessão do Streamlit
        if not hasattr(self, "_last_filtered") or self._last_filtered is None:
            self._last_filtered = {qid: None for qid in self.question_ids}
        if not hasattr(self, "qei_raw"):
            self.qei_raw = deque(maxlen=self.maxlen)
        if not hasattr(self, "qei_filtered"):
            self.qei_filtered = deque(maxlen=self.maxlen)
        if not hasattr(self, "_last_qei_filt"):
            self._last_qei_filt = None
        t = timestamp_s or time.time()
        self.timestamps.append(t)
        self.formatted_times.append(time.strftime("%H:%M:%S", time.localtime(t)))
        self.latencies.append(latency_ms)

        # Suavização EMA para o QEI [0 a 100]
        self.qei_raw.append(qei)
        if self._last_qei_filt is None:
            filt_qei = qei
        else:
            filt_qei = self.alpha * qei + (1.0 - self.alpha) * self._last_qei_filt
        self._last_qei_filt = filt_qei
        self.qei_filtered.append(filt_qei)

        for qid in self.question_ids:
            raw_val = probabilities.get(qid, 0.0)
            self.raw_history[qid].append(raw_val)

            # Aplicação do Filtro EMA
            last_filt = self._last_filtered[qid]
            if last_filt is None:
                filtered_val = raw_val
            else:
                filtered_val = self.alpha * raw_val + (1.0 - self.alpha) * last_filt

            self._last_filtered[qid] = filtered_val
            self.filtered_history[qid].append(filtered_val)

    def get_latest_values(self) -> Dict[str, Dict[str, float]]:
        """Retorna os valores mais recentes (raw, filtered e delta imediato)."""
        latest: Dict[str, Dict[str, float]] = {}
        for qid in self.question_ids:
            filt_list = list(self.filtered_history[qid])
            raw_list = list(self.raw_history[qid])

            curr_filt = filt_list[-1] if filt_list else 0.0
            prev_filt = filt_list[-2] if len(filt_list) >= 2 else curr_filt
            curr_raw = raw_list[-1] if raw_list else 0.0

            latest[qid] = {
                "filtered": curr_filt,
                "raw": curr_raw,
                "delta": curr_filt - prev_filt,
            }
        return latest

    def get_latest_qei(self) -> Dict[str, float]:
        """Retorna a métrica executiva mais recente do Qualified Engagement Index (QEI)."""
        filt_list = list(self.qei_filtered)
        raw_list = list(self.qei_raw)
        curr_filt = filt_list[-1] if filt_list else 0.0
        prev_filt = filt_list[-2] if len(filt_list) >= 2 else curr_filt
        curr_raw = raw_list[-1] if raw_list else 0.0
        return {
            "filtered": round(curr_filt, 1),
            "raw": round(curr_raw, 1),
            "delta": round(curr_filt - prev_filt, 1),
        }

    def get_latest_latency(self) -> float:
        """Retorna a latência mais recente em milissegundos."""
        return self.latencies[-1] if self.latencies else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Exporta os dados armazenados como DataFrame do Pandas pronto para visualização."""
        if not self.timestamps:
            cols = ["time", "timestamp", "latency_ms", "qei_raw", "qei_filt"]
            for qid in self.question_ids:
                cols.append(f"{qid}_raw")
                cols.append(f"{qid}_filt")
            return pd.DataFrame(columns=cols)

        data = {
            "time": list(self.formatted_times),
            "timestamp": list(self.timestamps),
            "latency_ms": list(self.latencies),
            "qei_raw": list(self.qei_raw),
            "qei_filt": list(self.qei_filtered),
        }
        for qid in self.question_ids:
            data[f"{qid}_raw"] = list(self.raw_history[qid])
            data[f"{qid}_filt"] = list(self.filtered_history[qid])

        return pd.DataFrame(data)

    def clear(self):
        """Limpa todo o histórico."""
        self.timestamps.clear()
        self.formatted_times.clear()
        self.latencies.clear()
        self.qei_raw.clear()
        self.qei_filtered.clear()
        self._last_qei_filt = None
        for qid in self.question_ids:
            self.raw_history[qid].clear()
            self.filtered_history[qid].clear()
            self._last_filtered[qid] = None
