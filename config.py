"""
Módulo de Configuração Centralizada do Sistema JEV + YOLO Time Series Analytics.
Gerencia variáveis de ambiente, modelos suportados e parâmetros do pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env caso exista
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """Configurações gerais do sistema e endpoints da OpenRouter / Jev."""

    # Autenticação e Modelos OpenRouter
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    JEV_MODEL: str = os.getenv("JEV_MODEL", "typesafe/jev-1.13").strip()
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/alpha/decisions"
    ).strip()

    # Metadados de Cabeçalho HTTP
    APP_TITLE: str = os.getenv("APP_TITLE", "Jev YOLO Time Series Analytics").strip()
    HTTP_REFERER: str = os.getenv(
        "HTTP_REFERER", "https://github.com/jev-yolo-analytics"
    ).strip()

    # Time Series e Suavização
    EMA_ALPHA: float = float(os.getenv("EMA_ALPHA", "0.35"))
    HISTORY_MAX_POINTS: int = int(os.getenv("HISTORY_MAX_POINTS", "120"))
    DASHBOARD_REFRESH_RATE_MS: int = int(os.getenv("DASHBOARD_REFRESH_RATE_MS", "1500"))

    # Parâmetros de Geometria e Calibração Padrão (Estande da Feira)
    BOOTH_NOMINAL_CAPACITY: int = 15
    BOOTH_DISPLAY_POS_M: tuple[float, float] = (0.0, 4.2)  # Posição da bancada do notebook no interior do estande
    ENGAGEMENT_DISTANCE_THRESHOLD_M: float = 1.8  # Limite para considerar engajamento em frente à bancada (m)
    ENGAGEMENT_VELOCITY_THRESHOLD_MPS: float = 0.35  # Velocidade máxima para engajamento (m/s)

    @classmethod
    def is_api_key_configured(cls) -> bool:
        """Verifica se a chave da OpenRouter foi devidamente informada."""
        return bool(cls.OPENROUTER_API_KEY and not cls.OPENROUTER_API_KEY.startswith("your_"))


settings = Settings()
