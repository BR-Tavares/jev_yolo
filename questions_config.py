"""
Definição das 4 Perguntas de Probabilidade (Tipo 'noul') para o Modelo Jev.
Contrato rigoroso conforme especificado na Seção 3 do PLAN_JEV_YOLO.md.
"""

from typing import Any, Dict


# As 4 perguntas estritas submetidas no payload do Jev
JEV_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "p_purchase_intent": {
        "type": "noul",
        "instructions": "Avalie a probabilidade de o visitante em frente ao estande ter intenção real de compra ou interesse comercial.",
        "instruction": "Avalie a probabilidade de o visitante em frente ao estande ter intenção real de compra ou interesse comercial.",
        "criteria": {
            "false": "Visitante em trânsito rápido pelo corredor, olhando para frente ou sem foco no produto.",
            "true": "Visitante desacelerou ou parou em frente ao mostruário com corpo e olhar voltados para a demonstração.",
        },
        "display_name": "Intenção de Compra / Interesse Comercial",
        "color": "#10B981",  # Verde esmeralda
        "threshold": 0.70,
        "alert_text": "Lead Qualificado! Alto interesse detectado no mostruário.",
    },
    "p_stand_retention_engagement": {
        "type": "noul",
        "instructions": "Avalie a probabilidade de o visitante estar altamente retido e engajado com o espaço do estande, em contraste com o fluxo rápido do corredor.",
        "instruction": "Avalie a probabilidade de o visitante estar altamente retido e engajado com o espaço do estande, em contraste com o fluxo rápido do corredor.",
        "criteria": {
            "false": "Pessoa em velocidade normal de caminhada de corredor, ignorando a fachada.",
            "true": "Pessoa rompeu o fluxo do corredor, permaneceu parada no espaço do estande e direcionou a atenção ao mostruário.",
        },
        "display_name": "Retenção e Engajamento no Estande",
        "color": "#F59E0B",  # Âmbar / Laranja
        "threshold": 0.70,
        "alert_text": "Alta Retenção de Público! Visitante absorvido pela exposição.",
    },
    "p_demo_attractiveness_peak": {
        "type": "noul",
        "instructions": "Avalie a probabilidade de a demonstração do produto estar atingindo um pico de atração e retenção de público.",
        "instruction": "Avalie a probabilidade de a demonstração do produto estar atingindo um pico de atração e retenção de público.",
        "criteria": {
            "false": "Público disperso, ignorando a tela interativa ou totem.",
            "true": "Múltiplos visitantes convergindo e mantendo foco simultâneo na demonstração.",
        },
        "display_name": "Pico de Atração da Demonstração",
        "color": "#3B82F6",  # Azul royal
        "threshold": 0.70,
        "alert_text": "Pico de Engajamento! Demonstração atraindo e retendo múltiplos visitantes.",
    },
    "p_airborne_transmission_risk": {
        "type": "noul",
        "instructions": "Avalie a probabilidade de haver risco aumentado de transmissão aérea de vírus respiratórios no local.",
        "instruction": "Avalie a probabilidade de haver risco aumentado de transmissão aérea de vírus respiratórios no local.",
        "criteria": {
            "false": "Ambiente ventilado ou pessoas em movimento com espaçamento adequado.",
            "true": "Aglomeração densa e estática por tempo prolongado sob ar estagnado.",
        },
        "display_name": "Risco de Transmissão Aérea (Biossegurança)",
        "color": "#EF4444",  # Vermelho
        "threshold": 0.70,
        "alert_text": "Alerta de Biossegurança! Alta densidade estática por tempo prolongado.",
    },
    "p_user_satisfaction_engagement": {
        "type": "noul",
        "instructions": "Avalie a probabilidade de o visitante que está interagindo diretamente com o notebook/demonstração estar altamente engajado e satisfeito com a experiência.",
        "instruction": "Avalie a probabilidade de o visitante que está interagindo diretamente com o notebook/demonstração estar altamente engajado e satisfeito com a experiência.",
        "criteria": {
            "false": "Visitante ausente, distraído, olhando para longe da tela ou com postura corporal desinteressada.",
            "true": "Visitante próximo à tela, mantendo contato visual direto contínuo e demonstrando expressão de foco ou satisfação positiva.",
        },
        "display_name": "Engajamento e Satisfação no Notebook",
        "color": "#8B5CF6",  # Roxo violeta
        "threshold": 0.70,
        "alert_text": "Lead Encantado no Notebook! Alta satisfação e foco na demonstração.",
    },
}


def get_jev_questions_payload() -> Dict[str, Dict[str, Any]]:
    """
    Retorna o dicionário de perguntas compatível com o schema estrito da OpenRouter / Jev:
    Cada item contém estritamente: type, instructions e criteria.
    """
    payload = {}
    for q_id, q_data in JEV_QUESTIONS.items():
        payload[q_id] = {
            "type": q_data["type"],
            "instructions": q_data["instructions"],
            "criteria": q_data["criteria"],
        }
    return payload


def get_question_ids() -> list[str]:
    """Retorna a lista ordenada dos IDs das 4 perguntas."""
    return list(JEV_QUESTIONS.keys())
