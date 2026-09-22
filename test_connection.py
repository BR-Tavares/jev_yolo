"""
Suíte de Testes e Validação do Pipeline JEV + YOLO (test_connection.py).
Executa 5 testes rigorosos de comunicação, validação de schema, extração numérica,
sensibilidade dinâmica e amortecimento EMA conforme Seção 4 do PLAN_JEV_YOLO.md.
"""

import sys
import json
import time
from config import settings
from questions_config import get_jev_questions_payload, get_question_ids, JEV_QUESTIONS
from sitrep_engine import SitrepEngine
from yolo_stream import YoloScenarioSimulator
from jev_client import JevClient
from history_tracker import HistoryTracker


def test_1_strict_schema():
    """Valida a conformidade estrita do schema do Jev na OpenRouter."""
    print("=" * 70)
    print("TESTE 1: Validação do Schema Estrito TypeSafe (OpenRouter /api/alpha/decisions)")
    print("=" * 70)

    # 1.1 Modelo
    assert settings.JEV_MODEL.startswith("typesafe/jev"), f"Modelo inesperado: {settings.JEV_MODEL}"
    print(f"[OK] Modelo configurado: {settings.JEV_MODEL}")

    # 1.2 Estrutura das 5 perguntas noul
    payload = get_jev_questions_payload()
    expected_ids = [
        "p_purchase_intent",
        "p_stand_retention_engagement",
        "p_demo_attractiveness_peak",
        "p_airborne_transmission_risk",
        "p_user_satisfaction_engagement",
    ]
    assert set(payload.keys()) == set(expected_ids), f"Divergência de IDs: {list(payload.keys())}"

    for qid, qdef in payload.items():
        assert qdef["type"] == "noul", f"Tipo deve ser 'noul', recebido: {qdef['type']}"
        instr = qdef.get("instructions") or qdef.get("instruction", "")
        assert len(instr) > 10, f"Instrução ausente em {qid}"
        assert "criteria" in qdef, f"Criteria ausente em {qid}"
        assert "false" in qdef["criteria"] and "true" in qdef["criteria"], f"Par false/true ausente em {qid}"
        print(f"  - Pergunta '{qid}': OK (noul, instructions + criteria)")

    # 1.3 Validação de tamanho compacto do State (< 1000 tokens)
    sim = YoloScenarioSimulator()
    engine = SitrepEngine()
    frame = sim.generate_frame("lead_quente_curioso", step=1)
    state = engine.process_frame(frame)
    state_str = json.dumps(state)
    approx_tokens = len(state_str) // 4
    print(f"[OK] Estado serializado: {len(state_str)} caracteres (~{approx_tokens} tokens)")
    assert approx_tokens < 1000, "State excedeu tamanho recomendado"
    print("[TESTE 1 APROVADO]\n")


def test_2_numerical_extraction_and_api():
    """Valida a extração numérica das 4 probabilidades float retornadas."""
    print("=" * 70)
    print("TESTE 2: Extração Numérica das 4 Probabilidades e Conectividade")
    print("=" * 70)

    sim = YoloScenarioSimulator()
    engine = SitrepEngine()
    client = JevClient()

    frame = sim.generate_frame("lead_quente_curioso", step=1)
    state = engine.process_frame(frame)

    print(f"Enviando requisição (Modo: {'Real API' if settings.is_api_key_configured() else 'Smart Dry-Run Simulation'})...")
    probs, latency_ms, raw = client.decide_sync(state)

    print(f"[OK] RTT Latência: {latency_ms:.1f} ms")
    for qid, val in probs.items():
        assert isinstance(val, float), f"Valor de {qid} deve ser float, recebido: {type(val)}"
        assert 0.0 <= val <= 1.0, f"Valor fora do intervalo [0.0, 1.0]: {val}"
        print(f"  - {qid}: {val:.3f} ({val * 100:.1f}%)")

    print("[TESTE 2 APROVADO]\n")


def test_3_dynamic_sensitivity():
    """Certifica que variações físicas no YOLO provocam respostas coerentes nas probabilidades."""
    print("=" * 70)
    print("TESTE 3: Sensibilidade Dinâmica a Variações do YOLO")
    print("=" * 70)

    client = JevClient()

    # Estado A: Transeunte Rápido (1.35 m/s, olhando para o corredor, dwell time 0.5s)
    state_a = {
        "zone_telemetry": {
            "location": "corredor_feira_tecnologia",
            "booth_capacity_nominal": 15,
            "ventilation_status": "enclosed_pavilion_low_turnover",
        },
        "metrics": {
            "active_visitors_count": 1,
            "corridor_passersby_count": 1,
            "booth_visitors_retained_count": 0,
            "max_dwell_time_seconds": 0.5,
            "gaze_alignment_with_demo_pct": 12.0,
            "walking_velocity_mps": 1.35,
            "crowd_density_people_per_sqm": 0.1,
            "sustained_close_contact_seconds": 0.0,
        },
    }
    probs_a, lat_a, _ = client.decide_sync(state_a)

    # Estado B: Lead Quente Retido no Mostruário (0.05 m/s, focado na demo por 35s)
    state_b = {
        "zone_telemetry": {
            "location": "corredor_feira_tecnologia",
            "booth_capacity_nominal": 15,
            "ventilation_status": "enclosed_pavilion_low_turnover",
        },
        "metrics": {
            "active_visitors_count": 1,
            "corridor_passersby_count": 0,
            "booth_visitors_retained_count": 1,
            "max_dwell_time_seconds": 35.0,
            "gaze_alignment_with_demo_pct": 88.0,
            "walking_velocity_mps": 0.05,
            "crowd_density_people_per_sqm": 0.2,
            "sustained_close_contact_seconds": 0.0,
        },
    }
    probs_b, lat_b, _ = client.decide_sync(state_b)

    print(f"Estado A (Passante Rápido)   -> P(Compra): {probs_a['p_purchase_intent']:.3f} | P(Retenção): {probs_a['p_stand_retention_engagement']:.3f}")
    print(f"Estado B (Lead Engajado 35s) -> P(Compra): {probs_b['p_purchase_intent']:.3f} | P(Retenção): {probs_b['p_stand_retention_engagement']:.3f}")

    assert probs_b["p_purchase_intent"] > probs_a["p_purchase_intent"], "P(Compra) deveria aumentar com engajamento"
    assert probs_b["p_stand_retention_engagement"] > probs_a["p_stand_retention_engagement"], "P(Retenção) deveria aumentar com permanência no estande"

    print("[OK] O modelo respondeu com sensibilidade direcional esperada às alterações cinéticas.")
    print("[TESTE 3 APROVADO]\n")


def test_4_ema_filter_hysteresis():
    """Valida o amortecimento de ruído e oclusões transitórias via filtro EMA."""
    print("=" * 70)
    print("TESTE 4: Filtro de Histerese Temporal e Amortecimento de Ruído (EMA)")
    print("=" * 70)

    tracker = HistoryTracker(maxlen=50, alpha=0.35)

    # Sequência estável com probabilidade alta (0.85)
    for _ in range(5):
        tracker.add_point({"p_purchase_intent": 0.85, "p_stand_retention_engagement": 0.80, "p_demo_attractiveness_peak": 0.50, "p_airborne_transmission_risk": 0.10, "p_user_satisfaction_engagement": 0.70})

    prev_filt = tracker.get_latest_values()["p_purchase_intent"]["filtered"]
    print(f"Valor estável antes do drop: {prev_filt:.3f}")

    # Injeção de ruído / oclusão transitória de 1 frame (queda para 0.0)
    tracker.add_point({"p_purchase_intent": 0.0, "p_stand_retention_engagement": 0.0, "p_demo_attractiveness_peak": 0.0, "p_airborne_transmission_risk": 0.0, "p_user_satisfaction_engagement": 0.0})

    drop_filt = tracker.get_latest_values()["p_purchase_intent"]["filtered"]
    print(f"Valor filtrado após oclusão instantânea (Raw: 0.0): {drop_filt:.3f}")

    # Com alpha=0.35, novo_valor = 0.35 * 0 + 0.65 * 0.85 = ~0.5525
    # O valor não cai bruscamente para zero!
    assert drop_filt > 0.40, f"Filtro EMA não amortizou a queda adequadamente: {drop_filt}"
    print("[OK] Filtro EMA absorveu o drop espúrio com amortecimento suave.")
    print("[TESTE 4 APROVADO]\n")


def run_all_tests():
    """Executa todos os testes sequencialmente."""
    print("\nINICIANDO SUÍTE DE TESTES: JEV + YOLO TIME SERIES ANALYTICS\n")
    try:
        test_1_strict_schema()
        test_2_numerical_extraction_and_api()
        test_3_dynamic_sensitivity()
        test_4_ema_filter_hysteresis()
        print("=" * 70)
        print("TODOS OS TESTES FORAM CONCLUÍDOS COM SUCESSO!")
        print("=" * 70)
    except AssertionError as err:
        print(f"\n[FALHA NO TESTE]: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as ex:
        print(f"\n[ERRO DE EXECUÇÃO]: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
