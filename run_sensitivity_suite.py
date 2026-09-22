"""
Script de Teste de Sensibilidade Controlado do Jev com Inspeção de Créditos/Conta e Log Detalhado.
Executa 4 requisições pontuais (uma por cenário), inspeciona cabeçalhos e metadados de créditos da OpenRouter
e grava um relatório completo em log.
"""

import os
import sys
import json
import time
from pathlib import Path
import httpx
from config import settings
from questions_config import get_jev_questions_payload, get_question_ids, JEV_QUESTIONS
from sitrep_engine import SitrepEngine

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "sensitivity_test.log"


def check_openrouter_account_info(api_key: str) -> dict:
    """Consulta os endpoints de metadados da chave e saldo total na OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    result = {}
    try:
        with httpx.Client(timeout=8.0) as client:
            resp_key = client.get("https://openrouter.ai/api/v1/auth/key", headers=headers)
            if resp_key.status_code == 200:
                result["key_info"] = resp_key.json().get("data", {})
            else:
                result["key_info"] = {"error": f"HTTP {resp_key.status_code}"}

            resp_credits = client.get("https://openrouter.ai/api/v1/credits", headers=headers)
            if resp_credits.status_code == 200:
                cdata = resp_credits.json().get("data", {})
                total_credits = float(cdata.get("total_credits", 0.0))
                total_usage = float(cdata.get("total_usage", 0.0))
                remaining_balance = total_credits - total_usage
                result["credits_info"] = {
                    "total_purchased_usd": total_credits,
                    "total_consumed_usd": total_usage,
                    "remaining_balance_usd": round(remaining_balance, 4),
                }
            else:
                result["credits_info"] = {"error": f"HTTP {resp_credits.status_code}"}
    except Exception as e:
        result["error"] = str(e)
    return result


def run_single_decision(api_key: str, state: dict, scenario_name: str) -> dict:
    """Executa uma chamada pontual ao endpoint /api/alpha/decisions e captura headers e corpo completo."""
    url = settings.OPENROUTER_BASE_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.HTTP_REFERER,
        "X-Title": settings.APP_TITLE,
    }
    payload = {
        "model": settings.JEV_MODEL,
        "state": state,
        "questions": get_jev_questions_payload(),
    }

    t0 = time.perf_counter()
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        rtt_ms = (time.perf_counter() - t0) * 1000.0

    resp_headers = dict(resp.headers)
    relevant_headers = {
        k: v for k, v in resp_headers.items()
        if any(term in k.lower() for term in ["rate", "credit", "usage", "cost", "balance", "x-", "cf-"])
    }

    resp_json = {}
    if resp.status_code == 200:
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = {"raw": resp.text}
    else:
        resp_json = {"error_code": resp.status_code, "error_body": resp.text}

    return {
        "scenario": scenario_name,
        "status_code": resp.status_code,
        "rtt_ms": round(rtt_ms, 1),
        "headers": relevant_headers,
        "response": resp_json,
        "input_state": state,
    }


def main():
    api_key = settings.OPENROUTER_API_KEY
    if not api_key or api_key.startswith("your_"):
        print("[ERRO] Chave da OpenRouter não encontrada no .env", file=sys.stderr)
        sys.exit(1)

    engine = SitrepEngine()

    print("=" * 85)
    print("INICIANDO SUÍTE CONTROLADA DE SENSIBILIDADE DO JEV + QEI (4 CHAMADAS)")
    print("=" * 85)

    # 1. Consulta dados de conta / créditos antes dos testes
    print("\n[1/3] Consultando status da conta / créditos na OpenRouter...")
    account_info = check_openrouter_account_info(api_key)
    print(f"Resultado da conta: {json.dumps(account_info, indent=2)}")

    # 2. Definição dos 4 cenários rigorosos de teste com telemetria dual (corredor + notebook)
    scenarios = [
        {
            "name": "1. Passante Rápido (Trânsito Contínuo)",
            "state": {
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
                    "gaze_alignment_with_demo_pct": 10.0,
                    "head_yaw_angle_deg": 78.0,
                    "walking_velocity_mps": 1.40,
                    "crowd_density_people_per_sqm": 0.1,
                    "sustained_close_contact_seconds": 0.0,
                },
                "notebook_interaction": {
                    "user_present": False,
                    "screen_gaze_contact_pct": 0.0,
                    "satisfaction_score_pct": 0.0,
                    "attention_dwell_time_seconds": 0.0,
                    "face_distance_cm": 0.0,
                },
            },
        },
        {
            "name": "2. Lead Quente Curioso (Parado no Totem sem Vendedor)",
            "state": {
                "zone_telemetry": {
                    "location": "corredor_feira_tecnologia",
                    "booth_capacity_nominal": 15,
                    "ventilation_status": "enclosed_pavilion_low_turnover",
                },
                "metrics": {
                    "active_visitors_count": 1,
                    "corridor_passersby_count": 0,
                    "booth_visitors_retained_count": 1,
                    "max_dwell_time_seconds": 40.0,
                    "gaze_alignment_with_demo_pct": 92.0,
                    "head_yaw_angle_deg": 8.0,
                    "walking_velocity_mps": 0.04,
                    "crowd_density_people_per_sqm": 0.15,
                    "sustained_close_contact_seconds": 0.0,
                },
                "notebook_interaction": {
                    "user_present": True,
                    "screen_gaze_contact_pct": 88.0,
                    "satisfaction_score_pct": 84.0,
                    "attention_dwell_time_seconds": 38.0,
                    "face_distance_cm": 52.0,
                },
            },
        },
        {
            "name": "3. Pico de Demonstração (Múltiplos Visitantes Focados)",
            "state": {
                "zone_telemetry": {
                    "location": "corredor_feira_tecnologia",
                    "booth_capacity_nominal": 15,
                    "ventilation_status": "enclosed_pavilion_low_turnover",
                },
                "metrics": {
                    "active_visitors_count": 5,
                    "corridor_passersby_count": 1,
                    "booth_visitors_retained_count": 4,
                    "max_dwell_time_seconds": 28.0,
                    "gaze_alignment_with_demo_pct": 85.0,
                    "head_yaw_angle_deg": 14.0,
                    "walking_velocity_mps": 0.12,
                    "crowd_density_people_per_sqm": 0.5,
                    "sustained_close_contact_seconds": 15.0,
                },
                "notebook_interaction": {
                    "user_present": True,
                    "screen_gaze_contact_pct": 94.0,
                    "satisfaction_score_pct": 90.0,
                    "attention_dwell_time_seconds": 25.0,
                    "face_distance_cm": 48.0,
                },
            },
        },
        {
            "name": "4. Risco Biossegurança (Aglomeração Fechada Prolongada)",
            "state": {
                "zone_telemetry": {
                    "location": "corredor_feira_tecnologia",
                    "booth_capacity_nominal": 15,
                    "ventilation_status": "enclosed_pavilion_low_turnover",
                },
                "metrics": {
                    "active_visitors_count": 8,
                    "corridor_passersby_count": 2,
                    "booth_visitors_retained_count": 6,
                    "max_dwell_time_seconds": 65.0,
                    "gaze_alignment_with_demo_pct": 60.0,
                    "head_yaw_angle_deg": 35.0,
                    "walking_velocity_mps": 0.08,
                    "crowd_density_people_per_sqm": 2.6,
                    "sustained_close_contact_seconds": 72.0,
                },
                "notebook_interaction": {
                    "user_present": False,
                    "screen_gaze_contact_pct": 10.0,
                    "satisfaction_score_pct": 30.0,
                    "attention_dwell_time_seconds": 5.0,
                    "face_distance_cm": 95.0,
                },
            },
        },
    ]

    print("\n[2/3] Executando as 4 chamadas de inferência pontuais...")
    results = []
    summary_table = []

    for idx, sc in enumerate(scenarios, start=1):
        print(f"  -> Chamando Cenário {idx}: {sc['name']}...")
        res = run_single_decision(api_key, sc["state"], sc["name"])
        results.append(res)

        answers = res["response"].get("answers", {})
        probs = {}
        for qid in get_question_ids():
            val = answers.get(qid, {})
            p = float(val.get("noul", 0.0)) if isinstance(val, dict) else float(val or 0.0)
            probs[qid] = p

        # Calcula o QEI padrão CEIR / RetailNext para o estado
        qei_val = engine.calculate_qei(sc["state"])

        summary_table.append({
            "cenario": sc["name"],
            "rtt_ms": res["rtt_ms"],
            "qei_score": qei_val,
            "p_compra": probs.get("p_purchase_intent", 0.0),
            "p_retencao": probs.get("p_stand_retention_engagement", 0.0),
            "p_pico_demo": probs.get("p_demo_attractiveness_peak", 0.0),
            "p_risco_aereo": probs.get("p_airborne_transmission_risk", 0.0),
            "p_satisfacao": probs.get("p_user_satisfaction_engagement", 0.0),
        })

    # 3. Consulta créditos após os testes
    print("\n[3/3] Consultando status da conta / créditos após os testes...")
    account_info_after = check_openrouter_account_info(api_key)

    # 4. Gravação do Log Completo
    log_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": settings.JEV_MODEL,
        "endpoint": settings.OPENROUTER_BASE_URL,
        "account_info_before": account_info,
        "account_info_after": account_info_after,
        "summary": summary_table,
        "detailed_runs": results,
    }

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Log detalhado gravado em: {LOG_FILE}\n")

    # 5. Exibição da Tabela no Console
    print("=" * 115)
    print(f"{'CENÁRIO':<45} | {'RTT':<7} | {'QEI':<6} | {'P(Comp)':<7} | {'P(Reten)':<8} | {'P(Demo)':<7} | {'P(Risco)':<8} | {'P(Satis)':<8}")
    print("-" * 115)
    for r in summary_table:
        print(f"{r['cenario']:<45} | {r['rtt_ms']:<5.0f}ms | {r['qei_score']:<5.1f} | {r['p_compra']*100:<6.1f}% | {r['p_retencao']*100:<7.1f}% | {r['p_pico_demo']*100:<6.1f}% | {r['p_risco_aereo']*100:<7.1f}% | {r['p_satisfacao']*100:<7.1f}%")
    print("=" * 115)


if __name__ == "__main__":
    main()
