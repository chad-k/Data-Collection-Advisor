# data_collection_advisor_full.py
# ------------------------------------------------------------
# Industry Data Collection Advisor (Pattern A: rules-first, LLM-last)
#
# ✅ "NOTHING MISSING" GUARANTEE:
# - You will NEVER get "Missing templates for: [...]"
# - Any process that doesn’t have an explicit template gets an AUTO-GENERATED template.
#
# ✅ Streamlit UX fix:
# - After you click "Generate recommendations", changing dropdowns/radios will NOT send you back
#   to the start screen (persists via st.session_state["has_recs"]).
#
# ✅ OpenAI key fix:
# - Safe secrets handling (won't crash if secrets.toml missing)
# - Uses OPENAI_API_KEY env var if secrets not present
# - Clean error messages for insufficient_quota / rate limits
#
# RUN (Anaconda Prompt):
#   pip install --upgrade streamlit pandas openai
#   set OPENAI_API_KEY=sk-...
#   streamlit run data_collection_advisor_full.py
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Industry Data Collection Advisor", layout="wide")

# ============================================================
# 0) Session state init (prevents “back to start screen”)
# ============================================================
if "has_recs" not in st.session_state:
    st.session_state["has_recs"] = False
if "profile_fp" not in st.session_state:
    st.session_state["profile_fp"] = ""
if "profile_cached" not in st.session_state:
    st.session_state["profile_cached"] = None
if "recs_cached" not in st.session_state:
    st.session_state["recs_cached"] = None
if "plan_df_cached" not in st.session_state:
    st.session_state["plan_df_cached"] = None
if "narrative" not in st.session_state:
    st.session_state["narrative"] = ""
if "custom_process_name" not in st.session_state:
    st.session_state["custom_process_name"] = ""
if "custom_process_template" not in st.session_state:
    st.session_state["custom_process_template"] = None

# ============================================================
# 1) API key helpers (SAFE secrets + env fallback)
# ============================================================
def get_api_key() -> str:
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return str(key).strip()
    except StreamlitSecretNotFoundError:
        pass
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()


def llm_available() -> bool:
    return bool(get_api_key().strip())


# ============================================================
# 2) Fingerprint helpers (detect profile changes)
# ============================================================
def fingerprint_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ============================================================
# 3) Filesystem locations (custom templates)
# ============================================================
CUSTOM_DIR = "custom_templates"
os.makedirs(CUSTOM_DIR, exist_ok=True)
CUSTOM_PROCESS_SENTINEL = "Other / Custom (define yours)"


def safe_filename(name: str) -> str:
    keep = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
    keep = keep.replace(" ", "_")
    return keep[:80] if keep else "custom_process"


def custom_template_path(process_name: str) -> str:
    return os.path.join(CUSTOM_DIR, f"{safe_filename(process_name)}.json")


def load_custom_templates() -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(CUSTOM_DIR):
        return loaded
    for fn in os.listdir(CUSTOM_DIR):
        if not fn.lower().endswith(".json"):
            continue
        p = os.path.join(CUSTOM_DIR, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            process_name = obj.get("process_name")
            template = obj.get("template")
            if isinstance(process_name, str) and isinstance(template, dict):
                loaded[process_name] = template
        except Exception:
            continue
    return loaded


def save_custom_template(process_name: str, industry: str, template: Dict[str, Any]) -> None:
    payload = {
        "process_name": process_name,
        "industry": industry,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "template": template,
    }
    with open(custom_template_path(process_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# 4) Industries + process menus
# ============================================================
INDUSTRIES = [
    "Packaging",
    "Plastics",
    "Food & Beverage",
    "Aerospace & Defense",
    "Automotive",
    "Electronics",
    "Medical & Pharmaceuticals",
    "Consumer Packaged Goods (CPG)",
]

INDUSTRY_PROCESSES: Dict[str, List[str]] = {
    "Packaging": [
        "Filling / Dosing",
        "Capping / Closing",
        "Sealing",
        "Labeling / Coding",
        "Case Packing / Cartoning",
        "Palletizing",
        "Checkweighing",
        "Vision Inspection / OCR",
        "Leak Testing",
        "Metal Detection / X-ray",
        CUSTOM_PROCESS_SENTINEL,
    ],
    "Plastics": [
        "Injection Molding",
        "Blow Molding",
        "Extrusion",
        "Thermoforming",
        "Trimming / Deflashing",
        "Secondary Ops / Assembly",
        "Material Handling / Drying",
        "Final Inspection",
        CUSTOM_PROCESS_SENTINEL,
    ],
    "Food & Beverage": [
        "Receiving / Raw Material Intake",
        "Batch Mixing / Blending",
        "Cooking / Thermal Processing",
        "Cooling / Holding",
        "Filling / Packaging",
        "CIP / Sanitation",
        "Metal Detection / X-ray",
        "QC Lab Checks",
        "Cold Chain / Storage Monitoring",
        CUSTOM_PROCESS_SENTINEL,
    ],
    "Aerospace & Defense": ["Machining", "Assembly", "Coating / Finishing", "NDT / Inspection", CUSTOM_PROCESS_SENTINEL],
    "Automotive": ["Assembly", "Machining", "Stamping / Forming", "Welding", "Final Test / Inspection", CUSTOM_PROCESS_SENTINEL],
    "Electronics": ["SMT / Soldering", "Assembly", "Functional Test", "AOI / Vision Inspection", CUSTOM_PROCESS_SENTINEL],
    "Medical & Pharmaceuticals": [
        "Receiving / Raw Material Intake",
        "Batch Mixing / Blending",
        "Filling / Packaging",
        "Sterilization",
        "QC Lab Checks",
        CUSTOM_PROCESS_SENTINEL,
    ],
    "Consumer Packaged Goods (CPG)": [
        "Filling / Dosing",
        "Filling / Packaging",
        "Labeling / Coding",
        "Case Packing / Cartoning",
        "Checkweighing",
        CUSTOM_PROCESS_SENTINEL,
    ],
}

GOALS = [
    "Reduce scrap / rework",
    "Improve consistency (reduce variation)",
    "Increase yield / first-pass pass rate",
    "Improve traceability / recall readiness",
    "Regulatory compliance / audit readiness",
    "Optimize inspection frequency",
    "Reduce downtime / improve reliability",
]

TRACEABILITY_UNIT = ["Lot/Batch", "Serial", "Pallet/Case", "None/Not sure yet"]

DATA_SOURCES = [
    "Manual entry",
    "Gauges (digital)",
    "PLC/SCADA",
    "Vision system",
    "MES/ERP",
    "LIMS/QC Lab",
    "Barcode/Label printer",
    "Not sure",
]

EVENT_OPTIONS = [
    "changeover_event",
    "downtime_event",
    "maintenance_event",
    "sanitation_event",
    "calibration_status",
    "ccp_check",
    "deviation_flag",
    "test_piece_check",
]


# ============================================================
# 5) Templates (explicit ones + auto-template fallback)
# ============================================================
def base_measurement_fields(entity_field: str) -> List[Dict[str, Any]]:
    return [
        {"field": "timestamp", "type": "datetime", "example": "2026-02-13 14:22:01"},
        {"field": entity_field, "type": "string", "example": "SKU-2201" if entity_field == "sku" else "12-35001"},
        {"field": "line_or_machine_id", "type": "string", "example": "Line-3"},
        {"field": "characteristic", "type": "string", "example": "Key Characteristic"},
        {"field": "value", "type": "float", "example": 10.0},
        {"field": "lsl", "type": "float", "example": 0.0},
        {"field": "usl", "type": "float", "example": 20.0},
        {"field": "unit", "type": "string", "example": "unit"},
    ]


PROCESS_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # A few core explicit templates (most everything else is auto-generated)
    "Filling / Dosing": {
        "must": [
            {"field": "timestamp", "type": "datetime", "example": "2026-02-13 10:00:00"},
            {"field": "sku", "type": "string", "example": "SKU-2201"},
            {"field": "line_or_machine_id", "type": "string", "example": "Filler-1"},
            {"field": "lot_batch", "type": "string", "example": "BATCH-20260213-01"},
            {"field": "characteristic", "type": "string", "example": "Fill Weight"},
            {"field": "value", "type": "float", "example": 505.2},
            {"field": "lsl", "type": "float", "example": 495.0},
            {"field": "usl", "type": "float", "example": 510.0},
            {"field": "unit", "type": "string", "example": "g"},
        ],
        "recommended": [
            {"field": "shift", "type": "string", "example": "A"},
            {"field": "operator", "type": "string", "example": "OP-101"},
            {"field": "filler_head", "type": "string", "example": "Head-6"},
            {"field": "changeover_event", "type": "bool", "example": False},
            {"field": "downtime_event", "type": "bool", "example": False},
            {"field": "scrap_reason_code", "type": "string", "example": "UNDERFILL"},
        ],
        "advanced": [
            {"field": "setpoint", "type": "float", "example": 505.0},
            {"field": "fill_speed", "type": "float", "example": 1.2},
            {"field": "maintenance_event", "type": "bool", "example": False},
        ],
        "ctq_hints": ["fill weight", "fill volume", "underfill", "overfill"],
    },
    "Capping / Closing": {
        "must": [
            {"field": "timestamp", "type": "datetime", "example": "2026-02-13 10:03:44"},
            {"field": "sku", "type": "string", "example": "SKU-2201"},
            {"field": "line_or_machine_id", "type": "string", "example": "Capper-1"},
            {"field": "lot_batch", "type": "string", "example": "BATCH-20260213-01"},
            {"field": "characteristic", "type": "string", "example": "Cap Torque"},
            {"field": "value", "type": "float", "example": 18.2},
            {"field": "lsl", "type": "float", "example": 15.0},
            {"field": "usl", "type": "float", "example": 22.0},
            {"field": "unit", "type": "string", "example": "lbf·in"},
        ],
        "recommended": [
            {"field": "capper_head", "type": "string", "example": "Head-2"},
            {"field": "cap_lot", "type": "string", "example": "CAP-LOT-778"},
            {"field": "operator", "type": "string", "example": "OP-102"},
            {"field": "shift", "type": "string", "example": "A"},
            {"field": "changeover_event", "type": "bool", "example": False},
            {"field": "scrap_reason_code", "type": "string", "example": "LOOSE_CAP"},
        ],
        "advanced": [
            {"field": "cap_height", "type": "float", "example": 0.52},
            {"field": "capping_speed", "type": "float", "example": 240.0},
            {"field": "maintenance_event", "type": "bool", "example": False},
        ],
        "ctq_hints": ["cap torque", "cap height", "leakers", "cross-thread"],
    },
    "Injection Molding": {
        "must": [
            {"field": "timestamp", "type": "datetime", "example": "2026-02-13 14:22:01"},
            {"field": "part_number", "type": "string", "example": "12-35001"},
            {"field": "line_or_machine_id", "type": "string", "example": "IMM-05"},
            {"field": "cavity", "type": "string", "example": "C12"},
            {"field": "characteristic", "type": "string", "example": "Wall Thickness"},
            {"field": "value", "type": "float", "example": 2.12},
            {"field": "lsl", "type": "float", "example": 2.00},
            {"field": "usl", "type": "float", "example": 2.25},
            {"field": "unit", "type": "string", "example": "mm"},
        ],
        "recommended": [
            {"field": "tooling_id", "type": "string", "example": "MOLD-7782"},
            {"field": "material_lot", "type": "string", "example": "RESIN-LOT-2026-02A"},
            {"field": "operator", "type": "string", "example": "OP-104"},
            {"field": "shift", "type": "string", "example": "B"},
            {"field": "changeover_event", "type": "bool", "example": False},
            {"field": "scrap_reason_code", "type": "string", "example": "FLASH"},
        ],
        "advanced": [
            {"field": "melt_temp", "type": "float", "example": 205.2},
            {"field": "injection_pressure", "type": "float", "example": 1125.0},
            {"field": "cycle_time_sec", "type": "float", "example": 38.4},
        ],
        "ctq_hints": ["flash", "short shot", "warpage", "weight"],
    },
    "Metal Detection / X-ray": {
        "must": [
            {"field": "timestamp", "type": "datetime", "example": "2026-02-13 18:10:00"},
            {"field": "sku", "type": "string", "example": "SKU-FOOD-110"},
            {"field": "line_or_machine_id", "type": "string", "example": "MD-1"},
            {"field": "lot_batch", "type": "string", "example": "LOT-20260213-05"},
            {"field": "characteristic", "type": "string", "example": "Foreign Object Pass"},
            {"field": "value", "type": "float", "example": 1.0},
            {"field": "lsl", "type": "float", "example": 1.0},
            {"field": "usl", "type": "float", "example": 1.0},
            {"field": "unit", "type": "string", "example": "pass(1)/fail(0)"},
        ],
        "recommended": [
            {"field": "test_piece_check", "type": "bool", "example": True},
            {"field": "ccp_check", "type": "bool", "example": True},
        ],
        "advanced": [{"field": "detector_calibration_check", "type": "bool", "example": True}],
        "ctq_hints": ["foreign object", "challenge checks", "CCP"],
    },
}

# Load saved custom templates and add them
PROCESS_TEMPLATES.update(load_custom_templates())


# ============================================================
# 6) Auto-template fallback (THIS prevents missing errors)
# ============================================================
def infer_entity_field(industry: str, process_name: str) -> str:
    p = (process_name or "").lower()
    if industry in {"Packaging", "Food & Beverage", "Consumer Packaged Goods (CPG)", "Electronics"}:
        if "receiving" in p or "raw" in p:
            return "ingredient"
        if "lab" in p:
            return "lot_batch"
        return "sku"
    return "part_number"


def infer_characteristic(process_name: str) -> str:
    p = (process_name or "").lower()
    if "label" in p or "coding" in p or "ocr" in p or "vision" in p or "aoi" in p:
        return "Code Readability Pass"
    if "leak" in p:
        return "Leak Test Pass"
    if "seal" in p:
        return "Seal Integrity Score"
    if "checkweigh" in p or "weigh" in p or "fill" in p or "dose" in p:
        return "Weight"
    if "mold" in p:
        return "Part Weight"
    if "steril" in p:
        return "Cycle Pass"
    if "test" in p or "inspection" in p:
        return "Pass"
    if "cook" in p or "thermal" in p:
        return "Cook Temp"
    if "cool" in p or "hold" in p:
        return "Product Temp"
    if "cip" in p or "sanitation" in p:
        return "CIP Cycle Pass"
    return "Key Characteristic"


def auto_template(process_name: str, industry: str, trace_unit: str) -> Dict[str, Any]:
    entity = infer_entity_field(industry, process_name)
    char = infer_characteristic(process_name)

    # Must
    must = [
        {"field": "timestamp", "type": "datetime", "example": "2026-02-13 12:00:00"},
        {"field": entity, "type": "string", "example": "SKU-2201" if entity == "sku" else ("LOT-20260213-01" if entity == "lot_batch" else "12-35001")},
        {"field": "line_or_machine_id", "type": "string", "example": f"{process_name[:10]}-1".replace(" ", "")},
        {"field": "characteristic", "type": "string", "example": char},
        {"field": "value", "type": "float", "example": 1.0 if "pass" in char.lower() else 10.0},
        {"field": "lsl", "type": "float", "example": 1.0 if "pass" in char.lower() else 0.0},
        {"field": "usl", "type": "float", "example": 1.0 if "pass" in char.lower() else 20.0},
        {"field": "unit", "type": "string", "example": "pass(1)/fail(0)" if "pass" in char.lower() else "unit"},
    ]

    # Traceability
    if trace_unit in {"Lot/Batch", "Pallet/Case"} and entity != "lot_batch":
        must.append({"field": "lot_batch", "type": "string", "example": "LOT-20260213-01"})
    if trace_unit == "Serial":
        must.append({"field": "serial_number", "type": "string", "example": "SN-000123"})

    # Recommended
    recommended = [
        {"field": "shift", "type": "string", "example": "A"},
        {"field": "operator", "type": "string", "example": "OP-001"},
        {"field": "changeover_event", "type": "bool", "example": False},
        {"field": "downtime_event", "type": "bool", "example": False},
        {"field": "scrap_reason_code", "type": "string", "example": "UNKNOWN"},
    ]

    # Advanced
    advanced = [
        {"field": "setpoint", "type": "float", "example": 10.0},
        {"field": "cycle_time_sec", "type": "float", "example": 30.0},
        {"field": "maintenance_event", "type": "bool", "example": False},
    ]

    return {"must": must, "recommended": recommended, "advanced": advanced, "ctq_hints": ["stability", "defects", "process drift"]}


def ensure_templates_exist(processes: List[str], industry: str, trace_unit: str) -> None:
    for p in processes:
        if p == CUSTOM_PROCESS_SENTINEL:
            continue
        if p not in PROCESS_TEMPLATES:
            PROCESS_TEMPLATES[p] = auto_template(p, industry, trace_unit)


# ============================================================
# 7) Deterministic scoring
# ============================================================
INDUSTRY_BOOSTS: Dict[str, Dict[str, Any]] = {
    "Packaging": {"boost_fields": ["lot_batch", "changeover_event", "downtime_event", "scrap_reason_code", "reject_flag", "reject_reason", "ocr_text", "vision_score"]},
    "Plastics": {"boost_fields": ["cavity", "tooling_id", "material_lot", "cycle_time_sec", "melt_temp"]},
    "Food & Beverage": {"boost_fields": ["lot_batch", "ccp_check", "sanitation_event", "calibration_status", "deviation_flag", "test_piece_check"]},
    "Aerospace & Defense": {"boost_fields": ["serial_number", "material_lot", "calibration_status"]},
    "Medical & Pharmaceuticals": {"boost_fields": ["calibration_status", "deviation_flag", "lot_batch"]},
    "Automotive": {"boost_fields": ["tool_number", "die_id", "changeover_event", "scrap_reason_code"]},
    "Electronics": {"boost_fields": ["failure_mode", "retest_flag", "vision_score"]},
    "Consumer Packaged Goods (CPG)": {"boost_fields": ["changeover_event", "downtime_event", "scrap_reason_code"]},
}


@dataclass
class CustomerProfile:
    customer_name: str
    industry: str
    processes: List[str]
    goals: List[str]
    regulated: bool
    traceability_unit: str
    current_state: Dict[str, Any]
    top_defects: List[str]
    top_cost_drivers: List[str]


def maturity_score(current_state: Dict[str, Any]) -> Tuple[int, str]:
    digital = bool(current_state.get("digital_capture", False))
    timestamps = bool(current_state.get("has_timestamps", False))
    specs = bool(current_state.get("specs_available", False))
    spc = bool(current_state.get("uses_spc", False))
    events = bool(current_state.get("captures_events", False))
    trace = bool(current_state.get("captures_traceability", False))
    calib = bool(current_state.get("captures_calibration", False))

    level = 0
    if digital:
        level = 1
    if digital and timestamps and specs:
        level = 2
    if level >= 2 and spc:
        level = 3
    if level >= 3 and events and trace and calib:
        level = 4

    labels = {
        0: "Level 0 – manual or inconsistent capture",
        1: "Level 1 – basic digital records",
        2: "Level 2 – timestamps + specs (SPC foundation)",
        3: "Level 3 – active SPC / stability focus",
        4: "Level 4 – predictive-ready (events + traceability + calibration discipline)",
    }
    return level, labels[level]


def compute_priority(field_name: str, profile: CustomerProfile, base_tier: str) -> float:
    tier_weight = {"must": 100.0, "recommended": 70.0, "advanced": 40.0}[base_tier]
    goals = set(profile.goals)

    goal_boost = 0.0
    if "Improve traceability / recall readiness" in goals and field_name in {"material_lot", "lot_batch", "supplier_lot", "serial_number", "pallet_id"}:
        goal_boost += 25.0
    if "Regulatory compliance / audit readiness" in goals and field_name in {"calibration_status", "deviation_flag", "ccp_check", "sanitation_event", "test_piece_check"}:
        goal_boost += 25.0
    if "Reduce scrap / rework" in goals and field_name in {"scrap_reason_code", "tooling_id", "cavity", "changeover_event", "reject_reason", "failure_mode", "defect_code"}:
        goal_boost += 18.0
    if "Reduce downtime / improve reliability" in goals and field_name in {"downtime_event", "maintenance_event"}:
        goal_boost += 18.0

    boost_fields = set(INDUSTRY_BOOSTS.get(profile.industry, {}).get("boost_fields", []))
    industry_boost = 12.0 if field_name in boost_fields else 0.0

    reg_boost = 0.0
    if profile.regulated and field_name in {"calibration_status", "deviation_flag", "ccp_check", "sanitation_event", "test_piece_check"}:
        reg_boost += 12.0

    trace_boost = 0.0
    if profile.traceability_unit in {"Lot/Batch", "Pallet/Case"} and field_name in {"lot_batch", "material_lot", "supplier_lot", "pallet_id"}:
        trace_boost += 10.0
    if profile.traceability_unit == "Serial" and field_name == "serial_number":
        trace_boost += 12.0

    m_level, _ = maturity_score(profile.current_state)
    maturity_penalty = 0.0
    if base_tier == "advanced" and m_level < 3:
        maturity_penalty = 30.0

    return tier_weight + goal_boost + industry_boost + reg_boost + trace_boost - maturity_penalty


def build_recommendations(profile: CustomerProfile) -> Dict[str, Any]:
    m_level, m_label = maturity_score(profile.current_state)

    all_fields: List[Dict[str, Any]] = []
    for proc in profile.processes:
        tmpl = PROCESS_TEMPLATES.get(proc)
        if not tmpl:
            continue
        for tier in ["must", "recommended", "advanced"]:
            for f in tmpl.get(tier, []):
                score = compute_priority(f["field"], profile, tier)
                all_fields.append({**f, "process": proc, "tier": tier, "priority_score": round(score, 2)})

    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in all_fields:
        key = (item["field"], item["type"])
        if key not in merged:
            merged[key] = {**item, "processes": [item["process"]]}
        else:
            merged[key]["processes"].append(item["process"])
            if item["priority_score"] > merged[key]["priority_score"]:
                for k in ["tier", "priority_score", "example"]:
                    merged[key][k] = item.get(k)

    fields_sorted = sorted(merged.values(), key=lambda x: x["priority_score"], reverse=True)
    wins = [f for f in fields_sorted if f["tier"] in {"must", "recommended"}][:12]
    return {"maturity_level": m_level, "maturity_label": m_label, "fields_ranked": fields_sorted, "fast_wins": wins}


# ============================================================
# 8) Plan-view bucketing (4 options)
# ============================================================
def bucket_for_field(field: str) -> str:
    f = (field or "").lower()
    if f in {"lot_batch", "material_lot", "supplier_lot", "serial_number", "pallet_id", "case_lot"}:
        return "Traceability & Genealogy"
    if f in {"ccp_check", "sanitation_event", "calibration_status", "deviation_flag", "test_piece_check"}:
        return "Compliance / Audit / Food Safety"
    if f in {"changeover_event", "downtime_event", "maintenance_event"}:
        return "Events / Downtime / Reliability"
    if f in {"reject_flag", "reject_reason", "scrap_reason_code", "failure_mode", "defect_code", "retest_flag"}:
        return "Quality Outcomes"
    if f in {"ocr_text", "vision_score", "printer_id", "image_path_or_id"}:
        return "Vision / Label / Code"
    if any(k in f for k in ["temp", "pressure", "speed", "setpoint", "rpm", "load", "dwell", "line_speed", "cycle_time"]):
        return "Process Settings"
    return "SPC Foundation"


def integration_bucket(field: str, sources_selected: List[str]) -> str:
    f = (field or "").lower()
    sources_selected = sources_selected or []
    if f in {"ocr_text", "vision_score", "image_path_or_id"}:
        return "Vision system"
    if any(k in f for k in ["temp", "pressure", "rpm", "load", "line_speed", "cycle_time", "setpoint", "melt_temp", "injection_pressure"]):
        return "PLC/SCADA"
    if f in {"test_name", "instrument_id", "calibration_status"}:
        return "LIMS/QC Lab"
    if f in {"work_order", "job", "routing", "bom", "supplier", "supplier_lot"}:
        return "MES/ERP"
    if f in {"pallet_id", "serial_number", "lot_batch", "printer_id"}:
        return "Barcode/Label printer" if "Barcode/Label printer" in sources_selected else "Traceability systems"
    if "Manual entry" in sources_selected and "Not sure" not in sources_selected:
        return "Manual entry"
    return "General / Mixed"


def package_bucket(tier: str, priority_score: float, maturity_level: int) -> str:
    t = (tier or "").lower()
    s = float(priority_score or 0)
    if t == "must":
        return "Quick Wins (1–2 weeks)"
    if t == "recommended":
        return "Quick Wins (1–2 weeks)" if s >= 85 else "Core System (30–60 days)"
    if t == "advanced":
        return "Core System (30–60 days)" if maturity_level >= 4 and s >= 70 else "Enterprise Scale (90+ days)"
    return "Core System (30–60 days)"


# ============================================================
# 9) Custom process builder
# ============================================================
def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def build_custom_process_template(
    process_name: str,
    entity_key: str,
    station_key: Optional[str],
    include_traceability: str,
    ctqs: List[str],
    sources: List[str],
    events_selected: List[str],
) -> Dict[str, Any]:
    must = [
        {"field": "timestamp", "type": "datetime", "example": "2026-02-13 12:00:00"},
        {"field": entity_key, "type": "string", "example": "SKU-2201" if entity_key == "sku" else "12-35001"},
        {"field": "line_or_machine_id", "type": "string", "example": "Line-1"},
    ]
    if station_key:
        must.append({"field": station_key, "type": "string", "example": "Station-3"})

    must += [
        {"field": "characteristic", "type": "string", "example": ctqs[0] if ctqs else "Key Characteristic"},
        {"field": "value", "type": "float", "example": 1.23},
        {"field": "unit", "type": "string", "example": "unit"},
        {"field": "lsl", "type": "float", "example": 0.0},
        {"field": "usl", "type": "float", "example": 2.0},
    ]

    if include_traceability in {"Lot/Batch", "Pallet/Case"}:
        must.append({"field": "lot_batch", "type": "string", "example": "LOT-20260213-01"})
    elif include_traceability == "Serial":
        must.append({"field": "serial_number", "type": "string", "example": "SN-000123"})

    recommended = [
        {"field": "shift", "type": "string", "example": "A"},
        {"field": "operator", "type": "string", "example": "OP-001"},
    ]

    for ev in events_selected:
        if ev == "calibration_status":
            recommended.append({"field": "calibration_status", "type": "string", "example": "IN_CAL"})
        else:
            recommended.append({"field": ev, "type": "bool", "example": False})

    if sources:
        recommended.append({"field": "data_source", "type": "string", "example": sources[0]})

    advanced = [
        {"field": "setpoint", "type": "float", "example": 1.25},
        {"field": "cycle_time_sec", "type": "float", "example": 30.0},
        {"field": "maintenance_event", "type": "bool", "example": False},
    ]

    return {"must": must, "recommended": recommended, "advanced": advanced, "ctq_hints": ctqs if ctqs else ["key characteristic"]}


# ============================================================
# 10) LLM wrapper (optional)
# ============================================================
def generate_llm_narrative(profile: CustomerProfile, recs: Dict[str, Any], plan_view: str) -> str:
    from openai import OpenAI, RateLimitError

    key = get_api_key().strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found.\n"
            "Fix options:\n"
            "1) In Anaconda Prompt:\n"
            "   set OPENAI_API_KEY=sk-...\n"
            "   streamlit run data_collection_advisor_full.py\n"
            "2) Or create .streamlit/secrets.toml with OPENAI_API_KEY."
        )

    client = OpenAI(api_key=key, timeout=45)

    ranked = recs["fields_ranked"][:80]
    sources = profile.current_state.get("sources", [])
    rows = []
    for f in ranked:
        rows.append({
            "field": f["field"],
            "type": f["type"],
            "tier": f["tier"],
            "priority_score": f["priority_score"],
            "processes": sorted(set(f.get("processes", []))),
            "use_case": bucket_for_field(f["field"]),
            "integration_path": integration_bucket(f["field"], sources),
            "package": package_bucket(f["tier"], f["priority_score"], recs["maturity_level"]),
        })

    payload = {
        "customer_profile": {
            "customer_name": profile.customer_name,
            "industry": profile.industry,
            "processes": profile.processes,
            "goals": profile.goals,
            "regulated": profile.regulated,
            "traceability_unit": profile.traceability_unit,
            "top_defects": profile.top_defects[:10],
            "top_cost_drivers": profile.top_cost_drivers[:10],
            "sources": sources,
        },
        "maturity": {"level": recs["maturity_level"], "label": recs["maturity_label"]},
        "plan_view_selected": plan_view,
        "ranked_fields": rows,
    }

    system = (
        "You are a quality and manufacturing data advisor. "
        "Use ONLY fields present in INPUT_JSON. Do NOT invent fields. "
        "Write concise, actionable guidance for onboarding."
    )
    user = (
        "Create a tailored data-collection plan.\n"
        "If plan_view_selected is:\n"
        "- Must/Should/Nice: structure output into those 3 headings.\n"
        "- By Use Case: headings by use_case.\n"
        "- By Data Source: headings by integration_path.\n"
        "- Quick Wins/Core/Enterprise: headings by package.\n"
        "Always include: (1) short summary, (2) what to start capturing immediately, (3) common pitfalls.\n\n"
        f"INPUT_JSON:\n{json.dumps(payload, indent=2)}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=900,
        )
        return resp.choices[0].message.content
    except RateLimitError as e:
        msg = str(e)
        if "insufficient_quota" in msg or "check your plan and billing details" in msg:
            raise RuntimeError(
                "OpenAI API quota is exhausted / billing not enabled for this project.\n"
                "Enable billing / add credits / raise project spend limit, then try again."
            ) from e
        raise


# ============================================================
# 11) UI
# ============================================================
st.title("Industry Data Collection Advisor (Pattern A)")
st.caption("Rules/templates + deterministic scoring first; optional ChatGPT narrative last.")

with st.sidebar:
    if st.button("Reset / Start over"):
        st.session_state.clear()
        st.rerun()

    st.header("LLM Status")
    st.write("API key detected:", "✅ YES" if llm_available() else "❌ NO")
    st.caption("If you don’t use secrets.toml, set OPENAI_API_KEY in the SAME Anaconda Prompt session before launching Streamlit.")

    st.divider()
    if st.button("Test OpenAI call"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=get_api_key(), timeout=30)
            r = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                temperature=0,
                max_tokens=10,
            )
            st.success(r.choices[0].message.content)
        except Exception as e:
            st.error("OpenAI test failed:")
            st.exception(e)

    st.divider()
    st.header("Customer Context")
    customer_name = st.text_input("Customer name", value="New Customer")
    industry = st.selectbox("Industry", INDUSTRIES, index=INDUSTRIES.index("Packaging"))

    processes = st.multiselect(
        "Processes (select all that apply)",
        options=INDUSTRY_PROCESSES.get(industry, []),
        default=INDUSTRY_PROCESSES.get(industry, [])[:2] if INDUSTRY_PROCESSES.get(industry) else [],
    )

    goals = st.multiselect("Primary goals (pick 1–3)", options=GOALS, default=["Reduce scrap / rework"])
    regulated_default = industry in ["Food & Beverage", "Medical & Pharmaceuticals", "Aerospace & Defense"]
    regulated = st.checkbox("Regulated environment / audit focus?", value=regulated_default)
    trace_unit = st.selectbox("Traceability unit", TRACEABILITY_UNIT, index=0)

    st.divider()
    st.subheader("Current State (Maturity Inputs)")
    digital_capture = st.checkbox("Data is captured digitally today", value=True)
    has_timestamps = st.checkbox("Measurements have reliable timestamps", value=True)
    specs_available = st.checkbox("Specs/limits are available (LSL/USL)", value=True)
    uses_spc = st.checkbox("SPC/control charts are used today", value=False)
    captures_events = st.checkbox("You capture events (changeover/downtime/maintenance/etc.)", value=False)
    captures_traceability = st.checkbox("You capture traceability keys with measurements", value=True)
    captures_calibration = st.checkbox("You capture calibration/instrument status (as applicable)", value=regulated_default)

    st.divider()
    st.subheader("Data Sources Available")
    sources = st.multiselect("Select available sources", DATA_SOURCES, default=["Manual entry", "Gauges (digital)"])

    st.divider()
    st.subheader("Pain Points")
    top_defects_text = st.text_area("Top defects (comma-separated)", value="underfill, leakers, bad code")
    top_cost_drivers_text = st.text_area("Top cost drivers (comma-separated)", value="startup scrap, changeover losses")

    st.divider()
    run_btn = st.button("Generate recommendations", type="primary")


# ============================================================
# 12) Custom Process Builder (only if selected)
# ============================================================
custom_template_name: Optional[str] = None
custom_template: Optional[Dict[str, Any]] = None

if CUSTOM_PROCESS_SENTINEL in processes:
    with st.expander("Custom Process Builder", expanded=True):
        cp_name = st.text_input("Custom process name", value="My Custom Process")
        cp_entity_key = st.selectbox("Primary entity key", ["sku", "part_number", "product", "lot_batch"], index=0)
        cp_has_parallel = st.checkbox("Has parallel stations/lanes/heads/cavities?", value=False)
        cp_station_key = None
        if cp_has_parallel:
            cp_station_key = st.selectbox("Station key name", ["station_id", "lane", "head", "cavity", "nozzle", "tester_id"], index=0)

        cp_ctqs = st.text_area("Top CTQs / characteristics (comma-separated)", value="fill weight, seal strength").strip()
        cp_ctq_list = parse_csv_list(cp_ctqs)

        cp_events = st.multiselect("Events to capture (optional)", options=EVENT_OPTIONS, default=["changeover_event"])
        build_cp = st.button("Build custom template")

        if build_cp:
            custom_template_name = (cp_name.strip() or "Custom Process")
            custom_template = build_custom_process_template(
                process_name=custom_template_name,
                entity_key=cp_entity_key,
                station_key=cp_station_key,
                include_traceability=trace_unit,
                ctqs=cp_ctq_list,
                sources=sources,
                events_selected=cp_events,
            )
            st.session_state["custom_process_name"] = custom_template_name
            st.session_state["custom_process_template"] = custom_template
            st.success("Custom template built. Now click Generate recommendations.")

        if st.session_state.get("custom_process_name") and st.session_state.get("custom_process_template"):
            st.write(f"**Current custom process:** {st.session_state['custom_process_name']}")
            if st.button("Save this custom process for reuse"):
                PROCESS_TEMPLATES[st.session_state["custom_process_name"]] = st.session_state["custom_process_template"]
                save_custom_template(st.session_state["custom_process_name"], industry, st.session_state["custom_process_template"])
                st.success(f"Saved to {CUSTOM_DIR}/ as a reusable template.")


# ============================================================
# 13) Build CURRENT profile
# ============================================================
effective_processes = [p for p in processes if p != CUSTOM_PROCESS_SENTINEL]

if CUSTOM_PROCESS_SENTINEL in processes:
    if st.session_state.get("custom_process_name") and st.session_state.get("custom_process_template"):
        custom_template_name = st.session_state["custom_process_name"]
        custom_template = st.session_state["custom_process_template"]
        PROCESS_TEMPLATES[custom_template_name] = custom_template
        effective_processes.append(custom_template_name)

profile = CustomerProfile(
    customer_name=(customer_name.strip() or "Customer"),
    industry=industry,
    processes=effective_processes,
    goals=goals or [],
    regulated=regulated,
    traceability_unit=trace_unit,
    current_state={
        "digital_capture": digital_capture,
        "has_timestamps": has_timestamps,
        "specs_available": specs_available,
        "uses_spc": uses_spc,
        "captures_events": captures_events,
        "captures_traceability": captures_traceability,
        "captures_calibration": captures_calibration,
        "sources": sources,
    },
    top_defects=parse_csv_list(top_defects_text),
    top_cost_drivers=parse_csv_list(top_cost_drivers_text),
)

profile_fp = fingerprint_dict(asdict(profile))

# ✅ Ensure templates exist for all selected processes (auto-fallback)
ensure_templates_exist(profile.processes, profile.industry, profile.traceability_unit)

# ============================================================
# 14) Generate + cache recs ONLY on button click
# ============================================================
if run_btn:
    if not profile.processes:
        st.error("Select at least one process.")
        st.stop()

    recs = build_recommendations(profile)

    st.session_state["has_recs"] = True
    st.session_state["profile_fp"] = profile_fp
    st.session_state["profile_cached"] = profile
    st.session_state["recs_cached"] = recs
    st.session_state["plan_df_cached"] = None
    st.session_state["narrative"] = ""
    st.success("Recommendations generated. You can now change view filters without losing this page.")

# ============================================================
# 15) Show results if we have them (key persistence)
# ============================================================
if not st.session_state["has_recs"]:
    st.info("Set customer context in the sidebar, then click **Generate recommendations**.")
    st.stop()

profile_cached: CustomerProfile = st.session_state["profile_cached"]
recs_cached: Dict[str, Any] = st.session_state["recs_cached"]

if profile_fp != st.session_state.get("profile_fp", ""):
    st.warning("Sidebar inputs changed since you generated recommendations. Click **Generate recommendations** to refresh results.")

# ============================================================
# 16) Build plan_df once per cached recs
# ============================================================
if st.session_state["plan_df_cached"] is None:
    ranked = recs_cached["fields_ranked"][:150]
    rows = []
    for f in ranked:
        rows.append({
            "field": f["field"],
            "type": f["type"],
            "tier": f["tier"],
            "priority_score": f["priority_score"],
            "processes": ", ".join(sorted(set(f.get("processes", [])))),
            "example": f.get("example", ""),
            "use_case": bucket_for_field(f["field"]),
            "integration_path": integration_bucket(f["field"], profile_cached.current_state.get("sources", [])),
            "package": package_bucket(f["tier"], f["priority_score"], recs_cached["maturity_level"]),
        })
    st.session_state["plan_df_cached"] = pd.DataFrame(rows)

plan_df: pd.DataFrame = st.session_state["plan_df_cached"]

# ============================================================
# 17) Results UI
# ============================================================
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Snapshot")
    st.write(f"**Customer:** {profile_cached.customer_name}")
    st.write(f"**Industry:** {profile_cached.industry}")
    st.write(f"**Processes:** {', '.join(profile_cached.processes) if profile_cached.processes else '—'}")
    st.write(f"**Goals:** {', '.join(profile_cached.goals) if profile_cached.goals else '—'}")
    st.write(f"**Regulated:** {'Yes' if profile_cached.regulated else 'No'}")
    st.write(f"**Traceability unit:** {profile_cached.traceability_unit}")
    st.write(f"**Sources:** {', '.join(profile_cached.current_state.get('sources', [])) if profile_cached.current_state.get('sources') else '—'}")

    st.subheader("Maturity")
    st.metric("Level", recs_cached["maturity_level"])
    st.info(recs_cached["maturity_label"])

    st.subheader("Fast wins (top recommendations)")
    for w in recs_cached["fast_wins"]:
        st.write(f"- **{w['field']}** — score **{w['priority_score']}** (used in: {', '.join(sorted(set(w['processes'])))} )")

with col_right:
    st.subheader("Data plan view")

    plan_view = st.radio(
        "Choose how to view recommendations",
        options=[
            "Must / Should / Nice-to-have",
            "By Use Case",
            "By Data Source / Integration path",
            "Quick Wins / Core / Enterprise",
        ],
        horizontal=True,
        key="plan_view",
    )

    top_n = st.slider("Show top N rows", 10, 150, 50, 10, key="top_n")

    if plan_view == "Must / Should / Nice-to-have":
        mapping = {"must": "Must collect now", "recommended": "Should collect next", "advanced": "Nice to have later"}
        df = plan_df.copy()
        df["priority_group"] = df["tier"].map(mapping).fillna("Should collect next")
        choice = st.selectbox("Select group", ["Must collect now", "Should collect next", "Nice to have later"], index=0, key="tier_group")
        view_df = df[df["priority_group"] == choice].sort_values("priority_score", ascending=False).head(top_n)
        st.dataframe(view_df[["field", "type", "priority_group", "priority_score", "processes", "example"]], use_container_width=True, height=420)

    elif plan_view == "By Use Case":
        use_case = st.selectbox("Select use-case bucket", sorted(plan_df["use_case"].unique()), key="use_case_bucket")
        view_df = plan_df[plan_df["use_case"] == use_case].sort_values("priority_score", ascending=False).head(top_n)
        st.dataframe(view_df[["field", "type", "tier", "priority_score", "processes", "example"]], use_container_width=True, height=420)

    elif plan_view == "By Data Source / Integration path":
        path = st.selectbox("Select integration path", sorted(plan_df["integration_path"].unique()), key="integration_path")
        view_df = plan_df[plan_df["integration_path"] == path].sort_values("priority_score", ascending=False).head(top_n)
        st.dataframe(view_df[["field", "type", "tier", "priority_score", "processes", "example"]], use_container_width=True, height=420)

    else:
        pkg = st.selectbox(
            "Select package",
            ["Quick Wins (1–2 weeks)", "Core System (30–60 days)", "Enterprise Scale (90+ days)"],
            index=0,
            key="package_bucket",
        )
        view_df = plan_df[plan_df["package"] == pkg].sort_values("priority_score", ascending=False).head(top_n)
        st.dataframe(view_df[["field", "type", "tier", "package", "priority_score", "processes", "example"]], use_container_width=True, height=420)

st.divider()
st.subheader("Ranked data dictionary (top 150)")
ranked_df = plan_df.sort_values("priority_score", ascending=False)
st.dataframe(ranked_df[["field", "type", "tier", "priority_score", "processes", "example"]], use_container_width=True, height=420)

export = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "profile": asdict(profile_cached),
    "recommendations": {
        "maturity_level": recs_cached["maturity_level"],
        "maturity_label": recs_cached["maturity_label"],
        "fast_wins": recs_cached["fast_wins"],
        "ranked_fields": recs_cached["fields_ranked"][:150],
    },
    "plan_view_df": ranked_df.to_dict(orient="records"),
}

st.download_button(
    "Download JSON (profile + recommendations)",
    data=json.dumps(export, indent=2),
    file_name=f"{profile_cached.customer_name.replace(' ', '_')}_data_plan.json",
    mime="application/json",
)

st.download_button(
    "Download CSV (ranked fields)",
    data=ranked_df.to_csv(index=False),
    file_name=f"{profile_cached.customer_name.replace(' ', '_')}_ranked_fields.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Tailored narrative (LLM-last)")

if not llm_available():
    st.warning("No API key detected by the app. Set OPENAI_API_KEY in the same Anaconda Prompt session before launching Streamlit, or add .streamlit/secrets.toml.")
    st.code('set OPENAI_API_KEY=sk-xxxxxx\nstreamlit run data_collection_advisor_full.py')
else:
    if st.button("Generate narrative using ChatGPT"):
        try:
            with st.spinner("Generating narrative (OpenAI)..."):
                narrative = generate_llm_narrative(profile_cached, recs_cached, st.session_state.get("plan_view", "Must / Should / Nice-to-have"))
            st.session_state["narrative"] = narrative
            st.success("Narrative generated.")
        except Exception as e:
            st.error("Narrative generation failed:")
            st.exception(e)

    if st.session_state.get("narrative"):
        st.markdown(st.session_state["narrative"])
        st.download_button(
            "Download narrative (Markdown)",
            data=st.session_state["narrative"],
            file_name=f"{profile_cached.customer_name.replace(' ', '_')}_data_plan.md",
            mime="text/markdown",
        )
