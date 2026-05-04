"""
Rule-based 5G telco troubleshooting classifier.

Parses drive-test data and engineering parameters from a question text,
computes quantitative features for each of the 8 root-cause categories,
and applies priority-ordered if-else rules to select the most likely cause.

No model, no GPU, no training required.

Usage:
    python qwen_train_new_rules.py --test_csv phase_1_test.csv --truth_csv phase_1_test_truth.csv
    python qwen_train_new_rules.py --self_test          # quick sanity check on built-in examples
"""

import os
import re
import math
import argparse
from collections import Counter

import pandas as pd
from tqdm.auto import tqdm


# ══════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════

def _safe_float(val, default=None):
    if val is None or str(val).strip() in ("-", ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def _parse_pipe_table(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return [], []
    headers = [h.strip() for h in lines[0].split("|")]
    rows = []
    for line in lines[1:]:
        fields = [f.strip() for f in line.split("|")]
        if len(fields) == len(headers):
            rows.append(dict(zip(headers, fields)))
    return headers, rows


def _get_vertical_beamwidth(beam_scenario, bw_rules=None):
    """Return vertical beamwidth based on beam scenario.

    bw_rules: dict with keys 'low', 'mid', 'high' mapping scenario
              thresholds to beamwidth values.
              Defaults: {low_max: 5, mid_max: 11, low_bw: 6, mid_bw: 12, high_bw: 25}
    """
    if bw_rules is None:
        bw_rules = {}
    low_max = bw_rules.get("low_max", 5)
    mid_max = bw_rules.get("mid_max", 11)
    low_bw  = bw_rules.get("low_bw", 6)
    mid_bw  = bw_rules.get("mid_bw", 12)
    high_bw = bw_rules.get("high_bw", 25)

    if not beam_scenario or beam_scenario.upper() == "DEFAULT":
        return low_bw
    m = re.search(r"(\d+)", beam_scenario)
    if not m:
        return low_bw
    num = int(m.group(1))
    if num <= low_max:
        return low_bw
    elif num <= mid_max:
        return mid_bw
    else:
        return high_bw


# ══════════════════════════════════════════════════════════════
# EXTRACT BASE STATS FROM QUESTION TEXT
# ══════════════════════════════════════════════════════════════

def extract_base_stats(question_text):
    """Try to regex-extract the configurable default values from the
    question text.  Falls back to hardcoded defaults if not found.

    Returns a dict with:
      - digital_tilt_default_code  (int, default 255)
      - digital_tilt_default_angle (int, default 6)
      - bw_rules  (dict for _get_vertical_beamwidth)
    """
    stats = {
        "digital_tilt_default_code": 255,
        "digital_tilt_default_angle": 6,
        "bw_rules": {
            "low_max": 5,
            "mid_max": 11,
            "low_bw": 6,
            "mid_bw": 12,
            "high_bw": 25,
        },
    }

    # --- electronic downtilt default ---
    # Pattern: "default electronic downtilt value is 255, representing a downtilt angle of 6 degrees"
    m = re.search(
        r"default\s+electronic\s+downtilt\s+value\s+is\s+(\d+).*?"
        r"downtilt\s+angle\s+of\s+(\d+)\s+degrees",
        question_text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        stats["digital_tilt_default_code"] = int(m.group(1))
        stats["digital_tilt_default_angle"] = int(m.group(2))

    # --- beamwidth rules ---
    # Pattern: "SCENARIO_1 to SCENARIO_5, the vertical beamwidth is 6 degrees"
    m_low = re.search(
        r"SCENARIO_(\d+)\s*(?:to|[-–])\s*SCENARIO_(\d+).*?vertical\s+beamwidth\s+"
        r"is\s+(\d+)\s+degrees",
        question_text, re.IGNORECASE,
    )
    if m_low:
        stats["bw_rules"]["low_max"] = int(m_low.group(2))
        stats["bw_rules"]["low_bw"] = int(m_low.group(3))

    # Find all beamwidth rules (there are usually 3)
    bw_patterns = re.findall(
        r"SCENARIO_(\d+)\s*(?:to|[-–])\s*SCENARIO_(\d+).*?vertical\s+beamwidth\s+"
        r"is\s+(\d+)\s+degrees",
        question_text, re.IGNORECASE,
    )
    if len(bw_patterns) >= 2:
        # First match: low range
        stats["bw_rules"]["low_max"] = int(bw_patterns[0][1])
        stats["bw_rules"]["low_bw"] = int(bw_patterns[0][2])
        # Second match: mid range
        stats["bw_rules"]["mid_max"] = int(bw_patterns[1][1])
        stats["bw_rules"]["mid_bw"] = int(bw_patterns[1][2])

    m_high = re.search(
        r"SCENARIO_(\d+)\s+or\s+above.*?vertical\s+beamwidth\s+is\s+(\d+)\s+degrees",
        question_text, re.IGNORECASE,
    )
    if m_high:
        stats["bw_rules"]["high_bw"] = int(m_high.group(2))

    # Also handle: "Default or SCENARIO_1 to SCENARIO_5 ... 6 degrees"
    m_default_low = re.search(
        r"Default\s+or\s+SCENARIO_\d+\s*(?:to|[-–])\s*SCENARIO_(\d+).*?"
        r"vertical\s+beamwidth\s+is\s+(\d+)\s+degrees",
        question_text, re.IGNORECASE,
    )
    if m_default_low:
        stats["bw_rules"]["low_max"] = int(m_default_low.group(1))
        stats["bw_rules"]["low_bw"] = int(m_default_low.group(2))

    return stats


# ══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_all_features(question_text):
    """Parse drive-test data and engineering params from the question text,
    compute quantitative values for each of the 8 root-cause categories.

    Returns a dict of computed feature values, or None if parsing fails.
    """
    dt_match = re.search(
        r"User plane drive test data as follows[：:]?\s*\n(.*?)(?:\n\s*\n\s*Eng|\nEng)",
        question_text, re.DOTALL,
    )
    eng_match = re.search(
        r"Eng[ei]neering parameters data as follows[：:]?\s*\n(.*?)$",
        question_text, re.DOTALL,
    )
    if not dt_match or not eng_match:
        return None

    _, dt_rows = _parse_pipe_table(dt_match.group(1))
    _, eng_rows = _parse_pipe_table(eng_match.group(1))
    if not dt_rows or not eng_rows:
        return None

    base = extract_base_stats(question_text)
    dt_default_code = base["digital_tilt_default_code"]
    dt_default_angle = base["digital_tilt_default_angle"]
    bw_rules = base["bw_rules"]

    # ── look-up tables ──
    pci_to_eng = {}
    pci_to_gnb = {}
    for e in eng_rows:
        pci = _safe_float(e.get("PCI"))
        if pci is not None:
            pci_to_eng[int(pci)] = e
            pci_to_gnb[int(pci)] = e.get("gNodeB ID", "")

    # ── extract arrays from drive-test rows ──
    speeds, rsrps, sinrs, tputs, rbs = [], [], [], [], []
    serving_pcis = []
    ue_coords = []
    for r in dt_rows:
        speeds.append(_safe_float(r.get("GPS Speed (km/h)")))
        rsrps.append(_safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]")))
        sinrs.append(_safe_float(r.get("5G KPI PCell RF Serving SS-SINR [dB]")))
        tputs.append(_safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]")))
        rbs.append(_safe_float(r.get("5G KPI PCell Layer1 DL RB Num (Including 0)")))
        serving_pcis.append(_safe_float(r.get("5G KPI PCell RF Serving PCI")))
        lat = _safe_float(r.get("Latitude"))
        lon = _safe_float(r.get("Longitude"))
        ue_coords.append((lat, lon))

    speeds = [v for v in speeds if v is not None]
    rsrps  = [v for v in rsrps  if v is not None]
    sinrs  = [v for v in sinrs  if v is not None]
    tputs  = [v for v in tputs  if v is not None]
    rbs    = [v for v in rbs    if v is not None]
    srv_pcis = [int(v) for v in serving_pcis if v is not None]

    n = len(dt_rows)
    feats = {}

    # ── Overall stats ──
    feats["avg_rsrp"] = sum(rsrps) / len(rsrps) if rsrps else 0
    feats["avg_sinr"] = sum(sinrs) / len(sinrs) if sinrs else 0
    feats["avg_tput"] = sum(tputs) / len(tputs) if tputs else 0
    feats["low_tput_count"] = sum(1 for t in tputs if t < 600)
    feats["n"] = n

    # ── C7: vehicle speed ──
    feats["avg_speed"] = sum(speeds) / len(speeds) if speeds else 0
    feats["max_speed"] = max(speeds) if speeds else 0

    # ── C8: average RBs ──
    feats["avg_rbs"] = sum(rbs) / len(rbs) if rbs else 0
    feats["low_rb_count"] = sum(1 for r in rbs if r < 160)
    feats["min_rbs"] = min(rbs) if rbs else 0

    # ── Serving cell analysis ──
    main_pci = Counter(srv_pcis).most_common(1)[0][0] if srv_pcis else None
    all_serving_pcis = set(srv_pcis)
    feats["main_pci"] = main_pci
    feats["all_serving_pcis"] = all_serving_pcis

    # ── C2: coverage distance ──
    max_dist_m = 0
    avg_dist_m = 0
    if main_pci and main_pci in pci_to_eng:
        cell = pci_to_eng[main_pci]
        clat = _safe_float(cell.get("Latitude"))
        clon = _safe_float(cell.get("Longitude"))
        if clat and clon:
            dists = []
            for lat, lon in ue_coords:
                if lat and lon:
                    dists.append(_haversine_km(lat, lon, clat, clon))
            if dists:
                avg_dist_m = sum(dists) / len(dists) * 1000
                max_dist_m = max(dists) * 1000
    feats["max_dist_m"] = max_dist_m
    feats["avg_dist_m"] = avg_dist_m

    # Also check distances for all serving PCIs (not just the most common)
    max_dist_any_serving = max_dist_m
    for spci in all_serving_pcis:
        if spci in pci_to_eng:
            cell = pci_to_eng[spci]
            clat = _safe_float(cell.get("Latitude"))
            clon = _safe_float(cell.get("Longitude"))
            if clat and clon:
                for lat, lon in ue_coords:
                    if lat and lon:
                        d = _haversine_km(lat, lon, clat, clon) * 1000
                        max_dist_any_serving = max(max_dist_any_serving, d)
    feats["max_dist_any_serving_m"] = max_dist_any_serving

    # ── C1: downtilt analysis (check ALL serving PCIs) ──
    max_downtilt_excess = 0  # how much total downtilt exceeds beamwidth
    c1_details = []
    for spci in all_serving_pcis:
        if spci in pci_to_eng:
            cell = pci_to_eng[spci]
            mech_dt = _safe_float(cell.get("Mechanical Downtilt"), 0)
            digi_tilt = _safe_float(cell.get("Digital Tilt"), 0)
            if digi_tilt == dt_default_code:
                digi_tilt = dt_default_angle
            total_dt = mech_dt + digi_tilt
            beam = cell.get("Beam Scenario", "DEFAULT")
            vbw = _get_vertical_beamwidth(beam, bw_rules)
            excess = total_dt - vbw
            c1_details.append({
                "pci": spci, "total_dt": total_dt, "mech": mech_dt,
                "digi": digi_tilt, "beam": beam, "vbw": vbw, "excess": excess,
            })
            if excess > max_downtilt_excess:
                max_downtilt_excess = excess
    feats["max_downtilt_excess"] = max_downtilt_excess
    feats["c1_details"] = c1_details

    # ── C5: frequent handovers ──
    ho_count = sum(
        1 for i in range(1, len(srv_pcis)) if srv_pcis[i] != srv_pcis[i - 1]
    )
    feats["handover_count"] = ho_count

    # ── C6: PCI mod 30 collision ──
    pci_mod30_collisions = []
    if srv_pcis:
        srv_mod30 = set(p % 30 for p in srv_pcis)
        all_nbr_pcis = set()
        for r in dt_rows:
            for i in range(1, 6):
                v = _safe_float(r.get(
                    f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI"
                ))
                if v is not None:
                    all_nbr_pcis.add(int(v))
        pci_mod30_collisions = sorted(
            p for p in all_nbr_pcis
            if p % 30 in srv_mod30 and p not in all_serving_pcis
        )
    feats["pci_mod30_collisions"] = pci_mod30_collisions

    # Also check among serving PCIs themselves if they differ but share mod30
    serving_mod30_collision = False
    if len(all_serving_pcis) > 1:
        mods = [p % 30 for p in all_serving_pcis]
        if len(mods) != len(set(mods)):
            serving_mod30_collision = True
    feats["serving_mod30_collision"] = serving_mod30_collision

    # Count timestamps with strong-neighbor that has mod30 collision
    # Now also track the RSRP difference to the colliding neighbor
    mod30_collision_strong_count = 0
    mod30_collision_any_count = 0
    for r in dt_rows:
        sp = _safe_float(r.get("5G KPI PCell RF Serving PCI"))
        sr = _safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]"))
        if sp is None:
            continue
        sp_mod30 = int(sp) % 30
        for i in range(1, 6):
            npci = _safe_float(r.get(
                f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI"
            ))
            nrsrp = _safe_float(r.get(
                f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]"
            ))
            if npci is not None and nrsrp is not None:
                if int(npci) % 30 == sp_mod30 and int(npci) not in all_serving_pcis:
                    mod30_collision_any_count += 1
                    # "strong" = colliding neighbor within 10dB of serving
                    if sr is not None and (sr - nrsrp) < 10:
                        mod30_collision_strong_count += 1
                    break
    feats["mod30_collision_strong_count"] = mod30_collision_strong_count
    feats["mod30_collision_any_count"] = mod30_collision_any_count

    # ── C3: neighbor provides higher signal ──
    nbr_stronger = 0
    for r in dt_rows:
        srv = _safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]"))
        if srv is None:
            continue
        for i in range(1, 6):
            nr = _safe_float(r.get(
                f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]"
            ))
            if nr is not None and nr > srv:
                nbr_stronger += 1
                break
    feats["nbr_stronger_count"] = nbr_stronger

    # ── C4: non-colocated co-frequency overlapping coverage ──
    non_coloc = 0
    for r in dt_rows:
        sp = _safe_float(r.get("5G KPI PCell RF Serving PCI"))
        sr = _safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]"))
        if sp is None or sr is None:
            continue
        srv_gnb = pci_to_gnb.get(int(sp), "")
        overlap = False
        for i in range(1, 6):
            npci = _safe_float(r.get(
                f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI"
            ))
            nrsrp = _safe_float(r.get(
                f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]"
            ))
            if npci is not None and nrsrp is not None:
                nbr_gnb = pci_to_gnb.get(int(npci), "")
                if nbr_gnb and nbr_gnb != srv_gnb and (sr - nrsrp) < 10:
                    overlap = True
                    break
        if overlap:
            non_coloc += 1
    feats["non_coloc_overlap_count"] = non_coloc

    # ── C3: check if the serving cells have low throughput and a neighbor
    #    from a co-located cell provides better throughput ──
    pci_tputs = {}
    for r in dt_rows:
        sp = _safe_float(r.get("5G KPI PCell RF Serving PCI"))
        tp = _safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]"))
        if sp is not None and tp is not None:
            pci_tputs.setdefault(int(sp), []).append(tp)
    feats["pci_tputs"] = pci_tputs

    # Check if there's a handover that notably improves throughput
    coloc_handover_gain = 0
    non_coloc_handover_gain = 0
    valid_tp_rows = []
    for r in dt_rows:
        sp = _safe_float(r.get("5G KPI PCell RF Serving PCI"))
        tp = _safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]"))
        if sp is not None and tp is not None:
            valid_tp_rows.append((int(sp), tp))
            
    for i in range(1, len(valid_tp_rows)):
        prev_pci, prev_tp = valid_tp_rows[i-1]
        curr_pci, curr_tp = valid_tp_rows[i]
        
        if prev_pci != curr_pci:
            # average up to 2 samples before
            start_before = max(0, i-2)
            before_tps = [x[1] for x in valid_tp_rows[start_before:i]]
            tput_before = sum(before_tps) / len(before_tps)
            
            # average up to 3 samples after (including current)
            end_after = min(len(valid_tp_rows), i+3)
            after_tps = [x[1] for x in valid_tp_rows[i:end_after]]
            tput_after = sum(after_tps) / len(after_tps)
            
            gain = tput_after - tput_before
            
            # check if co-located
            prev_gnb = pci_to_gnb.get(prev_pci, "A")
            curr_gnb = pci_to_gnb.get(curr_pci, "B")
            
            if prev_gnb == curr_gnb:
                if gain > coloc_handover_gain:
                    coloc_handover_gain = gain
            else:
                if gain > non_coloc_handover_gain:
                    non_coloc_handover_gain = gain
                
    feats["coloc_handover_gain"] = coloc_handover_gain
    feats["non_coloc_handover_gain"] = non_coloc_handover_gain

    # C3 additional: check if most neighbors are co-located (same gNodeB)
    # and throughput is mediocre — suggests sub-optimal cell selection
    coloc_neighbor_count = 0
    non_coloc_neighbor_count = 0
    for r in dt_rows:
        sp = _safe_float(r.get("5G KPI PCell RF Serving PCI"))
        if sp is None:
            continue
        srv_gnb = pci_to_gnb.get(int(sp), "")
        for i in range(1, 6):
            npci = _safe_float(r.get(
                f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI"
            ))
            if npci is not None and int(npci) in pci_to_gnb:
                if pci_to_gnb[int(npci)] == srv_gnb:
                    coloc_neighbor_count += 1
                else:
                    non_coloc_neighbor_count += 1
    total_nbr = coloc_neighbor_count + non_coloc_neighbor_count
    feats["coloc_ratio"] = coloc_neighbor_count / total_nbr if total_nbr > 0 else 0
    feats["c3_mediocre_tput"] = feats["avg_tput"] < 700 and feats["low_tput_count"] >= 3

    return feats


# ══════════════════════════════════════════════════════════════
# RULE-BASED CLASSIFIER
# ══════════════════════════════════════════════════════════════

def classify(question_text):
    """Apply priority-ordered if-else rules on computed features.

    Strategy:
      - Hard rules for clear-cut cases: C7, C8, C2, C5
      - Scoring + heuristics for ambiguous cases: C1, C3, C4, C6
    Returns (prediction, reasoning) where prediction is 'C1'-'C8'.
    """
    feats = compute_all_features(question_text)
    if feats is None:
        return "C3", "Could not parse input data; defaulting to C3"

    n = feats["n"]

    # ═══ PHASE 1: Hard rules for clear-cut cases ═══

    # ── C7 — speed > 40 km/h ──
    if feats["max_speed"] > 40:
        return "C7", f"C7: max speed={feats['max_speed']:.0f}km/h (>40)"

    # ── C8 — average RBs < 160 ──
    if feats["avg_rbs"] < 160:
        return "C8", f"C8: avg RBs={feats['avg_rbs']:.1f} (<160)"

    # ── C2 — coverage distance > 1 km ──
    if feats["max_dist_any_serving_m"] > 1000:
        return "C2", f"C2: distance={feats['max_dist_any_serving_m']:.0f}m (>1000)"

    # ── C5 — frequent handovers ──
    if feats["handover_count"] >= 3:
        return "C5", f"C5: {feats['handover_count']} handovers in {n} samples"

    # ── C8 secondary — many low-RB samples ──
    low_rb_ratio = feats["low_rb_count"] / n if n > 0 else 0
    if low_rb_ratio >= 0.3 and feats["avg_rbs"] < 190:
        return "C8", f"C8: {feats['low_rb_count']}/{n} with RBs<160, avg={feats['avg_rbs']:.1f}"

    # ═══ PHASE 2: Scoring for ambiguous C1/C3/C4/C6 ═══

    overlap_ratio = feats["non_coloc_overlap_count"] / n if n > 0 else 0
    nbr_ratio = feats["nbr_stronger_count"] / n if n > 0 else 0
    has_mod30 = (
        len(feats["pci_mod30_collisions"]) > 0
        or feats["serving_mod30_collision"]
    )
    mod30_strong_ratio = feats["mod30_collision_strong_count"] / n if n > 0 else 0

    scores = {}

    # C1 score: downtilt excess heavily weighted by RSRP weakness
    dt_excess = feats["max_downtilt_excess"]
    rsrp_factor = max(0, (-feats["avg_rsrp"] - 75)) / 15  # 0 at -75, 1 at -90
    c1_score = dt_excess * (0.5 + rsrp_factor)
        
    # Reduce C1 when overlap is present — the issue is more likely C4
    if overlap_ratio >= 0.35:
        c1_score *= 0.5
    # Reduce C1 when mod30 collisions are present — more likely C6
    if has_mod30 and feats["avg_sinr"] < 10:
        c1_score *= 0.6
    scores["C1"] = c1_score

    # C3 score: serving cell has low throughput, neighbors are mostly co-located
    # (same site), suggesting sub-optimal cell selection
    c3_score = 0
    co_gain = feats.get("coloc_handover_gain", 0)
    if co_gain > 250:
        c3_score += 10
    elif co_gain > 150:
        c3_score += 6
    elif co_gain > 50:
        c3_score += 3
        
    n_co_gain = feats.get("non_coloc_handover_gain", 0)
    if n_co_gain > 250:
        c3_score += 4
    elif n_co_gain > 150:
        c3_score += 2

    if feats["c3_mediocre_tput"]:
        c3_score += 4
    if feats["coloc_ratio"] > 0.3:
        c3_score += 3
    if nbr_ratio > 0.5:
        c3_score += 6
    elif nbr_ratio > 0.3:
        c3_score += 3
    # If all serving cells are from the same gNodeB and throughput is inconsistent
    if len(feats["all_serving_pcis"]) >= 2:
        # Just check pci_tputs variance
        if len(feats.get("pci_tputs", {})) >= 2:
            all_avg_tputs = []
            for sp, tps in feats["pci_tputs"].items():
                if len(tps) >= 2:
                    all_avg_tputs.append(sum(tps) / len(tps))
            if len(all_avg_tputs) >= 2:
                tput_range = max(all_avg_tputs) - min(all_avg_tputs)
                if tput_range > 300:  # big difference between serving cells
                    c3_score += 5
    # Penalize if there's strong non-colocated interference (more likely C4)
    if overlap_ratio > 0.4:
        c3_score -= 8
        
    # Structural rule: if C1 (downtilt) or C4 (overlap) is massively high, it is a structural
    # root cause. Sub-optimal cell selection (C3) should not override it unless C3 is perfect.
    if scores.get("C1", 0) > 13 or scores.get("C4", 0) > 13:
        c3_score *= 0.6
        
    scores["C3"] = c3_score

    # C4 score: non-colocated overlap — boosted
    scores["C4"] = overlap_ratio * 30

    # C6 score: mod30 collision with neighbors + low SINR
    c6_score = 0
    if has_mod30:
        # Use strong ratio first, fall back to any-count ratio
        effective_ratio = mod30_strong_ratio if mod30_strong_ratio > 0 else (
            feats["mod30_collision_any_count"] / n if n > 0 else 0
        ) * 0.5  # discount if not "strong"
        c6_score = effective_ratio * 15
        # Boost if serving PCIs themselves collide (very strong C6 signal)
        if feats["serving_mod30_collision"]:
            c6_score += 25
        # Boost if SINR is low (interference indicator)
        sinr_penalty = max(0, 10 - feats["avg_sinr"]) / 10
        c6_score *= (1 + sinr_penalty)
    scores["C6"] = c6_score

    best = max(scores, key=scores.get)
    detail = (
        f"C1={scores['C1']:.1f}(excess={dt_excess},dist={feats.get('avg_dist_m', 0):.0f}), "
        f"C3={scores['C3']:.1f}(co_gain={co_gain:.0f}, nco_gain={n_co_gain:.0f}), "
        f"C4={scores['C4']:.1f}(overlap={overlap_ratio:.0%}), "
        f"C6={scores['C6']:.1f}(mod30={has_mod30},strong={mod30_strong_ratio:.0%})"
    )
    return best, f"Scoring: {detail} → {best}"


def format_answer(label):
    """Format the classification answer with boxed notation."""
    return f"\\boxed{{{label}}}"


# ══════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description="Rule-based 5G telco troubleshooting classifier"
)
parser.add_argument("--test_csv",  default="phase_1_test.csv")
parser.add_argument("--truth_csv", default="phase_1_test_truth.csv")
parser.add_argument("--max_eval",  type=int, default=None)
parser.add_argument("--skip_eval", action="store_true")
parser.add_argument("--self_test", action="store_true",
                    help="Run quick self-test on built-in examples")
parser.add_argument("--verbose",   action="store_true",
                    help="Print reasoning for each prediction")
parser.add_argument("--train",   action="store_true",
                    help="Evaluate on train.csv directly")
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════
# SELF-TEST (built-in examples for sanity check)
# ══════════════════════════════════════════════════════════════
if args.self_test:
    print("=" * 60)
    print("SELF-TEST: Running on built-in examples")
    print("=" * 60)

    self_test_cases = []
    print("Self-test mode requires full question texts embedded in code.")
    print("Use --test_csv/--truth_csv for batch evaluation instead.")
    print("Exiting self-test.")
    exit(0)


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════
if args.skip_eval:
    print("Skipping evaluation (--skip_eval flag).")
else:
    print("=" * 60)
    print("Evaluating on test set (rule-based)")
    print("=" * 60)

    if args.train:
        df = pd.read_csv("train.csv")
        merged = df.copy()
        merged["gold"] = merged["answer"].astype(str).str.upper().str.extract(r"(C[1-8])", expand=False)
        print(f"Train rows: {len(df)} | Valid Truth: {merged['gold'].notna().sum()}")
    else:
        test_df = pd.read_csv(args.test_csv)
        truth_df = pd.read_csv(args.truth_csv)

        truth_cols = [c for c in truth_df.columns if c != "ID"]
        assert len(truth_cols) > 0, "No truth label column found"
        truth_col = truth_cols[-1]  # Always assume last column is target, or specify strictly

        truth_df = truth_df.copy()
        truth_df["base_id"] = truth_df["ID"].astype(str).str.replace(
            r"_[0-9]+$", "", regex=True
        )
        truth_base = truth_df.groupby("base_id", as_index=False)[truth_col].first()

        merged = test_df.merge(truth_base, left_on="ID", right_on="base_id", how="inner")
        merged["gold"] = (
            merged[truth_col].astype(str).str.upper().str.extract(r"(C[1-8])", expand=False)
        )

        print(f"Test rows: {len(test_df)} | Truth rows: {len(truth_df)} | "
              f"Merged rows: {len(merged)}")
        assert len(merged) > 0, "Merge produced 0 rows."

    eval_df = merged if args.max_eval is None else merged.head(args.max_eval).copy()

    preds, reasonings = [], []
    for q in tqdm(eval_df["question"].tolist(), total=len(eval_df), desc="Classifying"):
        p, r = classify(q)
        preds.append(p)
        reasonings.append(r)
        if args.verbose:
            print(f"  → {p}: {r}")

    eval_df = eval_df.copy()
    eval_df["pred"] = preds
    eval_df["reasoning"] = reasonings

    valid_mask = eval_df["gold"].notna() & eval_df["pred"].notna()
    accuracy = (
        (eval_df.loc[valid_mask, "pred"] == eval_df.loc[valid_mask, "gold"]).mean()
        if valid_mask.any() else 0.0
    )

    print(f"\nRows evaluated: {len(eval_df)}")
    print(f"Rows with valid gold+pred labels: {int(valid_mask.sum())}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Per-class breakdown
    if valid_mask.any():
        print("\nPer-class accuracy:")
        for c in sorted(eval_df.loc[valid_mask, "gold"].unique()):
            mask_c = valid_mask & (eval_df["gold"] == c)
            correct_c = (eval_df.loc[mask_c, "pred"] == eval_df.loc[mask_c, "gold"]).sum()
            total_c = mask_c.sum()
            print(f"  {c}: {correct_c}/{total_c} = {correct_c/total_c:.2%}")

    # Confusion matrix
    if valid_mask.any():
        print("\nConfusion matrix (rows=gold, cols=pred):")
        classes = sorted(set(eval_df.loc[valid_mask, "gold"].tolist() +
                             eval_df.loc[valid_mask, "pred"].tolist()))
        header = "      " + "  ".join(f"{c:>4}" for c in classes)
        print(header)
        for gc in classes:
            row_vals = []
            for pc in classes:
                cnt = ((eval_df.loc[valid_mask, "gold"] == gc) &
                       (eval_df.loc[valid_mask, "pred"] == pc)).sum()
                row_vals.append(f"{cnt:>4}")
            print(f"  {gc}  " + "  ".join(row_vals))

    # Show mistakes
    mistakes = eval_df[valid_mask & (eval_df["pred"] != eval_df["gold"])][
        ["ID", "gold", "pred", "reasoning"]
    ]
    print(f"\nMistakes: {len(mistakes)}")
    if len(mistakes) > 0 and len(mistakes) <= 20:
        print(mistakes.to_string(index=False))
    elif len(mistakes) > 20:
        print(mistakes.head(20).to_string(index=False))
        print(f"  ... and {len(mistakes) - 20} more")


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
