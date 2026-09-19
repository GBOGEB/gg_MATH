#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MISSION = ROOT / "mission" / "GRANDMISSION_I_B_TEMPORAL_PCA"
CONTRACT = MISSION / "GM_I_A_MULTI_CLOCK_PROVIDER_CONTRACT_v1.json"
SNAPSHOT = MISSION / "inbound" / "GM_I_A_MULTI_CLOCK_MEASURED_PULSES_v1.json"
OUT = ROOT / "artifacts" / "grandmission_i_b_temporal_pca" / "GM_I_A_MULTI_CLOCK_READINESS_RECEIPT.json"

def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

def strongly_connected(nodes, edges):
    graph = {n: set() for n in nodes}
    rev = {n: set() for n in nodes}
    for a, b in edges:
        graph[a].add(b)
        rev[b].add(a)
    def reach(g, start):
        seen, stack = set(), [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(g[cur] - seen)
        return seen
    if not nodes:
        return False
    root = nodes[0]
    return len(reach(graph, root)) == len(nodes) and len(reach(rev, root)) == len(nodes)

def main():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    raw = SNAPSHOT.read_bytes()
    d = json.loads(raw)
    assert c["authority_transfer"] is False
    assert c["formal_credit_delta"] == 0
    assert git_blob_sha(raw) == c["source_git_blob_sha"]
    assert d["schema"] == "missioncontrol.gm_i_a.multiclock_measured_pulses.v1"
    assert d["mission"] == "GM-I-A"
    assert d["authority_transfer"] is False
    assert d["evidence_policy"]["synthetic_rows"] == 0
    assert d["evidence_policy"]["expert_seeded_rows"] == 0

    features = d["features"]
    rows = d["rows"]
    assert len(features) == 9
    assert len(rows) == 1
    assert all(r["evidence_class"] == "MEASURED" for r in rows)

    n, p = len(rows), len(features)
    pca_min = max(10, 3 * p)
    pca_ready = n >= pca_min

    pairs = []
    edges = []
    for row in rows:
        for a, b in itertools.combinations(features, 2):
            av, bv = float(row[a]), float(row[b])
            if av > bv:
                w, l = a, b
                edges.append((w, l))
            elif bv > av:
                w, l = b, a
                edges.append((w, l))
            else:
                w = l = None
            pairs.append({"pulse_id": row["pulse_id"], "a": a, "b": b, "winner": w, "loser": l})

    bt_connected = strongly_connected(features, edges)
    bt_ready = n >= c["fit_policy"]["bt_minimum_repeat_pulses"] and bt_connected

    latest = rows[-1]
    order = sorted(
        [{"feature": f, "seconds": float(latest[f])} for f in features],
        key=lambda x: x["seconds"],
        reverse=True,
    )
    queue_features = [f for f in features if f.endswith("_queue_seconds")]
    queue_seconds = sum(float(latest[f]) for f in queue_features)
    total_seconds = sum(float(latest[f]) for f in features)

    receipt = {
        "schema": "gg_math.gm_i_a_multiclock_readiness.v1",
        "source_repo": c["source_repo"],
        "source_ref": c["source_ref"],
        "source_git_blob_sha": c["source_git_blob_sha"],
        "snapshot_git_blob_sha": git_blob_sha(raw),
        "measured_pulses_n": n,
        "feature_count_p": p,
        "p_over_n": p / n,
        "pca": {
            "status": "READY_FOR_FIT" if pca_ready else "DEFER_N_LT_MAX_10_3P",
            "minimum_rows_required": pca_min,
            "fit_executed": False
        },
        "bradley_terry_reverse_pressure": {
            "status": "READY_FOR_FIT" if bt_ready else "DEFER_NO_REPEAT_STRONG_CONNECTIVITY",
            "pair_records": len(pairs),
            "directed_win_graph_strongly_connected": bt_connected,
            "fit_executed": False,
            "semantics": "LONGER_DURATION_IS_RUNTIME_PRESSURE_WIN"
        },
        "descriptive_pressure_order": {
            "classification": "MEASURED_DURATION_ORDER_NOT_BT_FIT",
            "order": order,
            "queue_seconds": queue_seconds,
            "selected_clock_seconds": total_seconds,
            "queue_fraction": queue_seconds / total_seconds
        },
        "guards": [
            "NO_PCA_FIT_BEFORE_SAMPLE_GATE",
            "NO_BT_FIT_FROM_SINGLE_TRANSITIVE_PULSE",
            "PCA_SIGN_IS_NOT_POLICY_DIRECTION",
            "NO_ENGINEERING_HEPAK_QEQ_AUTHORITY_TRANSFER"
        ],
        "status": "PASS_READINESS_WITH_STATISTICAL_GATES_HELD",
        "authority_transfer": False,
        "formal_credit_delta": 0
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS_GM_I_A_MULTI_CLOCK_READINESS")
    print(json.dumps(receipt, sort_keys=True))

if __name__ == "__main__":
    main()
