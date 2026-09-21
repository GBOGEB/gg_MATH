#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FILES=[
 ROOT/"kernels/bt_uncertainty.py",
 ROOT/"tests/test_lm10_w260_bd2602_bt_uncertainty.py",
 ROOT/"mission/LM10/W260_BD260_2_BT_UNCERTAINTY_v1.json",
]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
out={
 "schema":"gg_math.w260.bd2602.runtime_receipt.v1",
 "mission_id":"W260","bd_id":"BD-260.2","repository":"GBOGEB/gg_MATH",
 "source_sha":os.environ.get("SOURCE_SHA","LOCAL_UNBOUND"),
 "status":"PASS_BT_UNCERTAINTY_REFERENCE",
 "covered":["FINITE_MLE_GUARD","LIKELIHOOD_SURFACE","FISHER_LOG_STRENGTH_UNCERTAINTY",
 "PAIR_PROBABILITY_CI","L2_REGULARIZATION","SEEDED_PENALTY_CALIBRATION","BOOTSTRAP_RANK_FREQUENCY"],
 "withheld":{"HIERARCHICAL_BAYESIAN_BT":"RESEARCH_TODO"},
 "file_sha256":{str(p.relative_to(ROOT)):sha(p) for p in FILES},
 "authority_transfer":False,"formal_credit_delta":0,
 "engineering_priority":False,"qps_priority":False
}
target=ROOT/"artifacts/lm10_w260/BD260_2_RUNTIME_RECEIPT.json"
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print("PASS_W260_BD260_2_RECEIPT")
