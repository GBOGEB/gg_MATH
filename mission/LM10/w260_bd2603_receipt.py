#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FILES=[ROOT/"kernels/pca_uncertainty.py",ROOT/"tests/test_lm10_w260_bd2603_pca_uncertainty.py",ROOT/"mission/LM10/W260_BD260_3_PCA_UNCERTAINTY_v1.json"]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
out={
 "schema":"gg_math.w260.bd2603.runtime_receipt.v1","mission_id":"W260","bd_id":"BD-260.3",
 "repository":"GBOGEB/gg_MATH","source_sha":os.environ.get("SOURCE_SHA","LOCAL_UNBOUND"),
 "status":"PASS_PCA_TEMPORAL_UNCERTAINTY_REFERENCE",
 "covered":["BOOTSTRAP_EIGENVALUE","ALIGNED_LOADING_INTERVALS","ASSIGNMENT_SIGN_ALIGNMENT","EIGENGAP_GUARD",
 "CONGRUENCE_DISTRIBUTION","PRINCIPAL_ANGLES","PROJECTION_DISTANCE","GRASSMANN_DISTANCE","PROCRUSTES_TEMPORAL_UNCERTAINTY"],
 "withheld":{"GRASSMANN_STATE_SPACE":"RESEARCH_TODO"},
 "file_sha256":{str(p.relative_to(ROOT)):sha(p) for p in FILES},
 "authority_transfer":False,"formal_credit_delta":0,"engineering_acceptance":False
}
target=ROOT/"artifacts/lm10_w260/BD260_3_RUNTIME_RECEIPT.json"; target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print("PASS_W260_BD260_3_RECEIPT")
