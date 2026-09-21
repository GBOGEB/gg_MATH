"""Execute the bounded W260 slice and bind the source/tests/environment receipt."""
from pathlib import Path
import hashlib, json, os, platform, subprocess, sys
import numpy, scipy

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
commands=[
    [sys.executable,'-m','unittest','discover','-s','tests','-p','test_h05_regressions.py'],
    [sys.executable,'-m','unittest','discover','-s','tests','-p','test_w260_uncertainty.py'],
    [sys.executable,'tests/test_bradley_terry_reference.py'],
    [sys.executable,'tests/test_lm10_h05_typed_relational_bridge.py'],
]
records=[]
for command in commands:
    run=subprocess.run(command,capture_output=True,text=True)
    records.append({'command':command[1:],'returncode':run.returncode,'output':run.stdout+run.stderr})
passed=all(r['returncode']==0 for r in records)
paths=['kernels/bradley_terry.py','kernels/typed_relational_bridge.py','kernels/stats_core.py',
       'kernels/uncertainty.py','tests/test_h05_regressions.py','tests/test_w260_uncertainty.py',
       'tests/test_bradley_terry_reference.py','tests/test_lm10_h05_typed_relational_bridge.py',
       'mission/LM10/w260_prove.py']
receipt={'schema':'gg-math-w260-bd-runtime/v1','result':'PASS' if passed else 'FAIL',
    'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
    'workflow_run_id':os.environ.get('GITHUB_RUN_ID'),
    'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__},
    'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
    'commands':records,'authority_transfer':False,'formal_credit_delta':0,
    'scope':['H05_historians_three_defects','BD-260.1_fixed_sample_scalar_uncertainty'],
    'deferred':['anytime_valid_confidence_sequences','BT_uncertainty','PCA_subspace_uncertainty',
                'measured_consumer','independent_governed_attestation']}
out=Path('artifacts/w260');out.mkdir(parents=True,exist_ok=True)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'result':receipt['result'],'command_groups':len(records),'receipt':str(out/'receipt.json')}))
sys.exit(0 if passed else 1)
