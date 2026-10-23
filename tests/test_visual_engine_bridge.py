"""Bounded source/format parity and negative/edge-case testflight for visual engine."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from kernels.visual_engine_bridge import check, emit, verify

ROOT=Path(__file__).resolve().parents[1]
EX=ROOT/'mission/federation/W34_VISUAL_BRIDGE_SAMPLE.json'

class VisualBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.src=json.loads(EX.read_text(encoding="utf-8"))
    def p(self):return copy.deepcopy(self.src)
    def test_valid_declared_observation(self):self.assertEqual(check(self.p())['schema'],'gg_math.visual_engine_bridge.v1')
    def test_unknown_future_never_gets_numeric_imputation(self):
        p=self.p();p['series'][0]['points'][-1]['value']=12
        with self.assertRaisesRegex(ValueError,'FICTITIOUS_FUTURE_MEASUREMENT'):check(p)
    def test_empty_denominator_is_invalid(self):
        p=self.p();p['series'][0]['points'][0]['eligible']=0
        with self.assertRaisesRegex(ValueError,'INVALID_DENOMINATOR'):check(p)
    def test_negative_observation_invalid(self):
        p=self.p();p['series'][0]['points'][0]['value']=-1
        with self.assertRaisesRegex(ValueError,'INVALID_OBSERVED_VALUE'):check(p)
    def test_duplicate_pulse_key_invalid(self):
        p=self.p();p['series'][0]['points'][1]['pulse']='N−1'
        with self.assertRaisesRegex(ValueError,'DUPLICATE_PULSE_KEY'):check(p)
    def test_unqualified_native_receipt_rejected(self):
        p=self.p();p['series'][0]['points'][0]['native_receipt']={'run_id':12}
        with self.assertRaisesRegex(ValueError,'MISMATCHED_NATIVE_RECEIPT_STATE'):check(p)
    def test_stale_or_malformed_source_rejected(self):
        p=self.p();p['source']['head_sha']='main'
        with self.assertRaisesRegex(ValueError,'EXACT_SOURCE_HEAD_REQUIRED'):check(p)
    def test_false_qps_approval_rejected(self):
        p=self.p();p['authority']['qps_owner_acceptance']=True
        with self.assertRaisesRegex(ValueError,'OWNER_PROMOTION'):check(p)
    def test_matplotlib_and_plotly_format_parity(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);m=emit(self.p(),folder);self.assertEqual(m,verify(folder))
            self.assertEqual(set(m['files']),{'series.svg','series.png','series-plotly.html','figure.json'})
            self.assertTrue((folder/'series.png').read_bytes().startswith(b'\x89PNG'))
            self.assertIn(b'<svg',(folder/'series.svg').read_bytes())
            self.assertNotIn(b'<script src="https://',(folder/'series-plotly.html').read_bytes())
            fig=json.loads((folder/'figure.json').read_text(encoding="utf-8"))
            self.assertAlmostEqual(fig['data'][0]['y'][0],600/42)
            self.assertIsNone(fig['data'][0]['y'][-1])
            self.assertIsNone(m['native_ci_attestation'])
    def test_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);emit(self.p(),folder)
            (folder/'series.png').write_bytes(b'fake_png')
            with self.assertRaisesRegex(ValueError,'TAMPERED_VISUAL_MEMBER'):verify(folder)

if __name__=='__main__':unittest.main()
