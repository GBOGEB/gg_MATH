"""Provider-side N5 native bridge verification (public synthetic data only)."""
from __future__ import annotations
import copy
import json
import unittest
from pathlib import Path
from mission.federation.n5_repo_graph_math import evaluate, typed_graph, pca_descriptive, canonical
HERE=Path(__file__).resolve().parents[1]
FIXTURE=json.loads((HERE/'mission/federation/qps_n5_synthetic_input.json').read_text(encoding='utf-8'))
def provider():
    from kernels.matrix_core import matmul,symmetric_eigen_jacobi
    from kernels.pca_reference import pca_reference
    from kernels.stats_core import one_way_anova
    from kernels.manova_reference import manova_reference
    from kernels.uncertainty import mean_interval
    return dict(matmul=matmul,eigen=symmetric_eigen_jacobi,pca=pca_reference,anova=one_way_anova,manova=manova_reference,ci=mean_interval)
class QualifiedProvider(unittest.TestCase):
    def test_01_native_provider_executes_all_five_reference_families(self):
        r=evaluate(FIXTURE,provider=provider())
        self.assertEqual(r['provider'],'EXECUTED_AGAINST_GG_MATH_KERNELS')
        self.assertGreater(r['inferential_reference']['anova']['f_statistic'],0)
        self.assertIsNone(r['inferential_reference']['anova']['p_value'])
        self.assertIsNone(r['inferential_reference']['manova']['p_value'])
        self.assertFalse(r['inferential_reference']['mean_interval']['qualified_for_clock_inference'])
        self.assertEqual(r['formal_credit_delta'],0)
    def test_02_deterministic_exact_reference(self):
        a=evaluate(FIXTURE,provider=provider());b=evaluate(FIXTURE,provider=provider())
        self.assertEqual(canonical(a),canonical(b))
    def test_03_hard_dag_nilpotency_and_incidence_identity(self):
        g=evaluate(FIXTURE)['graphs']
        self.assertTrue(g['hard_power_n_zero'])
        self.assertLess(g['incidence_unnormalized_laplacian_inf_residual'],1e-10)
        self.assertLess(g['orthogonal_eigenbasis_frobenius_residual'],1e-10)
        self.assertLess(g['directed_hard_adjacency_max_abs_eigen'],1e-10)
        self.assertEqual(g['incidence_shape'],[6,6])
    def test_04_eigen_spectral_observables_bounded(self):
        g=evaluate(FIXTURE)['graphs']
        self.assertGreater(g['spectral_gap_lambda2'],0)
        self.assertGreaterEqual(min(g['normalized_laplacian_eigenvalues']),-1e-9)
    def test_05_padding_strict_zero_not_missing(self):
        x=copy.deepcopy(FIXTURE);del x['time_rows'][0]['repo_alpha']
        with self.assertRaisesRegex(ValueError,'missing feature'):evaluate(x)
    def test_06_boolean_authority_forbidden(self):
        x=copy.deepcopy(FIXTURE);x['authority_transfer']=True
        with self.assertRaisesRegex(ValueError,'authority'):evaluate(x)
    def test_07_private_population_forbidden(self):
        x=copy.deepcopy(FIXTURE);x['data_class']='QPS_PRIVATE'
        with self.assertRaisesRegex(ValueError,'synthetic'):evaluate(x)
    def test_08_dangling_link_forbidden(self):
        x=copy.deepcopy(FIXTURE);x['edges'][0]['source']='ghost'
        with self.assertRaisesRegex(ValueError,'dangling'):evaluate(x)
    def test_09_hard_cycle_forbidden(self):
        x=copy.deepcopy(FIXTURE);x['edges'].append({'source':'review','target':'source','hard':True})
        with self.assertRaisesRegex(ValueError,'cycle'):evaluate(x)
    def test_10_duplicate_rows_are_not_independent(self):
        r=evaluate(FIXTURE)['padding']
        self.assertFalse(r['bootstrap_independent_n_claimed'])
        self.assertFalse(r['imputations_admitted_into_observed'])
        self.assertEqual(r['seed'],314159)
    def test_11_aligned_zero_variance_excluded(self):
        p=pca_descriptive([[1,1,3],[2,2,3],[4,5,3],[7,8,3],[9,9,3]],['a','b','constant'])
        self.assertEqual(p['excluded_zero_variance_features'],['constant'])
    def test_12_observed_pca_small_n_not_inference(self):
        r=evaluate(FIXTURE)['pca']
        self.assertLessEqual(r['rank_ceiling'],min(r['n']-1,r['p']))
        self.assertEqual(r['inference'],'DESCRIPTIVE_ONLY')
    def test_13_source_only_cannot_claim_provider_run(self):
        self.assertEqual(evaluate(FIXTURE)['provider'],'NOT_EXECUTED_BY_PROVIDER')
    def test_14_corrupted_matmul_fails_closed(self):
        p=provider();p['matmul']=lambda a,b:[[0]*len(a) for _ in a]
        with self.assertRaisesRegex(ValueError,'matrix multiplication mismatch'):evaluate(FIXTURE,provider=p)
if __name__=='__main__':unittest.main()
