import unittest
from kernels.bradley_terry import rank_pairs
from kernels.typed_relational_bridge import typed_bt_pca_bridge


class H05Regression(unittest.TestCase):
    def test_complete_separation(self):
        self.assertEqual(rank_pairs([('A','B')]*4)['status'], 'DEFER_NO_FINITE_MLE')

    def test_separated_blocks_with_each_item_winning(self):
        pairs=[('A','B'),('B','A'),('C','D'),('D','C'),('B','C')]
        self.assertEqual(rank_pairs(pairs)['scores'], {})
        self.assertEqual(rank_pairs(pairs)['status'], 'DEFER_NO_FINITE_MLE')

    def test_cycle_has_finite_solution_without_reverse_pair_edges(self):
        r=rank_pairs([('A','B'),('B','C'),('C','A')])
        self.assertEqual(r['status'],'PASS_TESTABLE_ENGINE')
        for value in r['scores'].values(): self.assertAlmostEqual(value,1)

    def test_two_item_analytic_probability(self):
        r=rank_pairs([('A','B')]*3+[('B','A')])['scores']
        self.assertAlmostEqual(r['A']/(r['A']+r['B']),.75)

    def test_bridge_rejects_malformed_records(self):
        for pair in [('A','B','ignored'),('A',),(), 'AB', None]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                typed_bt_pca_bridge(observed_pairs=iter([pair]),population_id='fixture',evidence_class='SYNTHETIC')

    def test_separation_propagates_defer(self):
        r=typed_bt_pca_bridge(observed_pairs=[('A','B')],population_id='fixture',evidence_class='SYNTHETIC')
        self.assertTrue(r['consumer_disposition'].startswith('DEFER'))
        self.assertEqual(r['bt']['result']['status'],'DEFER_NO_FINITE_MLE')
