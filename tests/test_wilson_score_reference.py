"""Reference numerical, metamorphic, boundary and adversarial controls."""
from __future__ import annotations

import unittest

from kernels.stats_core import wilson_proportion_interval as wilson


class WilsonReferenceTests(unittest.TestCase):
    def test_normal_reference_90_95_99_for_five_of_ten(self):
        targets = {0.90: (0.269271, 0.730729),
                   0.95: (0.236593, 0.763407),
                   0.99: (0.184226, 0.815774)}
        for level, (lower, upper) in targets.items():
            with self.subTest(level=level):
                r = wilson(5, 10, level=level)
                self.assertAlmostEqual(r['interval'][0], lower, places=5)
                self.assertAlmostEqual(r['interval'][1], upper, places=5)
                self.assertEqual(r['estimate'], 0.5)
                self.assertIsNone(r['independence_authenticated'])
                self.assertFalse(r['authority_transfer'])

    def test_level_nesting_for_interior_and_edges(self):
        for n, k in ((1, 0), (1, 1), (10, 0), (10, 10),
                     (10, 5), (100, 27)):
            bands = [wilson(k, n, level=level)['interval']
                     for level in (.90, .95, .99)]
            self.assertLessEqual(bands[2][0], bands[1][0] + 1e-14)
            self.assertLessEqual(bands[1][0], bands[0][0] + 1e-14)
            self.assertLessEqual(bands[0][1], bands[1][1] + 1e-14)
            self.assertLessEqual(bands[1][1], bands[2][1] + 1e-14)
            self.assertTrue(all(0 <= lo <= hi <= 1 for lo, hi in bands))

    def test_complement_symmetry_metaprop(self):
        for n in (1, 2, 3, 7, 20, 100):
            for k in (0, n // 2, n):
                for level in (.90, .95, .99):
                    lo, hi = wilson(k, n, level=level)['interval']
                    vlo, vhi = wilson(n-k, n, level=level)['interval']
                    self.assertAlmostEqual(lo, 1-vhi, places=12)
                    self.assertAlmostEqual(hi, 1-vlo, places=12)

    def test_interval_shrinks_with_more_identical_rate_trials(self):
        for level in (.90, .95, .99):
            widths = []
            for n in (10, 20, 50, 100):
                lo, hi = wilson(n//2, n, level=level)['interval']
                widths.append(hi-lo)
            self.assertEqual(widths, sorted(widths, reverse=True))

    def test_reject_malformed_counts_and_levels(self):
        for k, n in ((-1, 3), (4, 3), (0, 0), (True, 2),
                     (1, True), (1.0, 2), ('1', 2), (1, -1)):
            with self.subTest(k=k, n=n):
                with self.assertRaises(ValueError):
                    wilson(k, n)
        for level in (True, False, 0.0, .89, .999, 1.0, float('nan'), '0.95'):
            with self.subTest(level=level):
                with self.assertRaises(ValueError):
                    wilson(1, 2, level=level)

    def test_no_confidence_claim_from_numeric_reference(self):
        output = wilson(26, 30)
        self.assertEqual(output['status'], 'REFERENCE_ONLY_DESIGN_UNVERIFIED')
        self.assertNotIn('engineering_accepted', output)
        self.assertNotIn('merge_authorized', output)
        self.assertIn('IID', output['assumptions'])


if __name__ == '__main__':
    unittest.main()
