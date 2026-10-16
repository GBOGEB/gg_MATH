"""Native MANOVA reference math, invariance and negative diagnostics."""
import math
import unittest

from kernels.manova_reference import manova_reference


def population():
    patterns = [
        [float(i)*0.6 + (i % 3), float((i % 5)*2) - i*0.12,
         float((i % 7)*1.3) + i*0.04] for i in range(15)
    ]
    return [
        [[v + (5.0 if col == 0 else j * 0.4)
          for col, v in enumerate(row)] for row in patterns]
        for j in range(3)
    ]


class ManovaReferenceTests(unittest.TestCase):
    def test_positive_effect_and_no_unearned_p_value(self):
        r = manova_reference(population())
        self.assertEqual(r["n"], 45)
        self.assertEqual(r["p"], 3)
        self.assertEqual(r["groups"], 3)
        self.assertEqual(r["df_within"], 42)
        self.assertLess(r["wilks_lambda"], 1.0)
        self.assertGreater(r["pillai_trace"], 0)
        self.assertLessEqual(r["pillai_trace"], 2.0)
        self.assertIsNone(r["p_value"])
        self.assertIsNone(r["independence_authenticated"])
        self.assertFalse(r["authority_transfer"])

    def test_null_group_means_have_near_zero_between_sscp(self):
        gs = population()
        for j in range(1, len(gs)):
            gs[j] = [row[:] for row in gs[0]]
        r = manova_reference(gs)
        self.assertAlmostEqual(r["wilks_lambda"], 1.0, places=10)
        self.assertAlmostEqual(r["pillai_trace"], 0.0, places=10)

    def test_invariant_to_group_permutation_and_row_order(self):
        gs = population()
        baseline = manova_reference(gs)
        changed = [list(reversed(gs[2])), list(reversed(gs[0])), gs[1]]
        other = manova_reference(changed)
        self.assertAlmostEqual(baseline["wilks_lambda"], other["wilks_lambda"],
                               places=10)
        self.assertAlmostEqual(baseline["pillai_trace"], other["pillai_trace"],
                               places=10)

    def test_invariant_to_common_translation_and_nonsingular_scale(self):
        gs = population()
        baseline = manova_reference(gs)
        changed = [[[3.0*r[0]+100.0, -2.0*r[1]-5.0, 0.25*r[2]+7.0]
                    for r in group] for group in gs]
        other = manova_reference(changed)
        self.assertAlmostEqual(baseline["wilks_lambda"], other["wilks_lambda"],
                               places=9)
        self.assertAlmostEqual(baseline["pillai_trace"], other["pillai_trace"],
                               places=9)

    def test_reject_singular_within_group_covariance(self):
        gs = population()
        invalid = [[[row[0], row[1], row[0] * 2] for row in group]
                   for group in gs]
        with self.assertRaisesRegex(ValueError, "singular residual"):
            manova_reference(invalid)

    def test_reject_invalid_shape_population_and_nonfinite(self):
        invalids = (
            [],
            [[[1, 2]], [[3, 4]]],
            [[[1, 2], [2, 4]], [[4, 6], [5, 8]]],
            [[[1, 2], [2, 3]], [[1, 2, 3], [2, 3, 4]]],
            [[[1, True], [2, 3]], [[4, 5], [6, 7]]],
            [[[1, math.nan], [2, 3]], [[4, 5], [6, 7]]],
        )
        for groups in invalids:
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    manova_reference(groups)

    def test_no_fake_inference_from_duplicating_rows(self):
        gs = population()
        expanded = [group + [row[:] for row in group] for group in gs]
        r = manova_reference(expanded)
        self.assertEqual(r["n"], 90)
        self.assertIsNone(r["p_value"])
        self.assertIsNone(r["covariance_homogeneity_authenticated"])

    def test_malformed_input_types_rejected(self):
        for row in ((1, 2, 3), "not a group", [0, 1]):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    manova_reference([row, population()[1]])


if __name__ == "__main__":
    unittest.main()
