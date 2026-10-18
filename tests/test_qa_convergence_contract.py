"""Pure regression tests for six-card synthetic vs nine-card observed G3P browser QA."""
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("qa_convergence_html.py")
spec = importlib.util.spec_from_file_location("g3p_qa", MODULE_PATH)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


class FakeLocator:
    def __init__(self, count):
        self.value = count

    def count(self):
        return self.value


class FakePage:
    def __init__(self, count):
        self.value = count
        self.selectors = []

    def locator(self, selector):
        self.selectors.append(selector)
        return FakeLocator(self.value)


class CardResetContract(unittest.TestCase):
    def test_synthetic_contract_six(self):
        self.assertEqual(qa.expected_card_count(observed=False), 6)

    def test_observed_contract_nine(self):
        self.assertEqual(qa.expected_card_count(observed=True), 9)

    def test_observed_reset_nine_passes(self):
        page = FakePage(9)
        self.assertEqual(qa.assert_visible_card_count(page, 9, "filter reset"), 9)
        self.assertEqual(page.selectors, [".metric-card:visible"])

    def test_synthetic_reset_six_passes(self):
        self.assertEqual(qa.assert_visible_card_count(FakePage(6), 6, "filter reset"), 6)

    def test_observed_reset_six_fails(self):
        with self.assertRaisesRegex(AssertionError, "filter reset: visible metric cards 6, expected 9"):
            qa.assert_visible_card_count(FakePage(6), 9, "filter reset")

    def test_observed_surplus_fails(self):
        with self.assertRaisesRegex(AssertionError, "10, expected 9"):
            qa.assert_visible_card_count(FakePage(10), 9, "initial render")

    def test_synthetic_surplus_fails(self):
        with self.assertRaises(AssertionError):
            qa.assert_visible_card_count(FakePage(9), 6, "filter reset")


if __name__ == "__main__":
    unittest.main()
