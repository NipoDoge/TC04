import unittest

import numpy as np

from src import config
from src.batch_drift import status_for
from src.drift_stats import bin_edges, ks, psi


class PsiTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.reference = self.rng.lognormal(8.5, 0.6, 20000)       # cauda longa, como a renda

    def test_identical_samples_have_zero_psi(self):
        self.assertAlmostEqual(psi(self.reference, self.reference), 0.0, places=12)

    def test_same_distribution_stays_far_below_warning(self):
        current = self.rng.lognormal(8.5, 0.6, 4000)
        self.assertLess(psi(self.reference, current), config.PSI_WARNING / 5)

    def test_shift_crosses_alert_threshold(self):
        shifted = self.rng.lognormal(8.5, 0.6, 4000) * 0.6          # renda 40% menor
        self.assertGreater(psi(self.reference, shifted), config.PSI_ALERT)

    def test_psi_grows_with_the_size_of_the_shift(self):
        values = [psi(self.reference, self.reference * factor) for factor in (1.0, 0.9, 0.7, 0.5)]
        self.assertEqual(values, sorted(values))

    def test_bins_are_reference_deciles(self):
        edges = bin_edges(self.reference)
        self.assertEqual(len(edges), 11)
        self.assertTrue(np.isinf(edges[[0, -1]]).all())
        shares = np.histogram(self.reference, edges)[0] / len(self.reference)
        np.testing.assert_allclose(shares, 0.1, atol=0.001)

    def test_low_cardinality_uses_one_bin_per_value(self):
        reference = self.rng.poisson(0.3, 20000)                    # contagem de atrasos
        self.assertEqual(len(bin_edges(reference)), len(np.unique(reference)) + 1)
        self.assertLess(psi(reference, self.rng.poisson(0.3, 4000)), 0.01)
        self.assertGreater(psi(reference, self.rng.poisson(1.0, 4000)), config.PSI_ALERT)

    def test_empty_bin_in_production_gives_finite_psi(self):
        reference = np.repeat([0.0, 1.0, 2.0], [700, 200, 100])
        value = psi(reference, np.zeros(500))                       # faixas de 1 e 2 sem nenhuma linha
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, config.PSI_ALERT)

    def test_out_of_range_values_fall_into_edge_bins(self):
        # Limitação documentada: as faixas das pontas são abertas. Um valor acima do
        # máximo da referência conta na última faixa e só aparece se mudar a massa dela.
        reference = np.repeat([0.0, 1.0], [900, 100])
        self.assertEqual(psi(reference, np.repeat([0.0, 7.0], [900, 100])), 0.0)
        self.assertGreater(psi(reference, np.repeat([0.0, 7.0], [600, 400])), config.PSI_ALERT)

    def test_empty_sample_is_an_error(self):
        for reference, current in [([], [1.0]), ([1.0], [])]:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                psi(reference, current)

    def test_status_follows_configured_thresholds(self):
        cases = [(0.0, "ok"), (config.PSI_WARNING - 1e-9, "ok"), (config.PSI_WARNING, "warning"),
                 (config.PSI_ALERT - 1e-9, "warning"), (config.PSI_ALERT, "alert"), (1.0, "alert")]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(status_for(value), expected)


class KsTests(unittest.TestCase):
    def test_same_sample_and_shifted_sample(self):
        rng = np.random.default_rng(7)
        sample = rng.normal(0, 1, 5000)
        statistic, pvalue = ks(sample, sample)
        self.assertEqual((statistic, pvalue), (0.0, 1.0))
        statistic, pvalue = ks(sample, rng.normal(0.5, 1, 5000))
        self.assertGreater(statistic, 0.15)
        self.assertLess(pvalue, config.KS_ALPHA)


if __name__ == "__main__":
    unittest.main()
