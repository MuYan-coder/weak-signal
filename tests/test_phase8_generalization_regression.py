import unittest

from src.domain.domain_pack_regression import (
    PHASE8_REQUIRED_DIMENSIONS,
    evaluate_phase8_regression_matrix,
)


class Phase8GeneralizationRegressionTest(unittest.TestCase):
    def test_phase8_regression_matrix_covers_every_documented_dimension(self):
        report = evaluate_phase8_regression_matrix()

        self.assertTrue(report.is_complete, report.missing_dimensions)
        self.assertEqual(set(report.covered_dimensions), set(PHASE8_REQUIRED_DIMENSIONS))
        for dimension in report.dimensions:
            self.assertTrue(dimension.test_modules, dimension.dimension_id)
            self.assertTrue(dimension.acceptance_metric, dimension.dimension_id)


if __name__ == "__main__":
    unittest.main()
