import unittest

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix as sk_cm

from metrics import accuracy, confusion_matrix


class TestMetrics(unittest.TestCase):
    def test_metrics_match_sklearn(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 10, 1000)
        y_pred = rng.integers(0, 10, 1000)
        labels = np.arange(10)

        self.assertAlmostEqual(accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))
        np.testing.assert_array_equal(
            confusion_matrix(y_true, y_pred),
            sk_cm(y_true, y_pred, labels=labels),
        )

    def test_confusion_matrix_keeps_all_classes(self):
        y_true = np.array([0, 1, 1, 2])
        y_pred = np.array([0, 2, 1, 2])
        labels = np.arange(10)

        result = confusion_matrix(y_true, y_pred)
        expected = sk_cm(y_true, y_pred, labels=labels)

        self.assertEqual(result.shape, (10, 10))
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()