import unittest

from forgesight.vision_evaluate import detection_metrics


class DetectionMetricsTests(unittest.TestCase):
    def test_confusion_and_threshold_equality(self):
        result = detection_metrics([1, 1, 0, 0], [0.9, 0.5, 0.8, 0.1], 0.5)
        self.assertEqual([result[k] for k in ('tp', 'tn', 'fp', 'fn')], [1, 1, 1, 1])
        self.assertEqual(result['image_auroc'], 0.75)

    def test_tied_ranks(self):
        self.assertEqual(detection_metrics([0, 1], [1., 1.], 0.)['image_auroc'], 0.5)
        self.assertIsNone(detection_metrics([0], [0.], 1.)['recall'])

    def test_invalid_measurements(self):
        for labels, scores in [([], []), ([1], []), ([2], [0.]), ([0], [float('nan')])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                detection_metrics(labels, scores, 0.5)


if __name__ == '__main__':
    unittest.main()
