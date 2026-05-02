import unittest
import numpy as np

class TestDrivingSystem(unittest.TestCase):

    def test_sequence_shape(self):
        data = np.random.rand(10, 6)
        self.assertEqual(data.shape, (10, 6))

    def test_feature_extraction(self):
        data = np.random.rand(10, 6)
        features = np.concatenate([data.mean(axis=0), data.std(axis=0)])
        self.assertEqual(features.shape[0], 12)

if __name__ == "__main__":
    unittest.main()