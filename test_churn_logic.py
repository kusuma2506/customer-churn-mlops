
import unittest
from churn_logic import predict_churn

class TestChurn(unittest.TestCase):
    def test_high_risk(self):
        self.assertEqual(predict_churn(3, 5), 1)

    def test_low_risk(self):
        self.assertEqual(predict_churn(24, 1), 0)

if __name__ == "__main__":
    unittest.main()
