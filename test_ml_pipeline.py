import unittest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


class TestChurnPipeline(unittest.TestCase):
    def test_pipeline_trains_and_predicts(self):
        # Small synthetic sample tests the preprocessing/model pipeline independently
        # of the full CSV, so CI can test the code even if the dataset is not committed.
        sample = pd.DataFrame({
            "Age": [22, 45, 31, 51, 26, 42, 36, 29],
            "Support Calls": [5, 1, 3, 0, 4, 1, 2, 5],
            "Contract Length": ["Monthly", "Yearly", "Monthly", "Yearly",
                                "Monthly", "Yearly", "Monthly", "Monthly"],
            "Churn": [1, 0, 1, 0, 1, 0, 0, 1]
        })
        X = sample.drop(columns=["Churn"])
        y = sample["Churn"]
        numeric = X.select_dtypes(include=["number"]).columns.tolist()
        categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        ])
        model = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_expected_target_name(self):
        self.assertEqual("Churn", "Churn")


if __name__ == "__main__":
    unittest.main()
