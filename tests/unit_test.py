from rrct.algorithm import RRCTFeatureSelection
import unittest
import numpy as np
from sklearn.datasets import load_diabetes

# Note that the features have been scaled see here
# https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)

# get feature names 
feature_names = list(X.columns)

# convert to numpy arrays
X, y = X.values, y.values


class TestOutputDimension(unittest.TestCase):

    def test_when_k_is_none(self):
        rrct = RRCTFeatureSelection(
            K=None,
            scale_feature=False
        )
        rrct.apply(X, y)
        selected_features = rrct.select(X)

        self.assertEqual(
            len(rrct.selected_features_indices_),
            len(feature_names)
        )

        self.assertEqual(
            selected_features.shape,
            X.shape
        )

    def test_when_k_is_nonzero(self):
        rrct = RRCTFeatureSelection(
            K=2,
            scale_feature=False
        )
        rrct.apply(X, y)
        selected_features = rrct.select(X)

        self.assertEqual(
            len(rrct.selected_features_indices_),
            2
        )

        self.assertEqual(
            selected_features.shape,
            (X.shape[0],  2)
        )

class TestInvalidInput(unittest.TestCase):

    def test_when_k_greater_than_M(self):
        rrct = RRCTFeatureSelection(
            K=X.shape[1]+2,
            scale_feature=False
        )
        
        self.assertRaises(
            ValueError,
           rrct.apply,
            X, y
        )
    
    def test_when_k_is_zero(self):
        rrct = RRCTFeatureSelection(
            K=0,
            scale_feature=False
        )
        
        self.assertRaises(
            ValueError,
            rrct.apply,
            X, y
        )

    def test_when_k_less_than_zero(self):
        rrct = RRCTFeatureSelection(
            K=-2,
            scale_feature=False
        )
        
        self.assertRaises(
            ValueError,
            rrct.apply,
            X, y
        )
    
    # add demo.py in tests folder
    
    def test_when_type_k_is_invalid(self):
        rrct = RRCTFeatureSelection(
            K=2.2,
            scale_feature=False
        )
        
        self.assertRaises(
            TypeError,
            rrct.apply,
            X, y

        )

class TestApplySelect(unittest.TestCase):

    def test_output_dimension_test(self):
        rrct = RRCTFeatureSelection(
            K=3,
            scale_feature=False
        )
        selected_features = rrct.apply_select(X, y)

        self.assertEqual(
            len(rrct.selected_features_indices_),
            3
        )

        self.assertEqual(
            selected_features.shape,
            (X.shape[0], 3)
        )


if __name__ == '__main__':
    unittest.main()