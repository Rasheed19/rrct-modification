from rrct.algorithm import RRCTFeatureSelection
import numpy as np
from sklearn.datasets import load_diabetes


def main():
    # Note that the features have been scaled see here
    # https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    # get feature names 
    feature_names = list(X.columns)
    feature_names = np.array(feature_names)

    # convert to numpy arrays
    X, y = X.values, y.values

    rrct = RRCTFeatureSelection(
            K=None,
            scale_feature=False
        )
    rrct.apply(X, y)
    selected_features = rrct.select(X)

    print(f"ranked features: {feature_names[rrct.selected_features_indices_]}", selected_features)

if __name__ == "__main__":
    main()

