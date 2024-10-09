from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from rrct import RRCTFeatureSelection


@dataclass(frozen=True)
class FixtureData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


@pytest.fixture
def load_data() -> FixtureData:
    # Note that the features have been scaled see here
    # https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    feature_names = list(X.columns)

    X, y = X.values, y.values

    return FixtureData(X=X, y=y, feature_names=feature_names)


def test_output_dim(load_data) -> None:
    # when K is None
    selector = RRCTFeatureSelection(K=None, scale_feature=False)
    selector.apply(load_data.X, load_data.y)
    selected_features = selector.select(load_data.X)
    assert len(selector.selected_features_indices_) == len(load_data.feature_names)
    assert selected_features.shape == load_data.X.shape

    # when k is non-zero
    selector = RRCTFeatureSelection(K=2, scale_feature=False)
    selector.apply(load_data.X, load_data.y)
    selected_features = selector.select(load_data.X)
    assert len(selector.selected_features_indices_) == 2
    assert selected_features.shape == (load_data.X.shape[0], 2)

    return None


def test_invalid_input(load_data) -> None:
    # when k > M
    with pytest.raises(ValueError):
        selector = RRCTFeatureSelection(K=load_data.X.shape[1] + 2, scale_feature=False)
        selector.apply(load_data.X, load_data.y)

    # when k = 0
    with pytest.raises(ValueError):
        selector = RRCTFeatureSelection(K=0, scale_feature=False)
        selector.apply(load_data.X, load_data.y)

    # when k < 0
    with pytest.raises(ValueError):
        selector = RRCTFeatureSelection(K=-2, scale_feature=False)
        selector.apply(load_data.X, load_data.y)

    # when k is float
    with pytest.raises(TypeError):
        selector = RRCTFeatureSelection(K=2.2, scale_feature=False)
        selector.apply(load_data.X, load_data.y)

    return None


def test_apply_select(load_data) -> None:
    selector = RRCTFeatureSelection(K=3, scale_feature=False)
    selected_features = selector.apply_select(load_data.X, load_data.y)
    assert len(selector.selected_features_indices_) == 3
    assert selected_features.shape == (load_data.X.shape[0], 3)

    return None
