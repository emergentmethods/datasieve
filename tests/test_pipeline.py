from datasieve.pipeline import Pipeline
import datasieve.transforms as transforms
from conftest import extract_features_and_labels, set_weights_higher_recent
import numpy as np


def test_pipeline_df_different_features_in_out(dummy_df_without_nans):
    """
    Create pipeline and pass a dummy df through
    """

    pipeline = Pipeline([
        ("detect_constants", transforms.DataSieveVarianceThreshold(threshold=0)),
        ("pre_svm_scaler", transforms.DataSieveMinMaxScaler(feature_range=(-1, 1))),
        ("svm", transforms.SVMOutlierExtractor()),
        ("pre_pca_scaler", transforms.DataSieveMinMaxScaler(feature_range=(-1, 1))),
        ("pca", transforms.DataSievePCA(n_components=0.95)),
        ("post_pca_scaler", transforms.DataSieveMinMaxScaler(feature_range=(-1, 1)))
    ])

    df = dummy_df_without_nans.copy()
    Xdf, ydf = extract_features_and_labels(df)
    weights = set_weights_higher_recent(Xdf.shape[0], 0.5)

    Xdf_t, ydf_t, weights_t = pipeline.fit_transform(Xdf, ydf, sample_weight=weights)

    assert Xdf_t.shape[0] < Xdf.shape[0]
    assert Xdf_t.shape[1] < Xdf.shape[1]
    assert ydf_t.shape[0] == Xdf_t.shape[0]
    assert weights_t.shape[0] == Xdf_t.shape[0]


def test_pipeline_df_same_features_in_out(dummy_df_without_nans):
    pipeline = Pipeline([
        ("pre_svm_scaler", transforms.DataSieveMinMaxScaler()),
        ("svm", transforms.SVMOutlierExtractor())
    ])

    df = dummy_df_without_nans.copy()
    Xdf, ydf = extract_features_and_labels(df)
    weights = set_weights_higher_recent(Xdf.shape[0], 0.5)

    Xdf_t, ydf_t, _ = pipeline.fit_transform(Xdf, ydf, sample_weight=weights)

    # check if two lists are identical
    assert Xdf_t.columns.tolist() == Xdf.columns.tolist()
    assert ydf_t.columns.tolist() == ydf.columns.tolist()


def test_pipeline_array_in_out(dummy_array_without_nans):
    pipeline = Pipeline([
        ("pre_svm_scaler", transforms.DataSieveMinMaxScaler()),
        ("svm", transforms.SVMOutlierExtractor())
    ])

    X = dummy_array_without_nans.copy()
    Y = X[:, -1]
    X = X[:, :-1]

    X_t, Y_t, _ = pipeline.fit_transform(X, Y)

    assert type(X_t) == np.ndarray
    assert type(Y_t) == np.ndarray


def test_check_outliers(dummy_array_without_nans):
    """
    Create pipeline and pass a dummy df through
    """

    pipeline = Pipeline([
        ("pre_svm_scaler", transforms.DataSieveMinMaxScaler(feature_range=(-1, 1))),
        ("svm", transforms.SVMOutlierExtractor(nu=0.01, shuffle=True, random_state=42))
    ])

    X = dummy_array_without_nans.copy()

    pipeline.fit(X)

    X, outliers, _ = pipeline.transform(X, outlier_check=True)

    assert X.shape[0] == dummy_array_without_nans.shape[0]
    assert outliers.sum() == 21
    assert outliers.sum() < X.shape[0]
