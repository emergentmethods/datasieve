import datasieve.transforms as transforms
from datasieve.transforms.sklearn_wrapper import SKLearnWrapper
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def test_min_max_scaler(dummy_array_without_nans):
    """
    Test the min max scaler
    """
    X = dummy_array_without_nans.copy()
    scaler = SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))
    X, _, _, _ = scaler.fit_transform(X)
    Y = dummy_array_without_nans.copy()
    Y, _, _, _ = scaler.transform(Y)

    Xinv, _, _, _ = scaler.inverse_transform(X)
    assert (X.max().max() - 1) / 1 < 1e-6
    assert (X.min().min() + 1) / -1 < 1e-6

    assert Y[0, 0] == X[0, 0]
    assert Xinv[0, 0] == dummy_array_without_nans[0, 0]


def test_pca(dummy_array_without_nans):
    """
    Test the pca
    """
    X = dummy_array_without_nans.copy()
    pca = transforms.PCA(n_components=0.95)
    X, _, _, _ = pca.fit_transform(X)
    Y = dummy_array_without_nans.copy()
    Y, _, _, _ = pca.transform(Y)

    Xinv, _, _, _ = pca.inverse_transform(X)

    assert X.shape[1] == Y.shape[1]
    assert Xinv.shape[1] == dummy_array_without_nans.shape[1]


def test_DBSCAN(dummy_array_without_nans):
    """
    Test the DBSCAN
    """
    X = dummy_array_without_nans.copy()
    dbscan = transforms.DBSCAN()
    X, _, _, _ = dbscan.fit_transform(X)
    Y = dummy_array_without_nans.copy()
    Y, _, _, _ = dbscan.transform(Y)

    assert X.shape[0] < dummy_array_without_nans.shape[0]
    assert Y.shape[0] == 73


def test_variance_threshold(dummy_array_without_nans):
    """
    Test the variance threshold
    """
    X = dummy_array_without_nans.copy()
    varthresh = transforms.VarianceThreshold(threshold=0.075)
    X, _, _, _ = varthresh.fit_transform(X)
    Y = dummy_array_without_nans.copy()
    Y, _, _, _ = varthresh.transform(Y)

    assert X.shape[1] < dummy_array_without_nans.shape[1]
    print(Y.shape)
    assert Y.shape[1] == 177


def test_dissimilarity_index(dummy_array_without_nans, dummy_array2_without_nans):
    """
    Test the dissimilarity index
    """
    X = dummy_array_without_nans.copy()
    dissimilarity_index = transforms.DissimilarityIndex(di_threshold=0.9)
    X, _, _, _ = dissimilarity_index.fit_transform(X)
    Y = dummy_array2_without_nans.copy()
    Y, _, _, _ = dissimilarity_index.transform(Y)

    assert Y.shape[0] == 51


def test_svm_outlier_extractor(dummy_array_without_nans, dummy_array2_without_nans):
    """
    Test the svm outlier extractor
    """
    X = dummy_array_without_nans.copy()
    svm_outlier_extractor = transforms.SVMOutlierExtractor()
    X, _, _, _ = svm_outlier_extractor.fit_transform(X)
    Y = dummy_array2_without_nans.copy()
    Y, _, _, _ = svm_outlier_extractor.transform(Y)

    assert X.shape[0] < dummy_array_without_nans.shape[0]
    assert Y.shape[0] == 81


def test_noise(dummy_array_without_nans):
    """
    Test Noise
    """
    X = dummy_array_without_nans.copy()
    noise = transforms.Noise()
    Xnoisey, _, _, _ = noise.fit_transform(X)

    Xtest, _, _, _ = noise.transform(dummy_array_without_nans)

    assert np.sum(Xnoisey - dummy_array_without_nans) != 0.0
    assert np.sum(Xtest - dummy_array_without_nans) == 0.0