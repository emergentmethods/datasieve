# DataSieve

DataSieve is very similar to the SKlearn Pipeline in that it:

- fits an arbitrary series of transformations to an array X
- transforms subsequent arrays of the same dimension according to the fit from the original X
- inverse transforms arrays by inverting the series of transformations

This means that it follows the SKLearn API very closely, and in fact most of the methods inherit directly from SKLearn methods.

The main **difference** is that DataSieve allows for the manipulation of the y and sample_weight arrays in addition to the X array. This is useful if you find yourself wishing to use the SKLearn pipeline for:

- removing outliers across your X, y, and sample_weights arrays according to simple or complex criteria
- remove feature columns based on arbitrary criteria (e.g. low variance features)
- change feature column names at certain transformations (e.g. PCA)
- passing dynamic parameters to individual transforms of the pipeline
- passing dataframes/arrays without worrying about converting to arrays and maintaining the proper feature columns

These improved flexibilities allow for more customized/creative transformations. For example, the included `DataSieveDBSCAN` has automated parameter fitting and outlier removal based on clustering. 

An example would be someone who wants to use `SGDOneClassSVM` to detect and remove outliers from their data set before training:

```python
class SVMOutlierExtractor(SGDOneClassSVM):
    """
    A subclass of the SKLearn SGDOneClassSVM that adds a transform() method
    for removing detected outliers from X (as well as the associated y and
    sample_weight if they are also furnished.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X, y, sample_weight=sample_weight)

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        super().fit(X, y=y, sample_weight=sample_weight)
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        y_pred = self.predict(X)

        X, y, sample_weight = remove_outliers(X, y, sample_weight, y_pred)

        num_tossed = len(y_pred) - len(X)
        if num_tossed > 0:
            logger.info(
                f"SVM detected {num_tossed} data points"
                "as outliers."
            )

        return X, y, sample_weight, feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        """
        Unused, pass through X, y, sample_weight, and feature_list
        """
        return X, y, sample_weight, feature_list
```


As shown here, the `fit()` method is actually identical to the SKLearn `fit()` method, but the `transform()` removes data points from X, y, and sample_weight for any outliers detected in the `X` array.


# Usage
The user builds the pipeline similarly to SKLearn:

```python
    from datasieve.pipeline import Pipeline
    from datasieve.transforms import DataSieveMinMaxScaler, DataSievePCA, DataSieveVarianceThreshold, SVMOutlierExtractor

    feature_pipeline = Pipeline([
        ("detect_constants", DataSieveVarianceThreshold(threshold=0)),
        ("pre_svm_scaler", DataSieveMinMaxScaler(feature_range=(-1, 1)))
        ("svm", SVMOutlierExtractor()),
        ("pre_pca_scaler", DataSieveMinMaxScaler(feature_range=(-1, 1)))
        ("pca", DataSievePCA(n_components=0.95),
        ("post_pca_scaler", DataSieveMinMaxScaler(feature_range=(-1, 1)))
    ])

```

Once the pipeline is built, it can be fit and transformed similar to a SKLearn pipeline:

```python
X, y, sample_weight = feature_pipeline.fit_transform(X, y, sample_weight)
```

This pipeline demonstrates the various components of `DataSieve` which are missing from SKLearn's pipeline. A dataframe `X` (if desired, else users can input a simply array without column names) is input with its associated `y` and `sample_weight` arrays/vectors. The `VarianceThreshold` will first detect and remove any features that have zero variance in X, the `SVMOutlierExtractor` will fit `SGDOneClassSVM` to `X` and then remove the detected outliers in `X`, while also propagating those row removals from `y` and `sample_weight`. Finally, the `PCA` will be fit to the remaining `X` array with the features count changing and getting renamed. The returned `X` dataframe will have the correctly named column features/count, and equal row counts across the `X`, `y`, and `sample_weight` arrays.

Next, the `feature_pipeline` can then be used to transform other datasets with the same input feature dimension:

```python
X2, _, _ = feature_pipeline.transform(X2)

```

Finally, similar to SKLearn's pipeline, the `feature_pipeline` can be used to inverse_transform an array `X3` array that has the same dimensions as the returned `X` array from the pipeline:

```python
Xinv, _ ,_ = feature_pipeline.inverse_transform(X)
```

# Installation

The easiest way to install `datasieve` is with:

```
pip install datasieve
```

but you can also clone this repository and install it with:

```
git clone https://github.com/emergentmethods/datasieve.git
cd datasieve
poetry install
```

# License

Copyright (c) 2023 DataSieve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.