# Multiclass Classification Using the Iris Dataset

This project predicts the species of an Iris flower from its measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Each flower is classified into one of 3 species:

* Setosa
* Versicolor
* Virginica

## Dataset

The project uses the classic [Iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) bundled with scikit-learn.

| Property | Value |
|----------|-------|
| Samples | 150 (50 per species) |
| Features | 4 (all in cm) |
| Classes | 3 (`0` = setosa, `1` = versicolor, `2` = virginica) |

## Workflow

1. **Load the data** with `load_iris()`.
2. **Split** into training (80%, 120 samples) and test (20%, 30 samples) sets with `train_test_split` (`random_state=1`).
3. **Preprocess** features with `StandardScaler`, fitted on the training set only to avoid data leakage.
4. **Train** a `LogisticRegression` classifier (`max_iter=200`).
5. **Evaluate** with accuracy, a classification report, and a confusion matrix.
6. **Predict** new flowers using the `predict_flower()` helper function.

## Results

The model reaches **96.7% accuracy** on the 30 held-out test samples.

| Species | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| Setosa | 1.00 | 1.00 | 1.00 | 11 |
| Versicolor | 1.00 | 0.92 | 0.96 | 13 |
| Virginica | 0.86 | 1.00 | 0.92 | 6 |

Only one sample was misclassified: a versicolor predicted as virginica.

## Usage

The notebook defines a helper that takes the four measurements (in cm) and returns the predicted species:

```python
predict_flower(5.1, 3.5, 1.4, 0.2)
# The specie of flower is setosa

predict_flower(6.0, 3.0, 4.8, 1.8)
# The specie of flower is virginica
```

## Getting Started

### Requirements

* Python 3.8+
* Dependencies are listed in [`requirements.txt`](requirements.txt) (scikit-learn, matplotlib, Jupyter)

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook Iris.ipynb
```

Or upload `Iris.ipynb` to [Google Colab](https://colab.research.google.com/) and run all cells.

## Project Structure

```
.
├── Iris.ipynb          # Data loading, training, evaluation, and prediction
├── requirements.txt    # Python dependencies
├── LICENSE
└── README.md
```

## Possible Improvements

* Train the model on the scaled features (`X_train_scaled`). The notebook scales the data but currently fits the model on the unscaled `X_train`.
* Compare other classifiers such as KNN, SVM, or Random Forest.
* Use cross-validation for a more reliable accuracy estimate, since the test set is only 30 samples.
* Add exploratory plots (pair plots, feature distributions).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
