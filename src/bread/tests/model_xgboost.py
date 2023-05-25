# define a module to perform cross-validation for XGBoost
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import generate_all_permutations, flatten_3d_array, grid_train_xgboost


def run_xgboost(X_train, y_train, X_test, y_test, params={}, augment=True):
    if augment:
        X_train, y_train = generate_all_permutations(X_train, y_train)
    X_train = flatten_3d_array(X_train)
    X_test = flatten_3d_array(X_test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    if params == {}:  # if no parameters are provided, use the best parameters found from grid search
        params = grid_train_xgboost(
            X_train, y_train, X_test, y_test, early_stopping_rounds=5, num_boost_round=200)
        print('best parameters found from grid search:', params)

    # Initialize XGBoost model and fit to training data
    model = xgb.train(params, dtrain, evals=[(dtest, 'test')],
                      num_boost_round=200, verbose_eval=False, early_stopping_rounds=5)
    preds = model.predict(dtest)
    preds = np.round(preds)
    print("Accuracy of the best model:", "%.4f " %
          accuracy_score(y_test, preds))
    return model, preds


def cv_xgboost(X, y, augment=True):
    # Initialize cross-validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train and evaluate model for each fold
    fold_scores = []
    models = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model, preds = run_xgboost(
            X_train, y_train, X_test, y_test, augment=augment)

        # Compute predictions on test data and evaluate accuracy
        fold_score = accuracy_score(y_test, preds)
        fold_scores.append(fold_score)
        models.append(model)
    # Compute mean and standard deviation of fold scores
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    # Print validation results
    print('Mean Score: {:.3f} +/- {:.3f}'.format(mean_score, std_score))
    return mean_score, std_score, models