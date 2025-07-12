# *** Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
    )
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump
from feat_engineering import (
    X,
    X_train,
    X_test,
    y_spread,
    y_train_spread,
    y_test_spread,
    y_ou,
    y_train_ou,
    y_test_ou,
    y_w
    )


def go(X_train, X_test, y_train_spread, y_test_spread, y_train_ou, y_test_ou,
        y_train_w, y_test_w,  merged_df):
    # Train/test split (spread for point spread,
    # ou for Over/Under, w for underdog wins)
    X_train, X_test, y_train_spread, y_test_spread, y_train_ou, y_test_ou,
    y_train_w, y_test_w = train_test_split(
        X, y_spread, y_ou, y_w, test_size=0.2, random_state=42
    )

    def evaluate_pca_model(
            X_train,
            X_test,
            y_train,
            y_test,
            n_components_list):
        scores = []
        for n in n_components_list:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n)),
                ('model', LogisticRegression(max_iter=500, random_state=42))
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores.append((n, acc))
            print(f"Components: {n:2d} | Accuracy: {acc:.4f}")
        return scores

    # Test PCA components for spread dataset
    components_to_test = list(
        range(2, min(X_train.shape[1], X_train.shape[0]) + 1, 2))
    results_spread = evaluate_pca_model(
        X_train, X_test, y_train_spread, y_test_spread, components_to_test)
    print(results_spread)

    # Test PCA components for Over/Under dataset
    components_to_test = list(
        range(2, min(X_train.shape[1], X_train.shape[0]) + 1, 2))
    results_ou = evaluate_pca_model(
        X_train, X_test, y_train_ou, y_test_ou, components_to_test)
    print(results_ou)

    # Test PCA components for underdog winning dataset
    components_to_test = list(
        range(2, min(X_train.shape[1], X_train.shape[0]) + 1, 2))
    results_w = evaluate_pca_model(
        X_train, X_test, y_train_w, y_test_w, components_to_test)
    print(results_w)

    models = {
        "Logistic Regression (Spread)": LogisticRegression(
            max_iter=500,
            random_state=42
            ),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for name, clf in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clf)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train_spread,
            cv=5,
            scoring='f1_weighted'
            )  # or 'roc_auc'
        print(f"{name} - F1 Score: {scores.mean():.4f}")

    models = {
        "Logistic Regression (Over/Under)": LogisticRegression(
            max_iter=500,
            random_state=42
            ),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for name, clf in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clf)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train_ou,
            cv=5,
            scoring='f1_weighted'
            )  # or 'roc_auc'
        print(f"{name} - F1 Score: {scores.mean():.4f}")

    models = {
        "Logistic Regression (Favorite Wins)": LogisticRegression(
            max_iter=500,
            random_state=42
            ),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for name, clf in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clf)
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train_w, cv=5, scoring='f1')
        print(f"{name} - F1 Score: {scores.mean():.4f}")

    params = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    grid_spread = GridSearchCV(
        RandomForestClassifier(
            random_state=42),
        params,
        cv=5,
        scoring='accuracy'
        )
    grid_spread.fit(X_train, y_train_spread)

    print("Best Accuracy:", grid_spread.best_score_)

    params = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    grid_ou = GridSearchCV(
        RandomForestClassifier(
            random_state=42),
        params,
        cv=5,
        scoring='accuracy'
    )
    grid_ou.fit(X_train, y_train_ou)

    print("Best Accuracy:", grid_ou.best_score_)

    params = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    grid_w = GridSearchCV(
        RandomForestClassifier(
            random_state=42),
        params,
        cv=5,
        scoring='accuracy'
        )
    grid_w.fit(X_train, y_train_w)

    print("Best Accuracy:", grid_w.best_score_)

    pipeline_spread = Pipeline([
        ('scaler_spread', StandardScaler()),
        ('model_spread', RandomForestClassifier())])

    params = {
        'model_spread__n_estimators': [100, 300],
        'model_spread__max_depth': [None, 10, 20],
        'model_spread__min_samples_split': [2, 5],
        'model_spread__min_samples_leaf': [1, 2],
        'model_spread__max_features': ['sqrt', 'log2']
    }

    grid_spread = GridSearchCV(
        estimator=pipeline_spread,
        param_grid=params,
        cv=5,
        scoring='f1_weighted')

    grid_spread.fit(X_train, y_train_spread)

    print("Best F1 Weighted:", grid_spread.best_score_)
    print("Best Parameters:", grid_spread.best_params_)

    scores_spread = cross_val_score(
        pipeline_spread, X, y_spread, cv=5, scoring='f1')
    print(
        f"Mean F1 (Spread): {scores_spread.mean():.4f} ± "
        f"{scores_spread.std():.4f}"
    )

    pipeline_ou = Pipeline([
        ('scaler_ou', StandardScaler()),
        ('PCA', PCA(n_components=18)),
        ('model_ou', LogisticRegression(max_iter=5000))
    ])

    param_grid = {
        # Regularization strength (inverse)
        'model_ou__C': [0.01, 0.1, 1.0, 10],
        # Standard choice; 'l1' requires solver='liblinear'
        'model_ou__penalty': ['l2'],
        # Optimizers for larger datasets
        'model_ou__solver': ['liblinear', 'lbfgs', 'saga']
    }

    grid_ou = GridSearchCV(
        estimator=pipeline_ou,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted'
    )
    grid_ou.fit(X_train, y_train_ou)

    scores_ou = cross_val_score(
        pipeline_ou, X, y_ou, cv=5, scoring='f1_weighted'
    )

    print(
        "Best F1 Weighted:", grid_ou.best_score_
    )
    print("Best Parameters:", grid_ou.best_params_)
    print(
        f"Mean F1 (Over/Under): {scores_ou.mean():.4f} ± "
        f"{scores_ou.std():.4f}"
    )

    pipeline_w = Pipeline([
        ('scaler_w', StandardScaler()),
        ('model_w', LogisticRegression(class_weight='balanced', max_iter=5000))
    ])

    param_grid = {
        # Regularization strength (inverse)
        'model_w__C': [0.01, 0.1, 1.0, 10],
        # Standard choice; 'l1' requires solver='liblinear'
        'model_w__penalty': ['l2'],
        # Optimizers for larger datasets
        'model_w__solver': ['liblinear', 'lbfgs', 'saga']
    }

    grid_w = GridSearchCV(
        estimator=pipeline_w,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted'
    )
    grid_w.fit(X_train, y_train_ou)

    scores_w = cross_val_score(
        pipeline_w, X, y_ou, cv=5, scoring='f1_weighted'
    )
    print(
        f"Mean F1 (Over/Under): {scores_w.mean():.4f} ± {scores_w.std():.4f}")

    print("Best F1 Weighted:", grid_w.best_score_)
    print("Best Parameters:", grid_w.best_params_)

    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    forest_spread = RandomForestClassifier(random_state=0)
    forest_spread.fit(X_train, y_train_spread)

    start_time = time.time()
    importances_spread = forest_spread.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in forest_spread.estimators_],
        axis=0)
    elapsed_time = time.time() - start_time

    print(
        f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_spread_importances = pd.Series(
        importances_spread,
        index=feature_names
        )

    fig, ax = plt.subplots()
    forest_spread_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    forest_ou = RandomForestClassifier(random_state=0)
    forest_ou.fit(X_train, y_train_spread)

    start_time = time.time()
    importances_ou = forest_ou.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest_ou.estimators_],
                 axis=0)
    elapsed_time = time.time() - start_time

    print(
        f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_ou_importances = pd.Series(importances_ou, index=feature_names)

    fig, ax = plt.subplots()
    forest_ou_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    forest_w = RandomForestClassifier(random_state=0)
    forest_w.fit(X_train, y_train_spread)

    start_time = time.time()
    importances_w = forest_w.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest_w.estimators_],
                 axis=0)
    elapsed_time = time.time() - start_time

    print(
        f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_w_importances = pd.Series(importances_w, index=feature_names)

    fig, ax = plt.subplots()
    forest_w_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    pipeline_spread = Pipeline([
        ('scaler_spread', StandardScaler()),
        ('model_spread', RandomForestClassifier(
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features='sqrt',
            n_estimators=300
        ))])

    pipeline_spread.fit(X_train, y_train_spread)

    dump(X_train.columns.tolist(), 'models/spread_feature_names.pkl')

    pipeline_ou = Pipeline([
        ('scaler_ou', StandardScaler()),
        ('model_ou', LogisticRegression(
            max_iter=5000,
            C=0.01,
            penalty='l2',
            solver='liblinear'
        ))
    ])

    pipeline_ou.fit(X_train, y_train_ou)

    dump(X_train.columns.tolist(), 'models/ou_feature_names.pkl')

    pipeline_w = Pipeline([
        ('scaler_w', StandardScaler()),
        ('model_w', LogisticRegression(
            max_iter=5000,
            C=0.1,
            penalty='l2',
            solver='saga'
        ))
    ])

    pipeline_w.fit(X_train, y_train_w)

    dump(X_train.columns.tolist(), 'models/w_feature_names.pkl')

    ConfusionMatrixDisplay.from_estimator(
        pipeline_ou, X_test, y_test_ou)
    ConfusionMatrixDisplay.from_estimator(
        pipeline_spread, X_test, y_test_spread)
    ConfusionMatrixDisplay.from_estimator(
        pipeline_w, X_test, y_test_w)

    dump(pipeline_spread, 'models/pipeline_spread.pkl')
    dump(pipeline_ou, 'models/pipeline_ou.pkl')
    dump(pipeline_w, 'models/pipeline_w.pkl')

    return pipeline_spread, pipeline_ou, pipeline_w, merged_df


if __name__ == "__main__":
    go()
