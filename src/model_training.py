from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def define_models_and_params(cfg):
    """
    Parse the config to build a dictionary mapping:
        "ModelName" -> (model_instance, param_grid)
    """
    models_and_params = {}

    if "SVM" in cfg["model_training"]:
        svm_params = cfg["model_training"]["SVM"]["param_grid"]
        models_and_params["SVM"] = (SVC(), svm_params)

    if "LogisticRegression" in cfg["model_training"]:
        lr_params = cfg["model_training"]["LogisticRegression"]["param_grid"]
        models_and_params["LogisticRegression"] = (
            LogisticRegression(max_iter=1000),
            lr_params
        )

    if "kNN" in cfg["model_training"]:
        knn_params = cfg["model_training"]["kNN"]["param_grid"]
        models_and_params["k-NN"] = (KNeighborsClassifier(), knn_params)

    if "RandomForest" in cfg["model_training"]:
        rf_params = cfg["model_training"]["RandomForest"]["param_grid"]
        # Handle "null" in max_depth config
        rf_params["max_depth"] = [None if v is None else v for v in rf_params["max_depth"]]
        models_and_params["RandomForest"] = (RandomForestClassifier(), rf_params)

    if "GradientBoosting" in cfg["model_training"]:
        gb_params = cfg["model_training"]["GradientBoosting"]["param_grid"]
        models_and_params["GradientBoosting"] = (GradientBoostingClassifier(), gb_params)

    return models_and_params


def cross_validate_and_select_model(X_train, y_train, model, param_grid, cfg):
    n_splits = cfg["cross_validation"]["n_splits"]
    random_state = cfg["cross_validation"]["random_state"]
    scoring = cfg["cross_validation"]["scoring"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"  Best CV {scoring.capitalize()}: {grid_search.best_score_:.4f}")
    print(f"  Best Params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_on_test_set(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("  Classification Report:")
    print(classification_report(y_test, y_pred))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------------------------\n")


def train_and_evaluate(X_train, y_train, X_test, y_test, cfg):
    models_and_params = define_models_and_params(cfg)
    best_models = {}

    for model_name, (model, param_grid) in models_and_params.items():
        print(f"Training {model_name}")
        best_model = cross_validate_and_select_model(X_train, y_train, model, param_grid, cfg)
        best_models[model_name] = best_model

    for model_name, model in best_models.items():
        print(f"Evaluating {model_name}")
        evaluate_on_test_set(model, X_test, y_test)

    return best_models
