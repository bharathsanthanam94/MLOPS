# library doc string
"""
This module predicts customer who are likely to churn based on the input features.
"""

# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import joblib

sns.set()
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    data_frame = pd.read_csv(pth)
    data_frame["Churn"] = data_frame["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return data_frame


def perform_eda(data_frame):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    column_names_list = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
        "heat_map",
    ]
    for column in column_names_list:
        plt.figure(figsize=(20, 10))
        if column == "Churn":
            data_frame["Churn"].hist()
        elif column == "Customer_Age":
            data_frame["Customer_Age"].hist()
        elif column == "Marital_Status":
            data_frame.Marital_Status.value_counts("normalize").plot(kind="bar")
        elif column == "Total_Trans_Ct":
            sns.histplot(data_frame["Total_Trans_Ct"], stat="density", kde=True)
        elif column == "heat_map":
            sns.heatmap(data_frame.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig("images/eda/" + column + ".jpg")
        plt.close()


def encoder_helper(data_frame, category_lst):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for cat_feature in category_lst:
        cat_feature_churn = cat_feature + "_Churn"
        cat_feature_list = []
        cat_feature_groups = data_frame.groupby(cat_feature).mean()["Churn"]
        for val in data_frame[cat_feature]:
            cat_feature_list.append(cat_feature_groups.loc[val])
        data_frame[cat_feature_churn] = cat_feature_list
    return data_frame


def perform_feature_engineering(data_frame):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    y_target = data_frame["Churn"]
    x_features = pd.DataFrame()
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    x_features[keep_cols] = data_frame[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.3, random_state=42
    )
    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    data = dict()
    data["train_rf"] = [
        "train_results_random_forest_classifier",
        y_train,
        y_train_preds_rf,
    ]
    data["test_rf"] = ["test_results_random_forest_classifier", y_test, y_test_preds_rf]
    data["train_lr"] = ["train_results_logistic_regression", y_train, y_train_preds_lr]
    data["test_lr"] = ["test_results_logistic_regression", y_test, y_test_preds_lr]
    for result_title, results_data in data.items():
        # Set the default size of figures to [5,5]
        # plt.rc("figure", figsize=(5, 5))
        # # print the title
        # plt.text(
        #     0.01,
        #     1.25,
        #     str(results_data[0]),
        #     {"fontsize": 10},
        #     fontproperties="monospace",
        # )
        # # print the classificaiton results
        # plt.text(
        #     0.01,
        #     0.05,
        #     str(classification_report(results_data[1], results_data[2])),
        #     {"fontsize": 10},
        #     fontproperties="monospace",
        # )
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(
            0.05,
            0.95,
            str(results_data[0]),
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.text(
            0.05,
            0.9,
            str(classification_report(results_data[1], results_data[2])),
            fontsize=12,
            transform=ax.transAxes,
            va="top",
        )
        ax.axis("off")
        plt.tight_layout()
        # plt.savefig("images/results/%s.jpg" % title)
        fig.savefig("images/results/" + result_title + ".jpg")
        plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + "Feature_Importance.jpg")
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
    feature_importance_plot(cv_rfc, x_test, "images/results")

    joblib.dump(cv_rfc.best_estimator_, "models/randomforest_model.pkl")
    joblib.dump(lrc, "models/logisticRegression_model.pkl")


def main():
    """
    Loads data, performs feature engineering, train using Random forest and
    Logistic regressions and prints results
    """
    pth = "./data/bank_data.csv"
    data_frame = import_data(pth)
    perform_eda(data_frame)
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    data_frame = encoder_helper(data_frame, cat_columns)
    x_train, x_test, y_train, y_test = perform_feature_engineering(data_frame)
    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
