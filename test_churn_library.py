"""
Runs test for churn_library.py and logs info and error
"""
import os
import logging
import pytest
import churn_library
import joblib

os.environ["QT_QPA_PLATFORM"] = "offscreen"
logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    force=True,
)


@pytest.fixture(name="test_dataframe")
def test_dataframe_data():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        data_frame = churn_library.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    return data_frame


def test_data(test_dataframe):
    """
    Check if the dataframe imported has actual data in it, that is
    the shape of rows and columns should be greater than Zero
    """
    try:
        assert test_dataframe.shape[0] > 0
        assert test_dataframe.shape[1] > 0
        logging.info("the data has rows and columns")
    except AssertionError as err:
        logging.error("Testing import data_the dataframe data is not correct")
        raise err


def test_eda(test_dataframe):
    """
    test perform_eda function: Ensure the images from exploratory data analysis
    are stored in the directory
    """
    column_names_list = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
        "heat_map",
    ]
    churn_library.perform_eda(test_dataframe)
    for names in column_names_list:
        try:
            with open("images/eda/" + names + ".jpg", "r"):
                logging.info("The images corresponding to: " + names + " is saved")
        except FileNotFoundError as err:
            logging.error(" The EDA for " + names + " is not saved")
            raise err


def test_encoder_helper(test_dataframe):
    """
    test encoder_helper function: check if the new encoded columns are valid and not NULL
    """
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    data_frame = churn_library.encoder_helper(test_dataframe, cat_columns)
    encoded_columns_list = [
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    try:
        assert all(
            encoded_column in data_frame.columns
            for encoded_column in encoded_columns_list
        )
        logging.info("categorical columns are encoded properly")
    except AssertionError as err:
        logging.error("The categorical columns are not encoded properly")
        raise err
    try:
        assert data_frame[encoded_columns_list].notnull().all().all()
        logging.info("the categorical columns that are encoded are not null")
    except AssertionError as err:
        logging.error("the categorical columns enocoded are null")
        raise err


def test_perform_feature_engineering(test_dataframe):
    """
    Test: test the function perform_feature_engineering
    """
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    data_frame = churn_library.encoder_helper(test_dataframe, cat_columns)
    x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(
        data_frame
    )
    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("SUCCESS: Feature engineering completeed")
    except AssertionError as err:
        logging.error("Error: mismatch in feature length and label lengths")
        raise err


def test_train_models(test_dataframe):
    """
    test train_models
    """
    # check if the training is completed successfully
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    data_frame = churn_library.encoder_helper(test_dataframe, cat_columns)
    x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(
        data_frame
    )
    churn_library.train_models(x_train, x_test, y_train, y_test)

    # Check if the trained models are saved and present:
    try:
        joblib.load("models/logisticRegression_model.pkl")
        joblib.load("models/randomforest_model.pkl")
        logging.info("SUCCESS: the trained models can be loaded")
    except FileNotFoundError as err:
        logging.info("Error: Trained model cannot be loaded")
