'''
Runs test for churn_library.py and logs info and error
'''
import os
import logging
import pytest
import churn_library
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)


@pytest.fixture(name='test_import')
def test_import_data():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = churn_library.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    return data_frame


def test_data(test_import):
    '''
    Check if the dataframe imported has actual data in it, that is 
    the shape of rows and columns should be greater than Zero
    '''
    try:
        assert test_import.shape[0] > 0
        assert test_import.shape[1] > 0
        logging.info("the data has rows and columns")
    except AssertionError as err:
        logging.error("Testing import data_the dataframe data is not correct")
        raise err

def test_eda(test_import):
    '''
    test perform_eda function: Ensure the images from exploratory data analysis
    are stored in the directory
    '''
    column_names_list = ["Churn", "Customer_Age",
                         "Marital_Status", "Total_Trans_Ct", "heat_map"]
    churn_library.perform_eda(test_import)
    for names in column_names_list:
        try:
            with open("images/eda/"+names+".jpg",'r'):
                logging.info("The images corresponding to: "+names+" is saved" )
        except FileNotFoundError as err:
            logging.error(" The EDA for "+names+" is not saved")
	
	

# def test_encoder_helper(encoder_helper):
#       '''
#       test encoder helper
#       '''


# def test_perform_feature_engineering(perform_feature_engineering):
#       '''
#       test perform_feature_engineering
#       '''


# def test_train_models(train_models):
#       '''
#       test train_models
#       '''