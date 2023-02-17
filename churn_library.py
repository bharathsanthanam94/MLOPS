# library doc string
'''
This module predicts customer who are likely to churn based on the input features.
'''

# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    data_frame["Churn"] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    column_names_list = ["Churn", "Customer_Age","Marital_Status","Total_Trans_Ct","heat_map"]
    for column in column_names_list:
        plt.figure(figsize=(20, 10))
        if column == "Churn":
            data_frame['Churn'].hist()
        elif column == "Customer_Age":
            data_frame["Customer_Age"].hist()
        elif column == "Marital_Status":
            data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
        elif column == "Total_Trans_Ct":
            sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
        elif column == "heat_map":
            sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig("images/eda/" + column + ".jpg")
        plt.close()


def encoder_helper(data_frame, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for cat_feature in category_lst:
        cat_feature_churn= cat_feature+"_Churn"
        cat_feature_list =[]
        cat_feature_groups =data_frame.groupby(cat_feature).mean()['Churn']
        for val in data_frame[cat_feature]:
            cat_feature_list.append(cat_feature_groups.loc[val])
        data_frame[cat_feature_churn]=cat_feature_list
    return data_frame

def perform_feature_engineering(data_frame):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that 
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_target=data_frame["Churn"]
    x_features=pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    x_features[keep_cols]= data_frame[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(x_features, 
                                                        y_target, test_size= 0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
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
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


def main():
    pth = "./data/bank_data.csv"
    df = import_data(pth)
    perform_eda(df)
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]
    df= encoder_helper(df,cat_columns)
    print(df["Education_Level_Churn"])
    x_train,x_test,y_train,y_test=perform_feature_engineering(df)



if __name__ == '__main__':
    main()