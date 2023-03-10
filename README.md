# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
- Predicts Churning customers using Random forest and Logistic regression models.
- Follows clean code principles adhering PEP8 standards and used Pylint tool.
- Test cases are implemented using pytest

## Files and data description
- Dataset is contained in the `data` directory in csv format
- The directory `images` contains the results of exploratory data analysis and results.
- The directory `logs` contains status of test cases and the errors.
- Project files
    - `churn_library.py`
    - `test_churn_library.py`
    - `churn_notebook.ipynb`

## Running Files
- Requires python 3.8 and install dependencies using 
```
pip install -r requirements_py3.8.txt
```
- To run the project(training), execute the python script  `churn_library.py` from the project directory as
 ```
 python3 churn_library.py
 ```
- This will run the model and save the results in the `images` directory
- The tests are written using `pytest` package
- To run the test scripts, execute the script `test_churn_library.py `from the project directory as 
```
pytest test_churn_library.py
``` 
This will run tests, and save log files in the `logs` diretory



