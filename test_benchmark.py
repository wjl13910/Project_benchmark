from benchmark import load_data, fitting, calculate_rmse, general_plot, overall_workflow, general_plot
from models import model_function1, model_function2, model_function3, model_function4
import pytest
import numpy as np


x_test = [1, 2, 3, 4, 5]
target_value = [15, 8, 4, 3.5, 3]
parameter = np.array([15])

expected_rmse = 0.51234753829798
# --------------test functions------------------


def test_load_data():
    with pytest.raises(FileNotFoundError) as excinfo:
        load_data("./data.csv")
    excinfo.match("File does not exist.")


def test_calculate_rmse():
    test_rmse = calculate_rmse(
        x_test, target_value, model_function1, parameter)
    assert test_rmse == expected_rmse
