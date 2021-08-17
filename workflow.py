from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.base import DataError
from scipy.optimize import curve_fit
import pandas as pd


from models import model_function1, model_function2, model_function3, model_function4
from benchmark import load_data, fitting, calculate_rmse, general_plot, overall_workflow


# syklake_data = load_data("./skylake_data.csv")

# x = syklake_data["x"]


# # general plot
# general_plot(syklake_data, "Skylake")

# CPU
# three possible methods are: dogbox, lm, trf
# cpu_parameters1 = fitting(
#     x, syklake_data["CPU Time"], model_function1, "dogbox", "cpu_model1")
# cpu_parameters2 = fitting(
#     x, syklake_data["CPU Time"], model_function2, "dogbox", "cpu_model2")

# cpu_rmse1 = calculate_rmse(
#     x, syklake_data["CPU Time"], model_function1, cpu_parameters1)
# cpu_rmse2 = calculate_rmse(
#     x, syklake_data["CPU Time"], model_function2, cpu_parameters2)

# # MB
# mb_parameters1 = fitting(
#     x, syklake_data["Memory Time"], model_function1, "lm", "memory_model1")
# mb_parameters2 = fitting(
#     x, syklake_data["Memory Time"], model_function2, "lm", "memory_model2")

# mb_rmse1 = calculate_rmse(
#     x, syklake_data["Memory Time"], model_function1, mb_parameters1)
# mb_rmse2 = calculate_rmse(
#     x, syklake_data["Memory Time"], model_function2, mb_parameters2)

# # IB
# ib_parameters1 = fitting(
#     x, syklake_data["MPI Time"], model_function3, "lm", "IB_model3")
# ib_parameters2 = fitting(
#     x, syklake_data["MPI Time"], model_function4, "lm", "IB_model4")

# ib_rmse1 = calculate_rmse(
#     x, syklake_data["MPI Time"], model_function3, ib_parameters1)
# ib_rmse2 = calculate_rmse(
#     x, syklake_data["MPI Time"], model_function4, ib_parameters2)


filepath = "./skylake_data.csv"

overall_workflow(filepath, "Skylake", model_function2,
                 model_function4, "lm")
