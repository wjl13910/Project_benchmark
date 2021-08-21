from models import model_function1, model_function2, model_function3, model_function4
from benchmark import overall_workflow, general_plot

# optionally you could import other functions in the benchmark module such as:
# from benchmark import load_data, fitting, calculate_rmse, general_plot


# data readin
skylake_data = "./data/skylake_data.csv"
icelake_total_data = "./data/icelake_data.csv"
icelake_ordered_data = "./data/icelake_data2.csv"


# make general plot
general_plot(skylake_data, "Skylake", "x_value")
general_plot(icelake_total_data, "Icelake", "nodes_cores")
general_plot(icelake_ordered_data, "Icelake", "x_value")


# make plot with different estimate models and compare their RMSE errors
parameters1, overall_rmse1 = overall_workflow(
    skylake_data, "Skylake_model_2_and_4", model_function2, model_function4, "lm")

parameters2, overall_rmse2 = overall_workflow(
    skylake_data, "Skylake_model_1_and_3", model_function1, model_function3, "lm")

# print the result for analysing
print("CPU                    MB                       IB")
print("RMSE from the model 2 and model 4")
print(overall_rmse1)
print("----------------------------------")
print("RMSE from the model 1 and model 3")
print(overall_rmse2)

# Accroding to the RMSE results. model 2 and model 4 are better than model 1 and 3.
