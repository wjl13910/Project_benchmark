from models import model_function1, model_function2, model_function3, model_function4
from benchmark import overall_fitting, general_plot, predict, load_data

# optionally you could import other functions in the benchmark module such as:
# from benchmark import load_data, fitting, calculate_rmse, general_plot


# data readin
skylake_data = "./data/skylake_data.csv"
icelake_total_data = "./data/icelake_data.csv"
icelake_ordered_data = "./data/icelake_data2.csv"

icelake_general_data = load_data(icelake_ordered_data)
x_value = icelake_general_data["x"]

# make general plot
general_plot(skylake_data, "Skylake", "x_value")
general_plot(icelake_total_data, "Icelake", "nodes_cores")
general_plot(icelake_ordered_data, "Icelake", "x_value")


# make plot with different estimate models and compare their RMSE errors
parameters1, overall_rmse1 = overall_fitting(
    skylake_data, "Skylake_model_2_and_4", model_function2, model_function4, "lm")

parameters2, overall_rmse2 = overall_fitting(
    skylake_data, "Skylake_model_1_and_3", model_function1, model_function3, "lm")

# print the result for analysing
print("CPU                    MB                       IB")
print("RMSE from the model 2 and model 4")
print(overall_rmse1)
print("----------------------------------")
print("RMSE from the model 1 and model 3")
print(overall_rmse2)

# Accroding to the RMSE results. model 2 and model 4 are better than model 1 and 3.


# previously we build model based on Skylake data
# here we use our model to make prediction in Icelake data
# plot preditions along with Skylake and Icelake data

# perform parameters adjustment
mb_devided_mb_ref = 204.8/119.21
cpu_devided_cpu_ref = 6.08/1.331
ib_devided_ib_ref = 12.5/7

memory_adjust_parameter = parameters1[1] / mb_devided_mb_ref
cpu_adjust_parameter = parameters1[0] / cpu_devided_cpu_ref
ib_adjust_parameter = parameters1[2] / ib_devided_ib_ref

ib_adjust_parameter[2] = ib_adjust_parameter[2] * ib_devided_ib_ref

# plot data
pred_cpu_value = predict(icelake_ordered_data, skylake_data, 'CPU Time',
                         model_function2, cpu_adjust_parameter, "Icelake", "Skylake", "CPU_Time_Prediction")

pred_cpu_value = predict(icelake_ordered_data, skylake_data, 'Memory Time',
                         model_function2, memory_adjust_parameter, "Icelake", "Skylake", "Memory_Time_Prediction")

pred_cpu_value = predict(icelake_ordered_data, skylake_data, 'MPI Time',
                         model_function4, ib_adjust_parameter, "Icelake", "Skylake", "MPI_Time_Prediction")
