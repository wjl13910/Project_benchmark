from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.base import DataError
from scipy.optimize import curve_fit
import pandas as pd


def load_data(filepath):
    """Load data filepath can be a String.

    The follwing algoright is taking a file path as an argument and ruturn a list of lists. In the case of not exist
    filepath the function will raise an error of type FileNotFoundError and with an appropriate message.
    We are casting the str to Path. Because when the user run it through command line the argument is comming as str.

    Parameters
    ----------
    filepath : str
        A given filepath, such C:Users/Documents/Data/samples.csv or C:Users/Documents/Data/weights.csv

    Returns
    -------
    data : pandas.DataFrame
        The data of .csv as a DataFrame  

    Raise
    -----
    FileNotFoundError
        Raise an FileNotFoundError if the given filepath does not exist, with a supplementary message.

    """

    file_path_casted = Path(filepath)
    if not file_path_casted.exists():
        raise FileNotFoundError(
            "The file does not exist, check the given argument.")

    data = pd.read_csv(filepath, delimiter=",")

    return data


def fitting(x, target_value, model_function, fitting_method, type_of_fit):
    """
    This is the function for fitting the timing data.


    Parameters
    ----------
    type_of_fit : str
        A type of the fitting options:{'CPU', 'MB', 'IB'}
    model_function: function
        Options:{model_function, model_function_3variables}
    fitting_method: str
        The method used to fit the data. The value can only been chosen from options:{'dogbox', 'lm', 'trf'}.


    Returns
    -------
    parameters : numpy.ndarray
        The data of fitting results. 
    """
    if len(target_value) != len(x):
        raise ValueError("Value length not match")
    fig = plt.figure()
    # three possible methods are: dogbox, lm, trf
    parameters, covariance = curve_fit(
        model_function, x, target_value, method=fitting_method)

    xspace = np.linspace(x[0], x[len(x) - 1], 100)
    if len(parameters) == 1:
        fit_A = parameters[0]
        fit_curve = model_function(xspace, fit_A)
    elif len(parameters) == 2:
        fit_A = parameters[0]
        fit_B = parameters[1]
        fit_curve = model_function(xspace, fit_A, fit_B)
    elif len(parameters) == 3:
        fit_A = parameters[0]
        fit_B = parameters[1]
        fit_C = parameters[2]
        fit_curve = model_function(xspace, fit_A, fit_B, fit_C)
    else:
        raise ValueError("Function do not support that Model")

    plt.plot(x, target_value, 'o', label='cpu')
    plt.plot(xspace, fit_curve, '-', label='fit')
    plt.xlabel("x")
    plt.ylabel("time/s")
    plt.title(type_of_fit + "_Fitting Plot")
    plt.legend()
    plt.savefig(type_of_fit + "_fit.jpg")
    plt.close()
    #perr = np.sqrt(np.diag(covariance))
    return parameters


def calculate_rmse(x, target_value, model_function, fit_parameters):
    """
    This function works to produce the Root Mean Squard Error of the fit against the original data.
    """
    if len(target_value) != len(x):
        raise ValueError("Value length not match")

    n = len(fit_parameters)
    if n == 1:
        fit_A = fit_parameters[0]
        fit_value = model_function(x, fit_A)
    elif n == 2:
        fit_A = fit_parameters[0]
        fit_B = fit_parameters[1]
        fit_value = model_function(x, fit_A, fit_B)
    elif n == 3:
        fit_A = fit_parameters[0]
        fit_B = fit_parameters[1]
        fit_C = fit_parameters[2]
        fit_value = model_function(x, fit_A, fit_B, fit_C)
    else:
        raise ValueError("Function do not support that model")

    RMSE = np.sqrt(sum((target_value - fit_value)**2)/len(target_value))
    return RMSE


def general_plot(general_data, system_information, figsize=(7, 5)):
    """
    This function is works for ploting the general time graph.

    """

    fig = plt.figure(figsize=figsize)
    # prepare data for plotting
    # NOTE: To make this function work, the x_label data can only be in the first order as we do in here.

    keys = general_data.keys()
    x_label = general_data[keys[0]]
    T_first = general_data[keys[1]]
    T_second = general_data[keys[2]]
    T_third = general_data[keys[3]]
    T_fourth = general_data[keys[4]]

    plt.plot(x_label, T_first, '.-', label=keys[1])
    plt.plot(x_label, T_second, '.-', label=keys[2])
    plt.plot(x_label, T_third, '.-', label=keys[3])
    plt.plot(x_label, T_fourth, '.-', label=keys[4])
    plt.legend()

    title = system_information + " General Time"
    plt.title(title)
    plt.xlabel("nodes_cores")
    plt.ylabel("time/s")
    fig.autofmt_xdate()

    plt.savefig(system_information + "_total.jpg")
    plt.close()


def overall_workflow(filepath, system_information, model_function, model_function_for_ib, fitting_method):
    """
    This workflow function helps to do the overall fit and plot in one time, which is the combination of all above functionality. To use this function, your data format should strictly follows the pattern we used, namely the headings, "x label,MPI Time,Memory time,CPU time,Total Time,x".
    """
    data = load_data(filepath)
    x = data["x"]
    general_plot(data, system_information)

    # cpu fit
    cpu_parameters = fitting(
        x, data["CPU Time"], model_function, fitting_method, system_information + "_CPU")
    cpu_rmse = calculate_rmse(
        x, data["CPU Time"], model_function, cpu_parameters)

    # mb_fit
    mb_parameters = fitting(
        x, data["Memory Time"], model_function, fitting_method, system_information + "_memory")
    mb_rmse = calculate_rmse(
        x, data["Memory Time"], model_function, mb_parameters)

    # IB
    ib_parameters = fitting(
        x, data["MPI Time"], model_function_for_ib, fitting_method, system_information + "_IB")
    ib_rmse = calculate_rmse(
        x, data["MPI Time"], model_function_for_ib, ib_parameters)
    overall_rmse = [cpu_rmse, mb_rmse, ib_rmse]
    return overall_rmse
