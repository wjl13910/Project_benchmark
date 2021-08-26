from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


def load_data(filepath):
    """Load data filepath can be a String.

    The follwing algoright is taking a file path as an argument and ruturn a list of lists. In the case of not exist filepath the function will raise an error of type FileNotFoundError and with an appropriate message.
    We are casting the str to Path. Because when the user run it through command line the argument is comming as str.

    Parameters
    ----------
    filepath : str
        A given filepath

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
        raise FileNotFoundError("File does not exist.")

    data = pd.read_csv(filepath, delimiter=",")

    return data


def fitting(x_value, target_value, model_function, fitting_method, type_of_fit):
    """
    This is the function for fitting the timing data.


    Parameters
    ----------
    x_value: list
        A given list of x_value data
    target_value: list
        A given list contans target fitting time data
    model_function: function
        Model for fitting the target time data
    type_of_fit : str
        The type of the data used to do fitting. options:{'CPU', 'MB', 'MPI'}
    fitting_method: str
        The method used to fit the data. The value can only been chosen from options:{'dogbox', 'lm', 'trf'}.


    Returns
    -------
    parameters : numpy.array
        List of fitting parameters.


    Raise
    -----
    ValueError
        Raise an ValueError if the given x_value does not match the length of target_value
    """
    if len(target_value) != len(x_value):
        raise ValueError("Value length not match")

    fig = plt.figure()

    parameters, _ = curve_fit(
        model_function, x_value, target_value, method=fitting_method)

    xspace = np.linspace(x_value[0], x_value[len(x_value) - 1], 100)
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

    # making plot
    plt.plot(x_value, target_value, 'o', label='cpu')
    plt.plot(xspace, fit_curve, '-', label='fit')
    plt.xlabel("cores_per_node")
    plt.ylabel("time/s")
    plt.title(type_of_fit + "_Fitting Plot")
    plt.legend()
    plt.savefig(type_of_fit + "_fit.jpg")
    plt.close()

    return parameters


def calculate_rmse(x_value, target_value, model_function, fit_parameters):
    """
    This function works to produce the Root Mean Squard Error of the fit against the original data.


    Parameters
    ----------
    x_value: list
        A given list of x_value data
    target_value: list
        A given list contans target fitting time data
    model_function: function
        Model for fitting the target time data
    fit_parameters: numpy.array
        List of fitting parameters.


    Returns
    -------
    RMSE : double
        Root Mean Squard Error of the fitting result.


    Raise
    -----
    ValueError
        Raise an ValueError if the given x_value does not match the length of target_value
    ValueError
        Raise an ValueError if the given model_function is unsupported
    """
    if len(target_value) != len(x_value):
        raise ValueError("Value length not match")

    n = len(fit_parameters)
    if n == 1:
        fit_A = fit_parameters[0]
        fit_value = model_function(x_value, fit_A)
    elif n == 2:
        fit_A = fit_parameters[0]
        fit_B = fit_parameters[1]
        fit_value = model_function(x_value, fit_A, fit_B)
    elif n == 3:
        fit_A = fit_parameters[0]
        fit_B = fit_parameters[1]
        fit_C = fit_parameters[2]
        fit_value = model_function(x_value, fit_A, fit_B, fit_C)
    else:
        raise ValueError("Function do not support that model")

    RMSE = np.sqrt(sum((target_value - fit_value)**2)/len(target_value))
    return RMSE


def general_plot(filepath, system_information, x_value_type, figsize=(7, 5)):
    """
    This function is works for ploting the general time graph.


    Parameters
    ----------
    filepath : str
        A given filepath    
    system_information: str
        A given system name for define the plot
    x_value_type: str
        A given type of x value for different x label. Which can only be chosen from:{'nodes_cores', 'x_value'}
    figsize: tuple,optional
        Figsize of the plot


    Raise
    -----
    ValueError
        Raise an ValueError if the given x_value_type does not match the given options


    """
    general_data = load_data(filepath)
    fig = plt.figure(figsize=figsize)

    # NOTE: To make this function work, the x_label and x value data can only be in the same order as we do in here.
    # prepare data

    keys = general_data.keys()

    if x_value_type == 'nodes_cores':
        x_label = general_data[keys[0]]
    elif x_value_type == 'x_value':
        x_label = general_data[keys[5]]
    else:
        raise ValueError("Unsupport x_value_type input")

    T_first = general_data[keys[1]]
    T_second = general_data[keys[2]]
    T_third = general_data[keys[3]]
    T_fourth = general_data[keys[4]]

    # making plot
    plt.plot(x_label, T_first, '.-', label=keys[1])
    plt.plot(x_label, T_second, '.-', label=keys[2])
    plt.plot(x_label, T_third, '.-', label=keys[3])
    plt.plot(x_label, T_fourth, '.-', label=keys[4])
    plt.legend()

    title = system_information + " General Time " + "(" + x_value_type + ")"
    plt.title(title)
    plt.xlabel(x_value_type)
    plt.ylabel("time/s")
    fig.autofmt_xdate()

    plt.savefig(system_information + "_total.jpg")
    plt.close()


def overall_fitting(filepath, system_information, model_function, model_function_for_ib, fitting_method):
    """
    This workflow function helps to do the overall fit, which is the combination of the above data fitting functionality. To use this function, the data format should strictly follows the pattern we used, namely the headings, "x label,MPI Time,Memory time,CPU time,Total Time,x".

    Parameters
    ----------
    filepath : str
        A given filepath    
    system_information: str
        A given system name for define the plot
    model_function: function
        Model for fitting the target time data   
    model_function_for_ib: function
        Model for fitting the target time data perticular for InifiniBnad
    fit_parameters: numpy.array
        List of fitting parameters.

    Returns
    -------
    overall_rmse : list
        A list contants all the Root Mean Squard Error of the fitting results.
    parameters: list of lists
        A list contants all the parameter used for fitting in order.
    """
    # load data
    data = load_data(filepath)

    x = data["x"]

    parameters = []
    overall_rmse = []
    # cpu fit
    cpu_parameters = fitting(
        x, data["CPU Time"], model_function, fitting_method, system_information + "_CPU")
    cpu_rmse = calculate_rmse(
        x, data["CPU Time"], model_function, cpu_parameters)
    parameters.append(cpu_parameters)
    overall_rmse.append(cpu_rmse)

    # mb fit
    mb_parameters = fitting(
        x, data["Memory Time"], model_function, fitting_method, system_information + "_memory")
    mb_rmse = calculate_rmse(
        x, data["Memory Time"], model_function, mb_parameters)
    parameters.append(mb_parameters)
    overall_rmse.append(mb_rmse)

    # ib fit
    ib_parameters = fitting(
        x, data["MPI Time"], model_function_for_ib, fitting_method, system_information + "_IB")
    ib_rmse = calculate_rmse(
        x, data["MPI Time"], model_function_for_ib, ib_parameters)
    parameters.append(ib_parameters)
    overall_rmse.append(ib_rmse)

    return parameters, overall_rmse


def predict(test_data_path, sample_data_filepath, target_value_type, model_function, parameters, test_system_information, sample_system_information, title):
    """
    This function works for make prediction. And plot the predict data against the Skylake and Icelake experiment data.




    Parameters
    ----------
    test_data_path : str
        A given filepath    
    sample_data_filepath : str
        A given filepath    
    target_value_type: str
        A given target value name for plot labels    
    test_system_information: str
        A given system name for define the plot
    sample_system_information: str
        A given system name for define the plot
    model_function: function
        Model for fitting the target time data
    parameters: numpy.array
        List of fitting parameters.  
    title: str
        A given title name of the plot


    Returns
    -------
    predict_value: numpy.array
        Array contans all the predict value

    Raise
    -----
    ValueError
        Raise an ValueError if the given x_value does not match the length of target_value
    ValueError
        Raise an ValueError if the given model_function is unsupported

    """

    test_data = load_data(test_data_path)
    sample_data = load_data(sample_data_filepath)

    x_test = test_data["x"]
    x_sample = sample_data["x"]

    n = len(parameters)
    if n == 1:
        parameter_a = parameters[0]
        predict_value = model_function(x_test, parameter_a)
    elif n == 2:
        parameter_a = parameters[0]
        parameter_b = parameters[1]
        predict_value = model_function(x_test, parameter_a, parameter_b)
    elif n == 3:
        parameter_a = parameters[0]
        parameter_b = parameters[1]
        parameter_c = parameters[2]
        predict_value = model_function(
            x_test, parameter_a, parameter_b, parameter_c)
    else:
        raise ValueError("Function do not support that model")

    plt.plot(x_test, predict_value, '.--', label="predict data")
    plt.plot(x_test, test_data[target_value_type], '.-',
             label=test_system_information + ' experiment data')
    plt.plot(x_sample, sample_data[target_value_type], '.-',
             label=sample_system_information + '  experiment data')
    plt.title(title)
    plt.xlabel('x_value')
    plt.ylabel("time/s")
    plt.legend()
    plt.savefig(title + ".jpg")
    plt.close()

    return predict_value
