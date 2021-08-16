from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
    data_array : numpy.ndarray
        The data of .csv as a List  e.g numpy.ndarray([1,2,3])

    Raise
    -----
    FileNotFoundError
        Raise an FileNotFoundError if the given filepath does not exist, with a supplementary message.

    """

    file_path_casted = Path(filepath)
    if not file_path_casted.exists():
        raise FileNotFoundError(
            "The file does not exist, check the given argument.")

    data_array = np.loadtxt(filepath, delimiter=",")

    return data_array


def model_function_3variables(x, a, b, c):
    """
    The estimated model function for the CPU extrapolation method. This is the 3 variables version for InifiniBand communication time.

    """
    y = a/(x+b) + c
    return y


def model_function(x, a, b):
    """
    The estimated model function for the CPU extrapolation method. This is the version for CPU computing and Memory Bandwidth.

    """
    y = a/(x+b)
    return y


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
    parameters, covariance = curve_fit(
        model_function, x, target_value, method=fitting_method)
    # three possible methods are: dogbox, lm, trf
    fit_A = parameters[0]
    fit_B = parameters[1]
    #fit_C = parameters[1]

    xspace = np.linspace(0.1, 8, 100)
    fit_y = model_function(xspace, fit_A, fit_B)
    plt.plot(x, target_value, 'o', label='cpu')
    plt.plot(xspace, fit_y, '-', label='fit')
    plt.xlabel("x")
    plt.ylabel("time/s")
    plt.title("Fitting Plot")
    plt.legend()
    plt.savefig(type_of_fit+"fit.jpg")

    perr = np.sqrt(np.diag(covariance))
    return parameters, perr
    # print(perr)
    # print(fit_A,fit_B)


def general_plot(x_label, figsize=(7, 5)):

    fig = plt.figure(figsize=figsize)
    plt.plot(x_label, T_cpu, '.-', label='computing time')
    plt.plot(x_label, T_mb, '.-', label="memory time")
    plt.plot(x_label, T_ib, '.-', label='communication time')
    plt.plot(x_label, total, '.-', label='total time')
    plt.legend()
    plt.title("General Time")
    plt.xlabel("nodes_cores")
    plt.ylabel("time/s")
    fig.autofmt_xdate()
    plt.savefig("total.jpg")

    return 0
