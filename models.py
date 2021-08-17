def model_function1(x, a):
    """
    The estimated model function for the CPU extrapolation method. This is the version for CPU computing and Memory Bandwidth.

    """
    y = a/x
    return y


def model_function2(x, a, b):
    """
    The estimated model function for the CPU extrapolation method. This is the version for CPU computing and Memory Bandwidth with one more variable.

    """
    y = a/(x+b)
    return y


def model_function3(x, a, b):
    """
    The estimated model function for the CPU extrapolation method. This is the 3 variables version for InifiniBand communication time.

    """
    y = a/x + b
    return y


def model_function4(x, a, b, c):
    """
    The estimated model function for the CPU extrapolation method. This is the 3 variables version for InifiniBand communication time with one more variable.

    """
    y = a/(x+b) + c
    return y
