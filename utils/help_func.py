import numpy as np
import os

import torch



def print_var_detail(var, name=""):
    """
    Args: Print basic detail of a variable

    :param var: input variable
    :param name: variable name, default is empty
    :return: string with basic information
    """
    print(name, "is a ", type(var), "with shape",
          var.shape if torch.is_tensor(var) or isinstance(var, np.ndarray) else None,
          "max: ",
          var.max() if torch.is_tensor(var) and not torch.is_complex(var) or isinstance(var, np.ndarray) else None,
          "min: ",
          var.min() if torch.is_tensor(var) and not torch.is_complex(var) or isinstance(var, np.ndarray) else None)


def create_path(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    else:
        print("Path already exists.")
