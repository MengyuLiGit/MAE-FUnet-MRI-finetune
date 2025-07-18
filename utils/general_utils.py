import os
import re
from datetime import date
import numpy as np
import json
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy"s implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]

import json
import re

import json
import re

def safe_json_load(path):
    with open(path, 'r') as f:
        raw = f.read()

        # This replaces any standalone -nan/nan/inf/-inf (surrounded by anything except quotes)
        raw = re.sub(
            r'(?<![\"\w])(-?nan|NaN|NAN|Infinity|-Infinity|INF|-INF)(?![\w\"])',
            'null',
            raw,
            flags=re.IGNORECASE
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to load {path}")
            print(f"Line {e.lineno}, Column {e.colno}: {e.msg}")
            print("----")
            raise



def fixPath(str1):
    '''
    Reformat path string to exceed 256 characters limit
    :param str1: input path string
    :return: new UNC formatted path string
    '''
    path = os.path.abspath(str1)

    if '²' in path:
        return path
    if path.startswith(u"/"):
        path = u"//?/UNC/" + path[2:]
    else:
        path = u"//?/" + path

    str1_unicode = path.encode('unicode_escape').decode()  # Step 1: Unicode
    file_path = os.path.abspath(os.path.normpath(str1_unicode))  # Step 3 and 4: no forward slashes and absolute path

    return file_path

def load_sorted_directory(directory, postfix_label = None):
    """
    load directory child item into a list with order according to natural keys
    (same order as the windows sort order)
    :param directory: string, path of root directory
    :return: list of strings, child item paths ordered by natural keys
    """
    try:
        directory = fixPath(directory)
        items = os.listdir(directory)
        items_output = []
        if postfix_label is not None:
            for file in items:
                file_extension = get_file_extension(file, prompt=postfix_label)
                file_extension_len = len(file_extension)
                if len(file) > file_extension_len:
                    if file[-file_extension_len:] == file_extension:
                        items_output.append(file)
        else:
            items_output = items
        items_output.sort(key=natural_keys)
        return items_output
    except Exception:
        return None

def sort_list_natural_keys(items):
    """
    sort a list of strings by natural keys
    :param items: list of strings, usually child dir/file names of a parent directory
    :return: sorted list of strings
    """

    items.sort(key=natural_keys)
    return items

def substring_after(s, delim):
    """
    return substring after the first delimiter string
    :param s: string, total string
    :param delim: string, delimiter string
    :return: string, substring
    """
    return s.partition(delim)[2]

def combine_str_list_to_path(list):
    """
    combine string list to a path
    :param list: list, list of strings
    :return: string, combined path
    """
    return "/".join(list)

def get_time_interval(earlier, later, unit='month'):
    """
    return time interval between earlier and later (in chosen time unit)
    :param earlier: list, [year, month, day]
    :param later: list, [year, month, day]
    :param unit: string, time unit of 'month', 'day', or 'year'
    :return: int, time interval in chosen time unit
    """
    if unit == 'year':
        return later[0] - earlier[0]
    elif unit == 'month':
        return 12 * (later[0] - earlier[0]) + (later[1] - earlier[1])
    elif unit == 'day':
        d0 = date(earlier[0], earlier[1], earlier[2])
        d1 = date(later[0], later[1], later[2])
        delta = d1 - d0
        return delta.days


def check_if_dir_list(root, item_list):
    """
    check if first item in root directory is a directory or a file
    :param root: string, root directory path
    :param item_list: string, list of item filenames
    :return: bool, True if first item in directory is a directory, False
    if it is a file or have no file/directory.
    """
    if item_list is not None and len(item_list) > 0:
        if os.path.isdir(fixPath(root + "/" + item_list[0])):
            return True
        else:
            return False
    else:
        return False


def check_if_dir(root, item_input):
    """
    check if given item is a directory or a file
    :param root: string, root directory path
    :param item_input: string, item filenames
    :return: bool, True if item in directory is a directory, False
    if it is a file or have no such file/directory.
    """
    if item_input is not None:
        if os.path.isdir(fixPath(root + "/" + item_input)):
            return True
        else:
            return False
    else:
        return False

def clear_string_char(string, char_list=[]):
    for char in char_list:
        string = string.replace(char, '')
    return string

def chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

import shutil

def copy_and_rename(src_path, dest_path, oldname, new_name):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    # Copy the file
    shutil.copy(src_path, dest_path)
    # Rename the copied file
    new_path = f"{dest_path}/{new_name}"
    shutil.move(f"{dest_path}/{oldname}", new_path)


def running_window(input_list, window_length):
    ### find the given range include the largest set of numbers in the list

    # Calculate the sum of values for each window of length 40
    window_sums = np.convolve(input_list, np.ones(window_length, dtype=int), 'valid')

    # Find the index of the maximum sum
    max_index = np.argmax(window_sums)

    # Define the range with the highest probability sum
    best_range = (max_index, max_index + window_length - 1)
    highest_sum = window_sums[max_index]

    return best_range, highest_sum

def get_file_extension(filename, prompt = ".nii"):
    ### get the file extension after the given prompt, return "" if no extension found

    # Check if ".nii" exists in the filename
    if prompt in filename:
        idx = filename.index(prompt)  # Find where ".nii" starts
        ext = filename[idx:]  # Extract from ".nii" to the end
    else:
        ext = ""  # No valid extension found
    return ext
def get_number_after_string(s, prompt):
    ### get number after given string and prompt, return None if no digit found
    pattern = rf"{re.escape(prompt)}(\d+)"
    match = re.search(pattern, s)
    return match.group(1) if match else None

from pathlib import Path

def get_root_folder(path_str):
    path = Path(path_str)
    parts = path.parts  # Get the path components as a tuple
    if len(parts) > 1:
        return str(Path(parts[0], parts[1]))  # Combine the first two parts
    return str(path)  # Return the original path if there's only one part

import pickle

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)