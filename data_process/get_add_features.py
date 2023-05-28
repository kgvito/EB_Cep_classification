import numpy as np
from scipy.signal import argrelextrema


def find_nearest(array, value):
    """
    Find the nearest value in array
    """
    array = np.asarray(array).reshape(-1)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_greater_peak_coordination(t, y):
    greater_coordination = t[argrelextrema(y, np.greater)]
    greater_value_time = find_nearest(greater_coordination, 0.)
    greater_flux = y[np.where(t == greater_value_time)[0]][0]
    return greater_value_time, greater_flux


def split_light_curve_to_left_right(t, y, greater_value_time):
    """
    Split light curve into 2 parts
    """
    left_idx = np.where(t < greater_value_time)[0]
    right_idx = np.where(t >= greater_value_time)[0]
    left_t = t[left_idx]
    left_y = y[left_idx]
    right_t = t[right_idx]
    right_y = y[right_idx]
    return left_t, left_y, right_t, right_y


def get_depth_ratio(left_y, right_y):
    """
    Get depth ratio
    """
    left_depth = np.max(left_y) - np.min(left_y)
    right_depth = np.max(right_y) - np.min(right_y)
    return (np.max([left_depth, right_depth]) - np.min([left_depth, right_depth])) / (
            np.max(np.concatenate([left_y, right_y])) - np.min(np.concatenate([left_y, right_y])))


def get_slope_diff_ratio(left_t, left_y, right_t, right_y):
    left_max_idx = np.where(left_y == np.max(left_y))[0]
    left_min_idx = np.where(left_y == np.min(left_y))[0]
    right_max_idx = np.where(right_y == np.max(right_y))[0]
    right_min_idx = np.where(right_y == np.min(right_y))[0]
    if len(left_max_idx) > 1:
        left_max_idx = [left_max_idx[0]]
    if len(left_min_idx) > 1:
        left_min_idx = [left_min_idx[0]]
    if len(right_max_idx) > 1:
        right_max_idx = [right_max_idx[0]]
    if len(right_min_idx) > 1:
        right_min_idx = [right_min_idx[0]]
    left_slope = ((left_y[left_max_idx] - left_y[left_min_idx]) / (left_t[left_max_idx] - left_t[left_min_idx]))[0]
    right_slope = ((right_y[right_max_idx] - right_y[right_min_idx]) / (right_t[right_max_idx] - right_t[right_min_idx]))[0]
    return max(abs(left_slope), abs(right_slope)) / min(abs(left_slope), abs(right_slope))
