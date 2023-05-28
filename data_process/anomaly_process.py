import numpy as np


def delete_anomaly_data(t, y, dy, min_quantile=1, max_quantile=99):
    idx = np.intersect1d(np.argwhere(
        np.percentile(y, min_quantile) < y), np.argwhere(
        np.percentile(y, max_quantile) > y))
    # print(idx)
    t = t[idx]
    y = y[idx]
    dy = dy[idx]
    return t, y, dy
