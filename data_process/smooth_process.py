import numpy as np
from scipy.signal import argrelextrema


def get_super_smoother(t, y, dy, period):
    """
    SuperSmoother
    """
    from supersmoother import SuperSmoother
    ss = SuperSmoother(period=period)
    ss.fit(t, y, dy)
    t_ss = np.linspace(t.min(), t.max(), 500)
    y_ss = ss.predict(t_ss)
    return t_ss, y_ss
