from astropy.timeseries import LombScargle
from scipy.optimize import leastsq
from numpy import *
import numpy as np
import warnings

from data_process.light_curve import ASASLightCurve
from utils.util import parse_model_args

from data_process.get_add_features import split_light_curve_to_left_right, get_greater_peak_coordination, \
    get_depth_ratio, get_slope_diff_ratio
from data_process.get_similarity import LeftRightSimilarity

warnings.filterwarnings("ignore")


# 将数据保存为npy文件


class RRLyraeLightCurve(ASASLightCurve):
    def run(self):
        self.shallow_run(self.flux.value, self.flux_err.value)
        if self.meta["period"] is None:
            frequency = np.linspace(0., 10., 100000)
            power = LombScargle(self.time.value, self.flux.value, self.flux_err.value).power(frequency)
            self.meta["period"] = (1. / frequency[np.argmax(power)])

        flc = self.fold(self.meta["period"], float_(self.meta["epoch_hjd"]))
        self.time, self.flux, self.flux_err = flc.time.value, flc.flux.value, flc.flux_err.value
        self.deep_run(self.time.value, self.flux.value, self.meta["period"], self.meta["epoch_hjd"])

    def shallow_run(self, measurements, errors):
        super().shallow_run(measurements, errors)

    def deep_run(self, time, measurements, period, epoch_hjd):
        self.get_fourier_feature(time, measurements, period)

        flc = self.fold(period=period, epoch_time=epoch_hjd)
        folded_time = flc.time.value
        folded_flux = flc.flux.value
        folded_flux_err = flc.flux_err.value

        self.meta["period_ratio"] = self.get_period_ratio(folded_time, folded_flux, folded_flux_err)

        # phase Eta
        self.meta["phase_eta"] = self.get_eta(folded_flux, self.weighted_std)

        # 斜率百分位数
        self.meta["slope_per10"], self.meta["slope_per90"], self.meta["slope_per20"], self.meta[
            "slope_per80"] = self.slope_percentile(folded_time, folded_flux)
        # 相位累积和
        self.meta["phase_cusum"] = self.get_cusum(folded_flux)

    def get_fourier_feature(self, date, mag, period):
        order = 5
        date = np.array(date)
        period = float(period)
        p0 = np.ones(order * 2 + 1)
        date_period = (date % period) / period
        p1, success = leastsq(self.residuals, p0, args=(date_period, mag, order), maxfev=20000)

        # 幅值
        self.meta["h_1"] = np.sqrt(p1[1] ** 2 + p1[2] ** 2)
        # 二一阶振幅比
        self.meta["r21"] = np.sqrt(p1[3] ** 2 + p1[4] ** 2) / self.meta["h_1"]
        # 三一阶振幅比
        self.meta["r31"] = np.sqrt(p1[5] ** 2 + p1[6] ** 2) / self.meta["h_1"]
        # 三二阶振幅比
        self.meta["r32"] = np.sqrt(p1[5] ** 2 + p1[6] ** 2) / np.sqrt(p1[3] ** 2 + p1[4] ** 2)
        # 四一阶振幅比
        self.meta["r41"] = np.sqrt(p1[7] ** 2 + p1[8] ** 2) / self.meta["h_1"]
        # 四二阶振幅比
        self.meta["r42"] = np.sqrt(p1[7] ** 2 + p1[8] ** 2) / np.sqrt(p1[3] ** 2 + p1[4] ** 2)
        # 四三阶振幅比
        self.meta["r43"] = np.sqrt(p1[7] ** 2 + p1[8] ** 2) / np.sqrt(p1[5] ** 2 + p1[6] ** 2)
        # 五一阶振幅比
        self.meta["r51"] = np.sqrt(p1[9] ** 2 + p1[10] ** 2) / self.meta["h_1"]
        # 五二阶振幅比
        self.meta["r52"] = np.sqrt(p1[9] ** 2 + p1[10] ** 2) / np.sqrt(p1[3] ** 2 + p1[4] ** 2)
        # 五三阶振幅比
        self.meta["r53"] = np.sqrt(p1[9] ** 2 + p1[10] ** 2) / np.sqrt(p1[5] ** 2 + p1[6] ** 2)
        # 五四阶振幅比
        self.meta["r54"] = np.sqrt(p1[9] ** 2 + p1[10] ** 2) / np.sqrt(p1[7] ** 2 + p1[8] ** 2)
        # 基础相位
        self.meta["f_phase"] = np.arctan(-p1[1] / p1[2])
        # 二一阶相位差
        self.meta["phi21"] = np.arctan(-p1[3] / p1[4]) - 2. * self.meta["f_phase"]
        # 三一阶相位差
        self.meta["phi31"] = np.arctan(-p1[5] / p1[6]) - 3. * self.meta["f_phase"]
        # 三二阶相位差
        self.meta["phi32"] = np.arctan(-p1[5] / p1[6]) - 3. * np.arctan(-p1[3] / p1[4])
        # 四一阶相位差
        self.meta["phi41"] = np.arctan(-p1[7] / p1[8]) - 4. * self.meta["f_phase"]
        # 四二阶相位差
        self.meta["phi42"] = np.arctan(-p1[7] / p1[8]) - 4. * np.arctan(-p1[3] / p1[4])
        # 四三阶相位差
        self.meta["phi43"] = np.arctan(-p1[7] / p1[8]) - 4. * np.arctan(-p1[5] / p1[6])
        # 五一阶相位差
        self.meta["phi51"] = np.arctan(-p1[9] / p1[10]) - 5. * self.meta["f_phase"]
        # 五二阶相位差
        self.meta["phi52"] = np.arctan(-p1[9] / p1[10]) - 5. * np.arctan(-p1[3] / p1[4])
        # 五三阶相位差
        self.meta["phi53"] = np.arctan(-p1[9] / p1[10]) - 5. * np.arctan(-p1[5] / p1[6])
        # 五四阶相位差
        self.meta["phi54"] = np.arctan(-p1[9] / p1[10]) - 5. * np.arctan(-p1[7] / p1[8])

    def residuals(self, pars, x, y, order):
        # 通过傅里叶级数所拟合出来的值与真实值之间的误差
        return y - self.fourier_series(pars, x, order)

    @staticmethod
    def fourier_series(pars, x, order):
        # 傅里叶级数 s = a_0 + \sum_{i=1}^{N} a_i*sin(2*pi*i*x)+b_i*cos(2*pi*i*x)
        sum = pars[0]
        for i in range(order):
            sum += pars[i * 2 + 1] * np.sin(2 * np.pi * (i + 1) * x) \
                   + pars[i * 2 + 2] * np.cos(2 * np.pi * (i + 1) * x)
        return sum

    @staticmethod
    def get_stetson_k(mag, avg, err):
        residual = (mag - avg) / err
        stetson_k = np.sum(np.fabs(residual)) \
                    / np.sqrt(np.sum(residual * residual)) / np.sqrt(len(mag))
        return stetson_k

    def get_MAD(self, mag):
        return np.median(np.fabs(mag - self.meta["median"]))

    @staticmethod
    def half_mag_amplitude_ratio(mag, avg, weight):
        index = np.where(mag > avg)
        lower_weight = weight[index]
        lower_weight_sum = np.sum(lower_weight)
        lower_mag = mag[index]
        lower_weighted_std = np.sum((lower_mag
                                     - avg) ** 2 * lower_weight) / \
                             lower_weight_sum
        index = np.where(mag <= avg)
        higher_weight = weight[index]
        higher_weight_sum = np.sum(higher_weight)
        higher_mag = mag[index]
        higher_weighted_std = np.sum((higher_mag
                                      - avg) ** 2 * higher_weight) / \
                              higher_weight_sum

        return np.sqrt(lower_weighted_std / higher_weighted_std)

    @staticmethod
    def get_eta(mag, std):
        diff = mag[1:] - mag[:len(mag) - 1]
        eta = np.sum(diff * diff) / (len(mag) - 1.) / std / std
        return eta

    @staticmethod
    def slope_percentile(date, mag):
        date_diff = date[1:] - date[:len(date) - 1]
        mag_diff = mag[1:] - mag[:len(mag) - 1]
        index = np.where(mag_diff != 0.)
        date_diff = date_diff[index]
        mag_diff = mag_diff[index]
        slope = date_diff / mag_diff
        percentile_10 = np.percentile(slope, 10.)
        percentile_90 = np.percentile(slope, 90.)
        percentile_20 = np.percentile(slope, 20.)
        percentile_80 = np.percentile(slope, 80.)

        return percentile_10, percentile_90, percentile_20, percentile_80

    def get_cusum(self, mag):
        c = np.cumsum(mag - self.weighted_mean) / len(mag) / self.weighted_std
        return np.max(c) - np.min(c)

    def get_Mm_ratio(self):
        median_mag = (np.max(self.flux.value) + np.min(self.flux.value)) / 2
        return max(len(self.flux.value[self.flux.value > median_mag]), len(
            self.flux.value[self.flux.value < median_mag])) / min(len(self.flux.value[self.flux.value > median_mag]),
                                                                  len(self.flux.value[self.flux.value < median_mag]))

    def get_period_ratio(self, time, mag, mag_err):
        frequency = np.linspace(0., 10., 100000)
        p = 1. / frequency
        power = LombScargle(time, mag, mag_err).power(frequency)
        p1 = (1. / frequency[np.argmax(power)])
        p_range = np.where((p > (p1 - 0.5)) & (p < (p1 + 0.5)))
        p2 = np.unique(p_range)[-2]
        return p2 / p1
