import glob
from collections import OrderedDict
from typing import Union, Any

import pandas as pd
from astropy.time import Time
from astropy.timeseries import LombScargle
from lightkurve import LightCurve

import scipy.stats as ss
from scipy.optimize import leastsq
from numpy import *
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class ASASLightCurve(LightCurve):
    def run(self):
        pass

    def shallow_run(self, measurements, errors):
        """
        提取不基于周期的光变曲线特征，其中使用到的特征包括:权重和，均值，方差，权重的均值和权重的方差
        偏度、峰度、夏皮罗威尔克检验系数等
        :param measurements: 光变曲线的观测值
        :param errors: 光变曲线的观测误差
        :return:
        """
        """提取不基于周期的曲线特征"""

        weight = 1. / errors
        self.weighted_sum = np.sum(weight)

        # 计算光变曲线的均值，中位数和方差
        self.meta["mean"] = np.mean(measurements)
        self.meta["median"] = np.median(measurements)
        self.meta["std"] = np.std(measurements)

        # 计算权重的均值和方差
        self.weighted_mean = np.sum(measurements * weight) / self.weighted_sum
        self.weighted_std = np.sqrt(
            np.sum((measurements - self.weighted_mean) ** 2 * weight) / self.weighted_sum)

        self.meta["skewness"] = ss.skew(measurements)
        self.meta["kurtosis"] = ss.kurtosis(measurements)
        shapiro = ss.shapiro(measurements)
        self.meta["shapiro_w"] = shapiro[0]
        self.meta["IQR"] = ss.iqr(measurements)
        # 计算星等最大值与最小值之间的差
        self.meta["p2p"] = np.max(measurements) - np.min(measurements)
        # 提取光变曲线斯泰森K系数（斯泰森变异系数）
        self.meta["stetson_k"] = self.get_stetson_k(measurements, float(self.meta["median"]), errors)
        # 计算星等的中位数绝对偏差
        self.meta["MAD"] = self.get_MAD(measurements)

        # 计算比平均幅度高的值与比平均幅度低的值之间的比值
        self.meta["hl_amp_ratio"] = self.half_mag_amplitude_ratio(
            measurements, float(self.meta["median"]), weight)

        # 计算观测最大值与观测最小值比
        self.meta["Mm_ratio"] = self.get_Mm_ratio()

    def deep_run(self, time, measurements, period, epoch_hjd):
        """提取基于周期的特征。"""
        # 使用Lomb-Scargle算法计算光变曲线的周期
        self.get_fourier_feature(time, measurements, period)

        flc = self.fold(period=period, epoch_time=epoch_hjd)
        folded_time = flc.time.value
        folded_flux = flc.flux.value

        # phase Eta
        self.meta["phase_eta"] = self.get_eta(folded_flux, self.weighted_std)

        # 斜率百分位数
        self.meta["slope_per10"], self.meta["slope_per90"], self.meta["slope_per20"], self.meta[
            "slope_per80"] = self.slope_percentile(folded_time, folded_flux)
        # 相位累积和
        self.meta["phase_cusum"] = self.get_cusum(folded_flux)

    def get_fourier_feature(self, date, mag, period):
        """
        将信号进行傅里叶分解，并计算傅里叶特征
        :param date: 时间
        :param mag: 星等
        :param period: 周期
        :return:
        """
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

        self.meta["a1"] = p1[1]
        self.meta["b1"] = p1[2]
        self.meta["a2"] = p1[3]
        self.meta["b2"] = p1[4]
        self.meta["a3"] = p1[5]
        self.meta["b3"] = p1[6]
        self.meta["a4"] = p1[7]
        self.meta["b4"] = p1[8]
        self.meta["a5"] = p1[9]
        self.meta["b5"] = p1[10]

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
    def get_lag_1_autocorrelation(mag, err):
        m_bar = (mag / err ** 2).sum() / (1 / err ** 2).sum()
        diff_1 = mag[:len(mag) - 1] - m_bar
        diff_2 = mag[1:] - m_bar
        l1 = (np.sum(diff_1 * diff_2)) / (np.sum((mag - m_bar) ** 2))
        return l1

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
