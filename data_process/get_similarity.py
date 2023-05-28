import math
import operator
from functools import reduce

import numpy as np


class LeftRightSimilarity:
    def __init__(self, left_t, left_y, right_t, right_y, bin_num=30):
        self.left_t = left_t - min(left_t)
        self.left_y = left_y
        self.right_t = right_t - min(right_t)
        self.right_y = right_y
        self.bin_num = bin_num

        self.min_t = np.min([np.min(left_t), np.min(right_t)])
        self.max_t = np.max([np.max(left_t), np.max(right_t)])
        self.min_y = np.min([np.min(left_y), np.min(right_y)])
        self.max_y = np.max([np.max(left_y), np.max(right_y)])

        self.bin_ranges = np.linspace(self.min_t, self.max_t, self.bin_num + 1, endpoint=True)

        self.area = (self.max_y - self.min_y) * (self.max_t - self.min_t)

    def calc_similarity(self):
        """
        Calculate similarity
        """
        all_similarity = 0.
        for i in range(self.bin_num - 1):

            left_range_1, right_range_1 = self.bin_ranges[i], self.bin_ranges[i + 1]
            left_range_2, right_range_2 = self.bin_ranges[i + 1], self.bin_ranges[i + 2]
            # print(self.right_t)
            left_idx_1 = np.where((self.left_t >= left_range_1) & (self.left_t < right_range_1))[0]
            right_idx_1 = np.where((self.right_t >= left_range_1) & (self.right_t < right_range_1))[0]
            temp_left_t_1_l = self.left_t[left_idx_1]
            temp_left_y_1_l = self.left_y[left_idx_1]
            temp_right_t_1_l = self.right_t[right_idx_1]
            temp_right_y_1_l = self.right_y[right_idx_1]

            left_idx_2 = np.where((self.left_t >= left_range_2) & (self.left_t < right_range_2))[0]
            right_idx_2 = np.where((self.right_t >= left_range_2) & (self.right_t < right_range_2))[0]
            temp_left_t_2_l = self.left_t[left_idx_2]
            temp_left_y_2_l = self.left_y[left_idx_2]
            temp_right_t_2_l = self.right_t[right_idx_2]
            temp_right_y_2_l = self.right_y[right_idx_2]

            if len(temp_left_t_1_l) == 0 or len(temp_left_t_2_l) == 0 or len(temp_right_t_1_l) == 0 or len(
                    temp_right_t_2_l) == 0:
                continue

            temp_left_t_1 = np.mean(temp_left_t_1_l)
            temp_left_y_1 = np.mean(temp_left_y_1_l)
            temp_right_t_1 = np.mean(temp_right_t_1_l)
            temp_right_y_1 = np.mean(temp_right_y_1_l)
            temp_left_t_2 = np.mean(temp_left_t_2_l)
            temp_left_y_2 = np.mean(temp_left_y_2_l)
            temp_right_t_2 = np.mean(temp_right_t_2_l)
            temp_right_y_2 = np.mean(temp_right_y_2_l)

            if (temp_right_y_1 - temp_left_y_1) * (temp_right_y_2 - temp_left_y_2) > 0:  # same direction
                points = [(temp_left_t_1, temp_left_y_1), (temp_left_t_2, temp_left_y_2),
                          (temp_right_t_1, temp_right_y_1), (temp_right_t_2, temp_right_y_2)]
                center = tuple(
                    map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
                points = sorted(points, key=lambda coord: (-135 - math.degrees(
                    math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
                a = self.compute_polygon_area(points)

            else:  # different direction
                x, y = self.get_line_cross_point([temp_left_t_2, temp_left_y_2, temp_left_t_1, temp_left_y_1],
                                                 [temp_right_t_2, temp_right_y_2, temp_right_t_1, temp_right_y_1])
                points_1 = [(temp_left_t_1, temp_left_y_1), (temp_right_t_1, temp_right_y_1), (x, y)]
                points_2 = [(temp_left_t_2, temp_left_y_2), (temp_right_t_2, temp_right_y_2), (x, y)]
                center_1 = tuple(
                    map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points_1), [len(points_1)] * 2))
                points_1 = sorted(points_1, key=lambda coord: (-135 - math.degrees(
                    math.atan2(*tuple(map(operator.sub, coord, center_1))[::-1]))) % 360, reverse=True)
                center_2 = tuple(
                    map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points_2), [len(points_2)] * 2))
                points_2 = sorted(points_2, key=lambda coord: (-135 - math.degrees(
                    math.atan2(*tuple(map(operator.sub, coord, center_2))[::-1]))) % 360, reverse=True)
                a = self.compute_polygon_area(points_1) + self.compute_polygon_area(points_2)
            all_similarity += a
        return 1. - all_similarity / self.area

    # 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
    @staticmethod
    def compute_polygon_area(points):
        point_num = len(points)
        if point_num < 3:
            return 0.0
        s = points[0][1] * (points[point_num - 1][0] - points[1][0])
        for i in range(1, point_num):
            s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
        return abs(s / 2.0)

    @staticmethod
    def calc_abc_from_line_2d(x0, y0, x1, y1):
        a = y0 - y1
        b = x1 - x0
        c = x0 * y1 - x1 * y0
        return a, b, c

    def get_line_cross_point(self, line1, line2):
        a0, b0, c0 = self.calc_abc_from_line_2d(*line1)
        a1, b1, c1 = self.calc_abc_from_line_2d(*line2)
        D = a0 * b1 - a1 * b0
        if D == 0:
            return None
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        return x, y
