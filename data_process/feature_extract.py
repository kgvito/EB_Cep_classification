import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_process.anomaly_process import delete_anomaly_data
from data_process.cepheid_light_curve import CepheidLightCurve
from data_process.eclipsing_binary_light_curve import EBLightCurve
from utils.util import parse_model_args


def save_lightcurves_to_class(save_name, args=None, lc_type="eclipsing_binary"):
    args = parse_model_args(args)
    # 获取工作空间路径
    root = os.path.dirname(os.path.abspath(__file__)).split("data_process")[0]

    # 定义原始数据路径
    origin_dir = os.path.join(root, "dataset", "raw_data", lc_type, "light_curves")
    catalog_dir = os.path.join(root, "dataset", "raw_data", lc_type, "catalog")
    save_root = os.path.join(root, args["dataset"], "light_curve_instance")

    # Todo: 说明数据类型
    # 导入光变曲线的目录文件
    variables_catalogs = pd.read_csv(
        os.path.join(catalog_dir, f"{lc_type}_catalog.csv"),
        low_memory=False)

    # 对光变曲线的属性进行约束，全部转换为小写，并且将"-"符号转换为"_"，对括号进行操作
    attributes = variables_catalogs.columns.values
    attributes = [attr.lower() for attr in attributes]
    for i in range(len(attributes)):
        if "-" in attributes[i]:
            attributes[i] = attributes[i].replace("-", "_")
        if "(" in attributes[i] and ")" in attributes[i]:
            attributes[i] = attributes[i].replace("(", "_")
            attributes[i] = attributes[i].replace(")", "")
    variables_catalogs.columns = attributes
    data = variables_catalogs.to_dict('records')  # 将数据转换为records格式，也就是字典格式

    lc_list = []
    save_dir = os.path.join(save_root, save_name)
    os.makedirs(save_dir, exist_ok=True)

    dat_list = glob.glob(os.path.join(origin_dir, "*.dat"))
    source_id_list = [i.split("\\")[-1][:-4] for i in dat_list]

    for i in tqdm(data):
        if "lc" + str(i["source_id"]) in source_id_list:
            # 根据项目的id进行匹配，如果id匹配成果，则对数据进行读取
            load_csv = pd.read_csv(os.path.join(origin_dir, "lc" + i["source_id"] + ".dat"),
                                   delim_whitespace=True,
                                   comment='#',
                                   names=["time", "cam", "mag", "mag_err", "flux", "flux_err"], engine='c')
            # 由于对一颗变星拍摄时可能会使用不同的望远镜，因此如果数据中有多个望远镜的拍摄结果，则把拍摄结果分开
            lc_group = load_csv.groupby("cam")
            for group in lc_group:
                # 有时在记录中，因为种种原因无法获得精准的光度，会以<代替，说明此时的光度不大于一个值，对于这种情况无法转换成numpy形式，因此剔除或存储为abnormal形式
                idx = []
                y = group[1]["mag"].tolist()
                for index in range(len(y)):
                    if ">" in str(y[index]) or "<" in str(y[index]):
                        idx.append(index)
                y = np.delete(y, idx).astype(np.float_)
                if len(y) < 80:
                    continue
                t = np.delete(group[1]["time"].tolist(), idx).astype(np.float_)
                dy = np.delete(group[1]["mag_err"].tolist(), idx).astype(np.float_)

                t, y, dy = delete_anomaly_data(t, y, dy)

                try:
                    if lc_type == "eclipsing_binary":
                        lc = EBLightCurve(time=t, flux=y, flux_err=dy)
                    elif lc_type == "cepheid":
                        lc = CepheidLightCurve(time=t, flux=y, flux_err=dy)
                    else:
                        assert False, "lc_type error"
                    for k, v in i.items():
                        lc.meta[k] = v
                    lc.meta["main_type"] = save_name
                    lc.run()
                    lc_list.append(lc)
                except:
                    continue

    lc_save_dir = os.path.join(save_root, save_name)
    os.makedirs(lc_save_dir, exist_ok=True)
    np.save(os.path.join(lc_save_dir, f'lc_{save_name}.npy'), lc_list)


if __name__ == '__main__':
    save_lightcurves_to_class(save_name="eclipsing_binary",lc_type="eclipsing_binary")
