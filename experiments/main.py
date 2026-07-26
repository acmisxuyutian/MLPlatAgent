from utils.evaluate_utils import workflow_graph_f1
import pandas as pd
import os
from utils.utils import get_project_root
root_dir = os.path.join(get_project_root(), 'experiments', "06_tool_retrieve_type")

# 存储文件路径列表
csv_file_paths = []

# 递归遍历所有文件夹
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # 判断后缀是csv（忽略大小写.CSV）
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(dirpath, filename)
            print(f"{file_path}")
            data = pd.read_csv(file_path)
            if "instruction" in data.columns and "requirements" in data.columns and "workflow" in data.columns:
                wgf1 = []
                for index, row in data.iterrows():
                    rg = row["requirements"]
                    wg = row["workflow"]
                    wgf1.append(workflow_graph_f1(wg, rg))
                data["wgf1"] = wgf1
                data.to_csv(file_path, index=False)