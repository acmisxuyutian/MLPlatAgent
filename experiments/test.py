from utils.evaluate_wgf1 import workflow_graph_f1
from utils.evaluate_rcr import RCR
from utils.evaluate_we import workflow_edit_metrics

import pandas as pd
import os
import json
from utils.utils import get_project_root
root_dir = os.path.join(get_project_root(), r"experiments/08_mi")
with open(os.path.join(get_project_root(), r"data/benchmark/DSEval-Kaggle-Ext/MI.json"), "r", encoding="utf-8") as f:
    MI = json.load(f)
_MI = {mi["Instruction"]: mi["workflow edit sets"] for mi in MI}

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
                rcrs = []
                weps = []
                wers = []
                wef1s = []
                ground_truth_workflow_edit_sets = []
                for index, row in data.iterrows():
                    rg = row["requirements"]
                    wg = row["workflow"]
                    wgf1.append(workflow_graph_f1(wg, rg))
                    rcrs.append(RCR(json.loads(rg), json.loads(wg)))
                    if "predicted_workflow_edit_sets" in data.columns:
                        wep, wer, wef1 = workflow_edit_metrics(
                            json.loads(row["predicted_workflow_edit_sets"]),
                            _MI[row["instruction"]],
                        )
                        ground_truth_workflow_edit_sets.append(json.dumps(_MI[row["instruction"]]))
                        weps.append(wep)
                        wers.append(wer)
                        wef1s.append(wef1)
                if len(weps) > 0:
                    data["wep"] = weps
                    data["wer"] = wers
                    data["wef1"] = wef1s
                    data["ground_truth_workflow_edit_sets"] = ground_truth_workflow_edit_sets
                data["wgf1"] = wgf1
                data["rcr"] = rcrs
                data.to_csv(file_path, index=False)