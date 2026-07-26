import pandas as pd
import os
from utils.utils import get_project_root
analysis_result = {
    "LLM": [],
    "Cases_Number": [],
    "Average_RCR": [],
    "Average_WGF1": [],
    "UCI_RCR": [],
    "UCI_WGF1": [],
    "CI_RCR": [],
    "CI_WGF1": [],
    "MI_RCR": [],
    "MI_WGF1": [],
    "MLB_RCR": [],
    "MLB_WGF1": [],
    "DSEval_RCR": [],
    "DSEval_WGF1": []
}
root = os.path.join(get_project_root(), r"experiments/05_case_number_ablation")
llms = ["glm4-0520", "qwen2_5-72b", "qwen2_5-14b-coder"]
cases_number = [0, 1, 2, 3, 4, 5]
benchmarks = {
    "uci": "UCI",
    "ci": "CI",
    "mi": "MI",
    "mlb": "MLB",
    "dseval": "DSEval"
}

for llm in llms:
    for case in cases_number:
        analysis_result["Cases_Number"].append(case)
        analysis_result["LLM"].append(llm)
        for b in benchmarks:
            file_path = os.path.join(root, llm, f"{b}_{case}.csv")
            data = pd.read_csv(file_path)
            try:
                rcr = data["score"].mean()
            except:
                rcr = data["rcr"].mean()
            wgf1 = data["wgf1"].mean()
            analysis_result[f"{benchmarks[b]}_RCR"].append(rcr)
            analysis_result[f"{benchmarks[b]}_WGF1"].append(wgf1)

        # 计算平均 RCR 和 Tokens
        analysis_result["Average_RCR"].append(
            round(
                (analysis_result["UCI_RCR"][-1] +
                 analysis_result["CI_RCR"][-1] +
                 analysis_result["MI_RCR"][-1] +
                 analysis_result["MLB_RCR"][-1] +
                 analysis_result["DSEval_RCR"][-1]) / 5,
                4
            )
        )
        analysis_result["Average_WGF1"].append(
            round(
                (analysis_result["UCI_WGF1"][-1] +
                 analysis_result["CI_WGF1"][-1] +
                 analysis_result["MI_WGF1"][-1] +
                 analysis_result["MLB_WGF1"][-1] +
                 analysis_result["DSEval_WGF1"][-1]) / 5,
                4
            )
        )

# 保存结果
pd.DataFrame(analysis_result).to_csv(os.path.join(root, "analysis_result.csv"), index=False)
