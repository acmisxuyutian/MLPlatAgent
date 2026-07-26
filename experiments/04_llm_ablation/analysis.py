import pandas as pd
import os
from utils.utils import get_project_root
analysis_result = {
    "LLM": [],
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
    "DSEval_WGF1": [],
    # "Average_Tokens": [],
    # "UCI_Tokens": [],
    # "CI_Tokens": [],
    # "MI_Tokens": [],
    # "MLB_Tokens": [],
    # "DSEval_Tokens": []
}

root = os.path.join(get_project_root(), r"experiments/04_llm_ablation")

llms = ["glm4-0520", "qwen2_5-72b", "llama3_1-70b", "gpt4o-mini", "gpt3_5-0125", "qwen2_5-14b", "qwen2_5-coder-14b", "llama3_1-8b"]

benchmarks = {
    "uci": "UCI",
    "ci": "CI",
    "mi": "MI",
    "mlb": "MLB",
    "dseval": "DSEval"
}
import json
def get_instrutions():
    uci_path = os.path.join(get_project_root(), r'data/benchmark/DSEval-Kaggle-Ext/UCI.json')
    with open(uci_path, 'r', encoding='utf-8') as f:
        uci_data = json.load(f)
    ci_path = os.path.join(get_project_root(), r'data/benchmark/DSEval-Kaggle-Ext/CI.json')
    with open(ci_path, 'r', encoding='utf-8') as f:
        ci_data = json.load(f)
    mi_path = os.path.join(get_project_root(), r'data/benchmark/DSEval-Kaggle-Ext/MI.json')
    with open(mi_path, 'r', encoding='utf-8') as f:
        mi_data = json.load(f)
    ml_benchmark_path = os.path.join(get_project_root(), r'data/benchmark/ml_benchmark.json')
    with open(ml_benchmark_path, 'r', encoding='utf-8') as f:
        ml_benchmark = json.load(f)
    dseval_kaggle_path = os.path.join(get_project_root(), r'data/benchmark/dseval_kaggle.json')
    with open(dseval_kaggle_path, 'r', encoding='utf-8') as f:
        dseval_kaggle = json.load(f)
    return [i["Instruction"] for i in uci_data[:14] + ci_data[:14] + mi_data + ml_benchmark + dseval_kaggle]
instrutions = get_instrutions()

for llm in llms:
    analysis_result["LLM"].append(llm)
    for b in benchmarks:
        file_path = os.path.join(root, "data", f"{llm}_{b}.csv")
        data = pd.read_csv(file_path)
        data = data[data["instruction"].isin(instrutions)]
        try:
            rcr = data["score"].mean()
        except:
            rcr = data["rcr"].mean()
        wgf1 = data["wgf1"].mean()
        # tokens = round(data["total_tokens"].mean(), 0)
        analysis_result[f"{benchmarks[b]}_RCR"].append(rcr)
        analysis_result[f"{benchmarks[b]}_WGF1"].append(wgf1)
        # analysis_result[f"{benchmarks[b]}_Tokens"].append(tokens)

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
    # analysis_result["Average_Tokens"].append(
    #     round(
    #         (analysis_result["UCI_Tokens"][-1] +
    #          analysis_result["CI_Tokens"][-1] +
    #          analysis_result["MI_Tokens"][-1] +
    #          analysis_result["MLB_Tokens"][-1] +
    #          analysis_result["DSEval_Tokens"][-1]) / 5,
    #         0
    #     )
    # )

# 保存结果
pd.DataFrame(analysis_result).to_csv(os.path.join(root, "analysis_result.csv"), index=False)

sorted_df = pd.DataFrame(analysis_result).sort_values(by='Average_RCR', ascending=False)
sorted_df.to_csv(os.path.join(root, "analysis_result_sorted_rcr.csv"), index=False)

sorted_df = pd.DataFrame(analysis_result).sort_values(by='Average_WGF1', ascending=False)
sorted_df.to_csv(os.path.join(root, "analysis_result_sorted_wgf1.csv"), index=False)
