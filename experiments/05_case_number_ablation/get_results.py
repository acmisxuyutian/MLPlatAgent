from utils.utils import get_project_root
import os
import pandas as pd
llm = "qwen2_5-14b-coder"
benchmarks = {
    "uci": "UCI",
    "ci": "CI",
    "mi": "MI",
    "mlb": "MLB",
    "dseval": "DSEval"
}
cases_number = [0, 1, 2, 4, 5]
import json
def get_instrutions():
    uci_path = os.path.join(get_project_root(), r'data/benchmark/NL2Workflow/UCI.json')
    with open(uci_path, 'r', encoding='utf-8') as f:
        uci_data = json.load(f)
    ci_path = os.path.join(get_project_root(), r'data/benchmark/NL2Workflow/CI.json')
    with open(ci_path, 'r', encoding='utf-8') as f:
        ci_data = json.load(f)
    mi_path = os.path.join(get_project_root(), r'data/benchmark/NL2Workflow/MI.json')
    with open(mi_path, 'r', encoding='utf-8') as f:
        mi_data = json.load(f)
    ml_benchmark_path = os.path.join(get_project_root(), r'data/benchmark/NL2Workflow/ml_benchmark.json')
    with open(ml_benchmark_path, 'r', encoding='utf-8') as f:
        ml_benchmark = json.load(f)
    dseval_kaggle_path = os.path.join(get_project_root(), r'data/benchmark/NL2Workflow/dseval_kaggle.json')
    with open(dseval_kaggle_path, 'r', encoding='utf-8') as f:
        dseval_kaggle = json.load(f)
    return [i["Instruction"] for i in uci_data[:14] + ci_data[:14] + mi_data + ml_benchmark + dseval_kaggle]
instrutions = get_instrutions()
root_dir = os.path.join(get_project_root(), r"experiments/MLAgent/default")
for c in cases_number:
    for benchmark in benchmarks:
        try:
            data = None
            for i in range(5):
                data0 = pd.read_csv(os.path.join(root_dir, f"{benchmark}_{llm}_default_{c}_{i}.csv"))
                data0 = data0[data0["instruction"].isin(instrutions)]
                if data is None:
                    data = data0
                else:
                    data = pd.concat([data, data0])
            data.to_csv(os.path.join(get_project_root(), "experiments_results复现实验结果/05_case_number_ablation", llm, f"{benchmark}_{c}.csv"), index=False)
        except Exception as e:
            print(f"Error processing {c}_{benchmark}: {e}")