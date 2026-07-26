from utils.utils import get_project_root
import os
import pandas as pd
benchmarks = {
    "uci": "UCI",
    "ci": "CI",
    "mi": "MI",
    "mlb": "MLB",
    "dseval": "DSEval"
}
tool_retrieve_type_map = ["Default", "wo_Task", "wo_Semantic", "wo_LLM", "Semantic"]
llm = "qwen2_5-72b"
# llm = "qwen2_5-14b-coder"
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

root_dir = r"/mnt/d/PythonCode/MLAgent/experiments/MLAgent/tool_retrieve_type"

for method in [0, 1, 2,3,4]:
    for benchmark in benchmarks:
        try:
            data = None
            for i in range(5):
                data0 = None
                if method == 0:
                    data_path = os.path.join(root_dir, f"{benchmark}_{llm}_default_{i}.csv")
                else:
                    data_path = os.path.join(root_dir, f"{benchmark}_{llm}_{method}_default_{i}.csv")
                data0 = pd.read_csv(data_path)
                data0 = data0[data0["instruction"].isin(instrutions)]
                if data is None:
                    data = data0
                else:
                    data = pd.concat([data, data0])
            data.to_csv(os.path.join(get_project_root(), "experiments/06_tool_retrieve_type", tool_retrieve_type_map[method], f"{benchmark}.csv"),index=False)
        except Exception as e:
            print(f"Error processing {method}_{benchmark}: {e}")