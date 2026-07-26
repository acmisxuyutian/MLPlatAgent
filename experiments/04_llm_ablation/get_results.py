from utils.utils import get_project_root
import os
import pandas as pd
llms_map = {
    "glm4-0520": "GLM-4",
    "qwen2_5-72b": "qwen2_5-72b-instruct",
    "llama3_1-70b": "llama70b",
    "gpt3_5-0125": "gpt-3.5-turbo-0125",
    "gpt4o-mini": "gpt-4o-mini",
    "glm4-9b": "glm4-9b-chat",
    "qwen2_5-14b": "qwen2_5-14b-instruct",
    "qwen2_5-coder-14b": "qwen2_5-coder-14b-instruct",
    "llama3_1-8b": "llama3_1-8b-instruct"
}
benchmarks = {
    "uci": "UCI",
    "ci": "CI",
    "mi": "MI",
    "mlb": "MLB",
    "dseval": "DSEval"
}
root_dir = os.path.join(get_project_root(), r"experiments/mlagent")
for llm in llms_map:
    for benchmark in benchmarks:
        try:
            data = None
            for i in range(3):
                data0 = None
                data1 = None
                if benchmark != "mi":
                    data0 = pd.read_csv(os.path.join(root_dir, "model_performance/default",
                                                     f"{benchmark}_{llms_map[llm]}_default_{i}.csv"))
                if benchmark not in ["mlb", "dseval"]:
                    data1 = pd.read_csv(os.path.join(root_dir, "workflow_construction/default",
                                                     f"{benchmark}_{llms_map[llm]}_default_{i}.csv"))
                if data is None:
                    if data0 is not None and data1 is not None:
                        data = pd.concat([data0, data1])
                    elif data0 is not None and data1 is None:
                        data = data0
                    elif data0 is None and data1 is not None:
                        data = data1

                else:
                    if data0 is not None and data1 is not None:
                        data = pd.concat([data, data0, data1])
                    elif data0 is not None and data1 is None:
                        data = pd.concat([data, data0])
                    elif data0 is None and data1 is not None:
                        data = pd.concat([data, data1])
            data.to_csv(
                os.path.join(get_project_root(), "experiments_results/04_llm_ablation/data", f"{llm}_{benchmark}.csv"),
                index=False)
        except Exception as e:
            print(f"Error processing {llm}_{benchmark}: {e}")