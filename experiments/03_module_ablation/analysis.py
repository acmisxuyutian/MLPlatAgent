import pandas as pd
import os
from utils.utils import get_project_root
analysis_result = {
    "method": [],
    "MLB+DSEval_RCR": [],
    "MLB+DSEval_WGF1": [],
    "MLB+DSEval_Tokens": [],
    "UCI+CI+MI_RCR": [],
    "UCI+CI+MI_WGF1": [],
    "UCI+CI+MI_Tokens": [],
    "MI_RCR": [],
    "MI_WGF1": [],
    "MI_Tokens": [],
    "UCI_RCR": [],
    "UCI_WGF1": [],
    "UCI_Tokens": [],
    "CI_RCR": [],
    "CI_WGF1": [],
    "CI_Tokens": [],
    "MLB_RCR": [],
    "MLB_WGF1": [],
    "MLB_Tokens": [],
    "DSEval_RCR": [],
    "DSEval_WGF1": [],
    "DSEval_Tokens": []
}
root = os.path.join(get_project_root(), r"experiments/03_module_ablation")
# step4
llms = ["qwen2_5-72b", "qwen2_5-14b-coder"]
methods = {
"_step3": "step3",
    # "_step3": "step3",
"_wo_step2": "wo_step2",
    # "_wo_step2": "wo_step2",
    "_wo_step1": "wo_step1",

    "": "defalut"
}
benchmarks = {
    "uci": "UCI",
    "ci": "CI",
    "mi": "MI",
    "mlb": "MLB",
    "dseval": "DSEval"
}
import statistics
def convert_cny_to_usd(cny_amount, exchange_rate=7.2):
    """
    将人民币金额转换为美元金额。

    参数:
    cny_amount (float): 人民币金额。
    exchange_rate (float): 人民币对美元的汇率，默认为 7.25。

    返回:
    float: 转换后的美元金额。如果输入无效，返回 None。
    """
    try:
        amount = float(cny_amount)
        if amount < 0:
            return None
        return amount / exchange_rate
    except (ValueError, TypeError):
        return None

for llm in llms:
    for method in methods:
        analysis_result["method"].append(methods[method])
        for b in benchmarks:
            file_path = os.path.join(root, llm, f"{b}{method}.csv")
            data = pd.read_csv(file_path)
            rcr = f'{round(statistics.mean(data["rcr"]), 3)}±{round(statistics.variance(data["rcr"]), 2)}'
            wgf1 = f'{round(statistics.mean(data["wgf1"]), 3)}±{round(statistics.variance(data["wgf1"]), 2)}'
            tokens = (data["input_tokens"].sum() * convert_cny_to_usd(0.004 * 1000) + data[
                "output_tokens"].sum() * convert_cny_to_usd(0.012 * 1000)) / 1000000
            analysis_result[f"{benchmarks[b]}_RCR"].append(rcr)
            analysis_result[f"{benchmarks[b]}_WGF1"].append(wgf1)
            analysis_result[f"{benchmarks[b]}_Tokens"].append(tokens)
        #
        # nl2workflow = pd.DataFrame()
        # for b in benchmarks:
        #     if b == "mlb": break
        #     file_path = os.path.join(root, llm, f"{b}{method}.csv")
        #     data = pd.read_csv(file_path)
        #     nl2workflow = pd.concat([nl2workflow, data])
        #
        # rcr = round(nl2workflow["score"].mean(), 4)
        # tokens = round(nl2workflow["total_tokens"].mean(), 0)
        # analysis_result["NL2Workflow_RCR"].append(rcr)
        # analysis_result["NL2Workflow_Tokens"].append(tokens)

        # 计算平均 RCR 和 Tokens
        data0 = pd.read_csv(os.path.join(root, llm, f"mlb{method}.csv"))
        data1 = pd.read_csv(os.path.join(root, llm, f"dseval{method}.csv"))
        dddd = pd.concat([data0, data1])

        analysis_result["MLB+DSEval_RCR"].append(
            f'{round(statistics.mean(dddd["rcr"]), 3)}±{round(statistics.variance(dddd["rcr"]), 2)}'
        )
        analysis_result["MLB+DSEval_WGF1"].append(
            f'{round(statistics.mean(dddd["wgf1"]), 3)}±{round(statistics.variance(dddd["wgf1"]), 2)}'
        )
        analysis_result["MLB+DSEval_Tokens"].append(
            round(
                (
                        analysis_result["MLB_Tokens"][-1] +
                        analysis_result["DSEval_Tokens"][-1]) / 2,
                3
            )
        )

        data0 = pd.read_csv(os.path.join(root, llm, f"mi{method}.csv"))
        data1 = pd.read_csv(os.path.join(root, llm, f"uci{method}.csv"))
        data2 = pd.read_csv(os.path.join(root, llm, f"ci{method}.csv"))
        dddd = pd.concat([data0, data1, data2])

        # 计算平均 RCR 和 Tokens
        analysis_result["UCI+CI+MI_RCR"].append(
            f'{round(statistics.mean(dddd["rcr"]), 3)}±{round(statistics.variance(dddd["rcr"]), 2)}'
        )
        analysis_result["UCI+CI+MI_WGF1"].append(
            f'{round(statistics.mean(dddd["wgf1"]), 3)}±{round(statistics.variance(dddd["wgf1"]), 2)}'
        )
        analysis_result["UCI+CI+MI_Tokens"].append(
            round(
                (
                        analysis_result["MI_Tokens"][-1] +
                        analysis_result["UCI_Tokens"][-1] +
                        analysis_result["CI_Tokens"][-1]) / 3,
                3
            )
        )

# 保存结果
pd.DataFrame(analysis_result).to_csv(os.path.join(root, "analysis_result.csv"), index=False)
