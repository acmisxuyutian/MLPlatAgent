import json
import os

import numpy as np
from colorama import Fore, Style

def print_with_color(text: str, color=""):
    if color == "red":
        print(Fore.RED + text)
    elif color == "green":
        print(Fore.GREEN + text)
    elif color == "yellow":
        print(Fore.YELLOW + text)
    elif color == "blue":
        print(Fore.BLUE + text)
    elif color == "magenta":
        print(Fore.MAGENTA + text)
    elif color == "cyan":
        print(Fore.CYAN + text)
    elif color == "white":
        print(Fore.WHITE + text)
    elif color == "black":
        print(Fore.BLACK + text)
    else:
        print(text)
    print(Style.RESET_ALL)

def load_json(text):
    try:
        return json.loads(text)
    except:
        import re
        json_pattern = re.compile(r"```(\s*json)?$(.*?)^```", re.MULTILINE | re.DOTALL)
        matches = re.findall(json_pattern, text)
        if len(matches) > 0:
            text_json = matches[0][1].strip()
        else:
            print("没有找到JSON代码块")
            return {}
        try:
            return json.loads(text_json)
        except:
            print(f"JSON解析错误")
            return {}

def load_python(text):
    try:
        import re
        pattern = re.compile(r"```(\s*python)?$(.*?)^```", re.MULTILINE | re.DOTALL)
        matches = re.findall(pattern, text)
        if len(matches) > 0:
            return True, matches[0][1].strip()
        else:
            return False, "没有找到Python代码块"
    except Exception as e:
        return False, f"python文本解析错误: 错误为\n{e}"

def get_project_root():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 向上一级
    current_directory = os.path.dirname(current_file_path)
    # 再次向上一级
    project_root = os.path.dirname(current_directory)

    return project_root

def plot_line_chart(X, Y, name):
    import matplotlib
    matplotlib.use('TkAgg')  # 或者其他适合您环境的后端
    import matplotlib.pyplot as plt

    # Sample data
    iterations = X
    scores = Y

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, scores, marker='o')
    plt.title('Relationship between Iterations and Score')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.grid(True)
    # 设置横坐标刻度
    plt.xticks(X)

    # 保存图表到文件
    project_root = get_project_root()
    save_path = os.path.join(project_root, f'data/line_chart/{name}.png')

    plt.savefig(save_path)

    # # 展示图表（可选）
    # plt.show()

def update_data_info(instruction="",dataset_info=""):
    data_info_path = os.path.join(get_project_root(), "data/data_info.json")
    try:
        with open(data_info_path, 'r', encoding='utf-8') as f:
            data_info = json.load(f)
    except:
        data_info = {
            "instruction": "",
            "dataset_info": ""
        }
    if instruction != "":
        data_info["instruction"] = instruction
    if data_info != "":
        data_info["dataset_info"] = dataset_info
    with open(data_info_path, 'w', encoding='utf8') as f:
        json.dump(data_info, f, ensure_ascii=False, indent=4)