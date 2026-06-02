# -*- coding: utf-8 -*-
import json
import os
import time

from utils.logs import logger
from llm.llm import Qwen_Model
from utils.utils import get_project_root, load_json, load_python
from ml_platform.actions import action_agent
from prompts.executor_prompts import TOOLS_PROMPT, Experiences

CODE_INI = """from ml_platform.actions import action_agent

def add_node(widget_name, node_name):
    args = {"widget_name": widget_name, "node_name": node_name}
    node_id = action_agent.add_node(args)
    return node_id

def delete_node(node_id):
    args = {"node_id": node_id}
    return action_agent.delete_node(args)

def update_node_params(node_id, widget_name, node_params):
    args = {"node_id": node_id, "widget_name": widget_name, "node_params": node_params}
    return action_agent.update_node_params(args)

def add_edge(source_node_id, target_node_id):
    args = {"source_node_id": source_node_id, "target_node_id": target_node_id}
    return action_agent.add_edge(args)

def delete_edge(edge_id):
    args = {"edge_id": edge_id}
    return action_agent.delete_edge(args)
"""

class Executor:

    def __init__(self, annotation, tool_retrieve_type = 0, user_intent=None):
        self.llm = Qwen_Model()
        self.project_root = get_project_root()
        self.widgets_path = os.path.join(self.project_root, r"data/ml_platform_data_example/widgets.json")
        with open(self.widgets_path, encoding='utf-8') as  f:
            self.widgets = json.load(f)

        self.action_agent = action_agent

        self.actions_path = os.path.join(self.project_root, r'data/ml_platform_data_example/actions_zh.json')
        with open(self.actions_path, encoding='utf-8') as  f:
            self.actions = json.load(f)
        # 0: 默认，1: 去掉基于类型的检索，2：去掉基于语义相似的检索（HuggingGPT），3：去掉基于LLM的检索，4：仅仅保留基于语义相似度检索
        self.tool_retrieve_type = tool_retrieve_type
        self.tools = self.actions
        self.total_price = 0
        self.total_tokens = 0
        self.annotation = annotation
        self.user_intent = user_intent
        self.input_tokens = 0
        self.output_tokens = 0
        self.time_cost = {
            "widget_retrieval": 0,
            "operation_sequence_generation": 0
        }

    def run(self, task, case_number, retriever):
        return self.act(task, case_number, retriever)

    def act(self, task, case_number, retriever, max_try=3):
        result = {
            "sub_task": task["description"],
            "widgets": [],
            "codes": ""
        }
        from prompts.executor_prompts import SYSTEM_PROMPT_default

        logger.info(f"current task: {task}")

        workflow_info = json.dumps(self.action_agent.get_workflow(), ensure_ascii=False)
        logger.info(f"current workflow: {workflow_info}")
        begin_time = time.time()
        relevant_widgets = self.get_relevant_widgets(task, retriever)
        end_time = time.time()
        self.time_cost["widget_retrieval"] += (end_time - begin_time)

        result["widgets"] = relevant_widgets
        if relevant_widgets == [] and self.user_intent != "Modify":
            return result, True

        self.action_agent.relevant_widgets_names = [w["widget_name"]for w in relevant_widgets]
        if case_number > 0:
            cases = self.retrieve_case(task=task["description"], case_number=case_number, retriever=retriever)
            system_prompt = SYSTEM_PROMPT_default.format(
                widget_list=json.dumps(relevant_widgets, ensure_ascii=False),
                widget_names=json.dumps(self.action_agent.relevant_widgets_names, ensure_ascii=False),
                cases=cases
            )
        else:
            # 特殊组件使用的注意事项
            notes = {
                "SQL Table": "确保SQL Table的参数不为空，否则无法读取表格数据，并且***data_description要使用英文，尽可能描述清楚可能的需要加载的数据集名称（英文名称）和可能的列名（英文列名）***",
                "One Hot Encoder": "One Hot Encoder组件只能有一个输入。因此，如果需要同时处理训练集和测试集时，应该添加两个独热编码组件分别处理。",
                "Test Score": "Test Score组件需要两个输入，一个输入为数据，一个是待评估的模型。",
                "Predictions": "Predictions组件需要两个输入，一个输入为待预测的数据，一个是用于预测的模型。***特别注意：在执行表格分类任务时，**Predictions组件的output_probabilities参数应为True**。***"
            }
            widget_list_string = json.dumps(relevant_widgets, ensure_ascii=False)
            # 添加在可用组件列表之后，对特殊组件的使用做出特殊强调
            for note in notes:
                if note in self.action_agent.relevant_widgets_names:
                    widget_list_string += f"\n## 使用组件{note}时的注意事项\n{notes[note]}"

            system_prompt = SYSTEM_PROMPT_default.format(
                widget_list=widget_list_string,
                widget_names=json.dumps(self.action_agent.relevant_widgets_names, ensure_ascii=False),
                cases=""
            )
            # 不给演示案例，就需要增加一些约束说明操作的注意事项
            cases = [
                "5.模型训练无特殊要求，请使用默认参数，否则将会带来严重后果",
                "***6.使用add_node函数时，一定要定义node_id来接收返回的节点id，否则你无法对添加的节点进行操作***",
            ]
            str_case = ""
            for case in cases:
                str_case += case + "\n"
            i = system_prompt.rfind("\n# 演示案例")
            system_prompt = system_prompt[:i] + str_case + system_prompt[i:]
        data_path = os.path.join(get_project_root(), r"data/data_info.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            data_info = json.load(f)
        is_timeseries_model = False
        if task["type"] == "model":
            for rw in relevant_widgets:
                for w in self.widgets:
                    if w["widget_name"] == rw["widget_name"] and w["package"] == "timeseries":
                        is_timeseries_model = True
        if is_timeseries_model:
            user_prompt = f"我的总目标为：{data_info['instruction']}，现在需要你帮我完成的任务为：{task['description']}" + "\n工作流状态\n" + workflow_info + "\n数据集信息\n" + data_info["dataset_info"]
        else:
            user_prompt = task["description"] + "\n工作流状态\n" + workflow_info + "\n数据集信息\n" + data_info["dataset_info"]

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        retry_time = 0
        begin_time = time.time()
        while retry_time < max_try:
            retry_time += 1
            content, message, input_tokens, output_tokens, price = self.llm.predict(messages=messages)
            self.total_tokens += (input_tokens + output_tokens)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_price += price

            if self.annotation:
                # 用于存储输入的文本
                content = ""

                # 提示用户开始输入
                print("输入code，以 '*' 结束：")

                # 循环读取输入，直到用户输入'*'
                while True:
                    line = input()
                    if line == '*':
                        break
                    content += line + '\n'  # 添加换行符以保持文本的格式

            messages.append({
                "role": "assistant",
                "content": content
            })

            if self.annotation:
                # 存储会话信息
                executor_path = os.path.join(get_project_root(), r"data/cases_library/executor.json")
                with open(executor_path, encoding="utf-8") as f:
                    executor_data = json.load(f)
                executor_data.append(messages)
                with open(executor_path, "w", encoding="utf-8") as f:
                    json.dump(executor_data, f, ensure_ascii=False, indent=4)

            is_succ, content = load_python(content)
            result["codes"] = content
            if is_succ:
                code = f"{CODE_INI}\naction_agent.relevant_widgets_names = {self.action_agent.relevant_widgets_names}\n" + content
                try:
                    exec(code)
                    end_time = time.time()
                    self.time_cost["operation_sequence_generation"] += (end_time - begin_time)
                    return result, True
                except Exception as e:
                    workflow_info = json.dumps(self.action_agent.get_workflow(), ensure_ascii=False)
                    messages.append({
                        "role": "user",
                        "content": f"给出的函数调用执行出错了：{e}。***工作流状态已更新为：{workflow_info}***\n你要先反思犯错的原因，避免再犯同样的错。最后请*****基于新的工作流******重新给出正确的函数调用代码！"
                    })
            else:
                messages.append({
                    "role": "user",
                    "content": f"给出的函数调用代码格式错误：{content}。你是输出应该有且只有一个Python代码块：```python\n生成的函数调用代码\n```"
                })

            print(f"Retry {retry_time} times...")
        end_time = time.time()
        self.time_cost["operation_sequence_generation"] += (end_time - begin_time)
        return result, False

    def get_relevant_widgets(self, task, retriever, topk=5):
        """
        1. Recall: 召回相关组件的方法：基于任务类型和BM25进行召回top-k
        2. Rank: 使用 LLM 对召回的组件进行选择
        """
        relevant_widgets = []
        data_path = os.path.join(get_project_root(), r"data/data_info.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            data_info = json.load(f)
        if self.user_intent != "Modify" and data_info["dataset_info"] == "" and task["type"] in ["preprocess","feature engineering"]:
            return []
        ############################################### 1.根据任务类型对组件进行初筛 ###############################################
        widgets = []
        if self.tool_retrieve_type not in [0, 2, 3]:
            logger.info("不进行任务类型检索")
        for widget in self.widgets:
            if self.tool_retrieve_type in [0, 2, 3]:
                if widget["type"] == task["type"]:
                    widgets.append({
                        "widget_name": widget["widget_name"],
                        "package": widget["package"],
                        "image": widget["image"],
                        "description": widget["description"],
                        "params": widget["params"]
                    })
            else:
                widgets.append({
                    "widget_name": widget["widget_name"],
                    "package": widget["package"],
                    "image": widget["image"],
                    "description": widget["description"],
                    "params": widget["params"]
                })

        widget_names = [widget["widget_name"] for widget in widgets]

        if task["type"] == "io":
            file_index = widget_names.index("File")
            if task["description"].rfind("示例数据") != -1 or task["description"].rfind("example dataset") != -1:
                relevant_widgets = [{
                    "widget_name": widgets[file_index]['widget_name'],
                    "description": widgets[file_index]['description'],
                    "params": widgets[file_index]["params"]
                }]
                logger.info(f"最终检索到的相关组件为：['File']")
                return relevant_widgets
            else:
                widgets.pop(file_index)
                widget_names.pop(file_index)

        ############################################### 2.语义相似度 Recall ###############################################
        if len(widgets) > topk and self.tool_retrieve_type in [0, 1, 3, 4]:
            from embedding_models.embedding_model import Embedding_Model
            corpus = [widget["widget_name"] + widget["description"] for widget in widgets]
            model = Embedding_Model(retriever)
            pairs_sorted = model.get_scores([task["description"]], corpus, topk)
            recalled_widgets = [widgets[index] for s, index in pairs_sorted]
        else:
            logger.info("不进行语义相似度检索")
            recalled_widgets = widgets

        relevant_widgets_names = []
        widgets_info = ""
        for recalled_widget in recalled_widgets:
            relevant_widgets_names.append(recalled_widget['widget_name'])
            widgets_info += "组件名：" + recalled_widget['widget_name'] + "，组件描述: " + recalled_widget['description'] + "\n"
            relevant_widgets.append({
                "widget_name": recalled_widget['widget_name'],
                "description": recalled_widget['description']
            })
        logger.info(f"召回的组件为：{relevant_widgets_names}")

        ############################################### 3.LLM Rank ###############################################
        if self.tool_retrieve_type in [0, 1, 2]:
            if self.user_intent != "Modify":
                experiences = Experiences[task["type"]]
            else:
                experiences = Experiences["Modify"]

            user_prompt = task["description"]
            if self.user_intent != "Modify" and data_info["dataset_info"] != "" and task["type"] in ["preprocess",
                                                                                                     "feature engineering",
                                                                                                     "model",
                                                                                                     "predict"]:
                user_prompt += "\n我的数据集信息如下：\n" + data_info["dataset_info"]

            system_prompt = TOOLS_PROMPT.format(
                widgets_info=widgets_info,
                experiences=experiences
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            max_try = 3
            retry_time = 0
            while retry_time < max_try:
                content, message, input_tokens, output_tokens, price = self.llm.predict(messages)
                self.total_tokens += (input_tokens + output_tokens)
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens
                self.total_price += price

                content = load_json(content)
                feedback = ""
                # 检查是否返回了JSON
                if content != {}:
                    try:
                        # 检查选择的组件列表是否为空
                        if len(content["widget_names"]) == 0:
                            return []
                        else:
                            # 若选择了组件，检查选择的组件是否合法
                            tag = True
                            for wn in content["widget_names"]:
                                if wn not in widget_names:
                                    tag = False
                                    feedback = f"请重新选择组件，组件{wn}不存在！你应该从可用组件{relevant_widgets_names}中选择。"
                                    break
                            if tag:
                                break
                    except:
                        feedback = "JSON不符合要求，请认真参考输出要求！"
                else:
                    feedback = "JSON解析错误！请返回严格的JSON格式，确保能够被python的json.loads()函数解析。"
                retry_time += 1
                logger.info(f"第{retry_time}次尝试")
                messages.append(message)
                messages.append({
                    "role": "user",
                    "content": feedback
                })
        else:
            logger.info("不进行LLM检索")
            content = {}
            content["widget_names"] = relevant_widgets_names
        relevant_widgets = []
        relevant_widget_names = content["widget_names"]
        for relevant_widget_name in relevant_widget_names:
            if relevant_widget_name in widget_names:
                relevant_widget_index = widget_names.index(relevant_widget_name)
                relevant_widgets.append({
                    "widget_name": widgets[relevant_widget_index]['widget_name'],
                    "description": widgets[relevant_widget_index]['description'],
                    "params": widgets[relevant_widget_index]["params"]
                })
        logger.info(f"最终检索到的相关组件为：{relevant_widget_names}")
        return relevant_widgets

    def retrieve_case(self, task, case_number, retriever):

        import json
        from embedding_models.embedding_model import Embedding_Model
        embedding_model = Embedding_Model(retriever)
        case_file = os.path.join(get_project_root(), r"data/cases_library/executor_cases.json")
        with open(case_file, "r", encoding="utf-8") as f:
            executor_cases = json.load(f)
        corpus = []
        for key in executor_cases:
            corpus.append(executor_cases[key]["task"])
        top_cases = embedding_model.get_scores([task], corpus, case_number)

        result = ""
        logger.info(f"检索到的top{case_number}演示案例：")
        for i in range(len(top_cases)):
            top_case = top_cases[i]
            logger.info(f"相似度：{top_case[0]:.4f}，case：{executor_cases[str(top_case[1]+1)]['task']}")
            result += f"## 示例{i + 1}\n"
            result += "### User\n" + executor_cases[str(top_case[1]+1)]["User"] + "\n### Assistant\n" + executor_cases[str(top_case[1]+1)]["Assistant"] + "\n"

        return result
