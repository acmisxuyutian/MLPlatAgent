# -*- coding: utf-8 -*-
import ast
import json
import os
import time

from utils.logs import logger
from llm.llm import Qwen_Model
from utils.utils import (
    get_data_info_path,
    get_project_root,
    load_json,
    load_python,
)
from ml_platform.actions import action_agent
from prompts.executor_prompts import TOOLS_PROMPT, Experiences

# Retained for compatibility with code and tests that inspect the five
# operation API.  Execution itself uses ``_build_execution_scope`` below so
# successful calls can be recorded without changing the LLM-facing functions.
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


WORKFLOW_OPERATION_NAMES = {
    "add_node",
    "delete_node",
    "update_node_params",
    "add_edge",
    "delete_edge",
}


class WorkflowCodeValidationError(ValueError):
    """Raised when generated code escapes the five-operation API."""


def _validate_workflow_value(node, defined_variables):
    if isinstance(node, ast.Constant):
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        if node.id not in defined_variables:
            raise WorkflowCodeValidationError(
                f"未定义的安全变量：{node.id}"
            )
        return
    if isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            _validate_workflow_value(element, defined_variables)
        return
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise WorkflowCodeValidationError("不允许字典解包")
            _validate_workflow_value(key, defined_variables)
            _validate_workflow_value(value, defined_variables)
        return
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return
    raise WorkflowCodeValidationError(
        "工作流操作参数只能使用字面量或本段代码定义的安全变量"
    )


def _validate_workflow_call(call, defined_variables):
    if not isinstance(call.func, ast.Name):
        raise WorkflowCodeValidationError("不允许属性调用或动态函数调用")
    if call.func.id not in WORKFLOW_OPERATION_NAMES:
        raise WorkflowCodeValidationError(
            f"不允许调用函数：{call.func.id}"
        )
    for argument in call.args:
        _validate_workflow_value(argument, defined_variables)
    for keyword in call.keywords:
        if keyword.arg is None:
            raise WorkflowCodeValidationError("不允许关键字参数解包")
        _validate_workflow_value(keyword.value, defined_variables)


def validate_workflow_code(code):
    """Allow only direct calls from the documented workflow operation space."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise WorkflowCodeValidationError(
            f"Python 语法错误：{exc.msg}"
        ) from exc

    defined_variables = set()
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value,
            ast.Call,
        ):
            _validate_workflow_call(
                statement.value,
                defined_variables,
            )
            continue
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                raise WorkflowCodeValidationError(
                    "仅允许对单个安全变量赋值"
                )
            target_name = statement.targets[0].id
            if (
                target_name in defined_variables
                or target_name in WORKFLOW_OPERATION_NAMES
                or target_name == "action_agent"
                or target_name.startswith("__")
            ):
                raise WorkflowCodeValidationError(
                    f"不允许定义或重复赋值变量：{target_name}"
                )
            if isinstance(statement.value, ast.Call):
                _validate_workflow_call(
                    statement.value,
                    defined_variables,
                )
                if (
                    not isinstance(statement.value.func, ast.Name)
                    or statement.value.func.id != "add_node"
                ):
                    raise WorkflowCodeValidationError(
                        "仅允许使用变量接收 add_node 的返回值"
                    )
            else:
                _validate_workflow_value(
                    statement.value,
                    defined_variables,
                )
            defined_variables.add(target_name)
            continue
        raise WorkflowCodeValidationError(
            "每条语句必须是工作流操作调用、add_node 返回值赋值，"
            "或工作流操作参数的字面量赋值"
        )
    return True


class _WorkflowEditRecorder:
    """Record successful calls from the five-operation workflow API.

    Runtime node IDs are platform-assigned and therefore cannot be compared
    directly across experiment runs.  Nodes created during the current Agent
    run receive stable local aliases; IDs from the input workflow remain
    opaque literals and are anonymized by the evaluation function.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._operations = []
        self._created_node_aliases = {}

    def get_operations(self):
        return list(self._operations)

    def _node_reference(self, node_id):
        alias = self._created_node_aliases.get(str(node_id))
        return alias if alias is not None else repr(node_id)

    @staticmethod
    def _normalized_params(node_params):
        if isinstance(node_params, str):
            try:
                decoded = json.loads(node_params)
            except json.JSONDecodeError:
                return node_params
            if isinstance(decoded, dict):
                return decoded
        return node_params

    def record_add_node(self, node_id, widget_name, node_name):
        alias = f"_added_node_{len(self._created_node_aliases) + 1}"
        self._created_node_aliases[str(node_id)] = alias
        self._operations.append(
            f"{alias} = add_node({widget_name!r}, {node_name!r})"
        )

    def record_delete_node(self, node_id):
        self._operations.append(
            f"delete_node({self._node_reference(node_id)})"
        )

    def record_update_node_params(
        self,
        node_id,
        widget_name,
        node_params,
    ):
        normalized_params = self._normalized_params(node_params)
        self._operations.append(
            "update_node_params("
            f"{self._node_reference(node_id)}, "
            f"{widget_name!r}, "
            f"{normalized_params!r})"
        )

    def record_add_edge(self, source_node_id, target_node_id):
        self._operations.append(
            "add_edge("
            f"{self._node_reference(source_node_id)}, "
            f"{self._node_reference(target_node_id)})"
        )

    def record_delete_edge(self, edge_id):
        self._operations.append(f"delete_edge({edge_id!r})")


def _build_execution_scope(current_action_agent, recorder):
    """Build the runtime API exposed to LLM-generated workflow code."""

    def add_node(widget_name, node_name):
        args = {"widget_name": widget_name, "node_name": node_name}
        node_id = current_action_agent.add_node(args)
        recorder.record_add_node(node_id, widget_name, node_name)
        return node_id

    def delete_node(node_id):
        args = {"node_id": node_id}
        result = current_action_agent.delete_node(args)
        recorder.record_delete_node(node_id)
        return result

    def update_node_params(node_id, widget_name, node_params):
        args = {
            "node_id": node_id,
            "widget_name": widget_name,
            "node_params": node_params,
        }
        result = current_action_agent.update_node_params(args)
        recorder.record_update_node_params(
            node_id,
            widget_name,
            node_params,
        )
        return result

    def add_edge(source_node_id, target_node_id):
        args = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        result = current_action_agent.add_edge(args)
        recorder.record_add_edge(source_node_id, target_node_id)
        return result

    def delete_edge(edge_id):
        args = {"edge_id": edge_id}
        result = current_action_agent.delete_edge(args)
        recorder.record_delete_edge(edge_id)
        return result

    return {
        "__name__": "__workflow_operations__",
        "action_agent": current_action_agent,
        "add_node": add_node,
        "delete_node": delete_node,
        "update_node_params": update_node_params,
        "add_edge": add_edge,
        "delete_edge": delete_edge,
    }


class Executor:

    def __init__(self, annotation, tool_retrieve_type = 0, user_intent=None):
        self.llm = Qwen_Model()
        self.project_root = get_project_root()
        self.action_agent = action_agent
        self.widgets_path = self.action_agent.widgets_path
        self.widgets = self.action_agent.widgets

        self.actions_path = os.path.join(self.project_root, r'data/actions_zh.json')
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
        self.widget_retrieval_status = None
        self._workflow_edit_recorder = _WorkflowEditRecorder()

    def reset_workflow_edit_sets(self):
        self._workflow_edit_recorder.reset()

    def get_workflow_edit_sets(self):
        return self._workflow_edit_recorder.get_operations()

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
        if (
            relevant_widgets == []
            and self.user_intent != "Modify"
            and self.widget_retrieval_status == "no_component_needed"
            and task["type"] in {"preprocess", "feature engineering"}
        ):
            result["skipped"] = True
            result["reason"] = "当前数据状态不需要执行该可选处理步骤"
            return result, True
        if relevant_widgets == [] and self.user_intent != "Modify":
            result["error"] = (
                f"当前平台 {self.action_agent.platform_name} "
                f"没有找到可完成任务“{task['description']}”的组件。"
            )
            return result, False

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
        data_path = get_data_info_path()
        with open(data_path, 'r', encoding='utf-8') as f:
            data_info = json.load(f)
        is_timeseries_model = False
        if task["type"] == "model":
            for rw in relevant_widgets:
                for w in self.widgets:
                    if (
                        w["widget_name"] == rw["widget_name"]
                        and w.get("package") == "timeseries"
                    ):
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
                try:
                    validate_workflow_code(content)
                except WorkflowCodeValidationError as exc:
                    result["error"] = (
                        "生成代码超出允许的工作流操作空间："
                        f"{exc}"
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "给出的代码超出了允许的五种工作流编辑操作："
                            f"{exc}。请只调用 add_node、delete_node、"
                            "update_node_params、add_edge、delete_edge，"
                            "不要生成或执行任意 Python 代码。"
                        ),
                    })
                    continue
                try:
                    execution_scope = _build_execution_scope(
                        self.action_agent,
                        self._workflow_edit_recorder,
                    )
                    exec(content, execution_scope)
                    end_time = time.time()
                    self.time_cost["operation_sequence_generation"] += (end_time - begin_time)
                    result.pop("error", None)
                    return result, True
                except Exception as e:
                    result["error"] = f"工作流操作执行失败：{e}"
                    workflow_info = json.dumps(self.action_agent.get_workflow(), ensure_ascii=False)
                    messages.append({
                        "role": "user",
                        "content": f"给出的函数调用执行出错了：{e}。***工作流状态已更新为：{workflow_info}***\n你要先反思犯错的原因，避免再犯同样的错。最后请*****基于新的工作流******重新给出正确的函数调用代码！"
                    })
            else:
                result["error"] = f"函数调用代码格式错误：{content}"
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
        self.widget_retrieval_status = "searching"
        relevant_widgets = []
        data_path = get_data_info_path()
        with open(data_path, 'r', encoding='utf-8') as f:
            data_info = json.load(f)
        if self.user_intent != "Modify" and data_info["dataset_info"] == "" and task["type"] in ["preprocess","feature engineering"]:
            self.widget_retrieval_status = "missing_dataset_context"
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
                        "package": widget.get("package", widget["type"]),
                        "image": widget.get("image", ""),
                        "description": widget["description"],
                        "params": widget["params"]
                    })
            else:
                widgets.append({
                    "widget_name": widget["widget_name"],
                    "package": widget.get("package", widget["type"]),
                    "image": widget.get("image", ""),
                    "description": widget["description"],
                    "params": widget["params"]
                })

        # Some platform catalogs use their own coarse category names. When the
        # active catalog has no exact task-type match, keep retrieval platform
        # agnostic by searching the complete catalog instead of hard-coding a
        # component mapping in the Agent core.
        if not widgets and self.tool_retrieve_type in [0, 2, 3]:
            widgets = [
                {
                    "widget_name": widget["widget_name"],
                    "package": widget.get("package", widget["type"]),
                    "image": widget.get("image", ""),
                    "description": widget["description"],
                    "params": widget["params"]
                }
                for widget in self.widgets
            ]

        widget_names = [widget["widget_name"] for widget in widgets]

        if task["type"] == "io" and "File" in widget_names:
            file_index = widget_names.index("File")
            if task["description"].rfind("示例数据") != -1 or task["description"].rfind("example dataset") != -1:
                relevant_widgets = [{
                    "widget_name": widgets[file_index]['widget_name'],
                    "description": widgets[file_index]['description'],
                    "params": widgets[file_index]["params"]
                }]
                logger.info(f"最终检索到的相关组件为：['File']")
                self.widget_retrieval_status = "selected"
                return relevant_widgets
            else:
                widgets.pop(file_index)
                widget_names.pop(file_index)

        if not widgets:
            self.widget_retrieval_status = "no_platform_widgets"
            return []

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
                            self.widget_retrieval_status = (
                                "no_component_needed"
                            )
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
        if (
            not isinstance(content, dict)
            or not isinstance(content.get("widget_names"), list)
        ):
            self.widget_retrieval_status = "invalid_selection"
            return []

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
        self.widget_retrieval_status = (
            "selected" if relevant_widgets else "invalid_selection"
        )
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
