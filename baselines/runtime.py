"""
Shared runtime adapters for the independently executable baselines.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents.executor import (
    Executor,
    _build_execution_scope,
)
from agents.mlagent import MLAgent
from agents.planner import Planner
from embedding_models.embedding_model import Embedding_Model
from prompts.executor_prompts import (
    DIVERSITY_PROMPT_dfsdt,
    SYSTEM_PROMPT_default,
    SYSTEM_PROMPT_dfsdt,
    SYSTEM_PROMPT_function_call,
    SYSTEM_PROMPT_react,
)
from utils.logs import logger
from utils.utils import (
    get_project_root,
    load_json,
    load_python,
    update_data_info,
)


DEFAULT_REQUIREMENT = (
    "你的任务是根据银行客户的各种信息，如账户活动、服务使用情况等，"
    "构建一个能够预测客户是否会流失的机器学习模型。"
)

SPECIAL_WIDGET_NOTES = {
    "SQL Table": (
        "确保SQL Table的参数不为空，否则无法读取表格数据，并且"
        "***data_description要使用英文，尽可能描述清楚可能的需要加载的"
        "数据集名称（英文名称）和可能的列名（英文列名）***"
    ),
    "One Hot Encoder": (
        "One Hot Encoder组件只能有一个输入。因此，如果需要同时处理训练集"
        "和测试集时，应该添加两个独热编码组件分别处理。"
    ),
    "Test Score": (
        "Test Score组件需要两个输入，一个输入为数据，一个是待评估的模型。"
    ),
    "Predictions": (
        "Predictions组件需要两个输入，一个输入为待预测的数据，一个是用于"
        "预测的模型。***特别注意：在执行表格分类任务时，"
        "**Predictions组件的output_probabilities参数应为True**。***"
    ),
}

DIRECT_SYSTEM_PROMPT = """# 角色
你是一个函数调用代码生成器

# 任务
你的任务是生成一段能够实现用户需求的函数调用代码。这段函数调用代码表示了对当前机器学习工作流进行的操作序列。

# 函数库
1.node_id=add_node(widget_name,node_name)
该函数可以从当前任务可用组件中选择一个组件添加到当前的机器学习工作流中，添加成功将会返回一个节点id。
使用add_node函数的注意事项：node_name需要使用中文名称。仅能使用当前任务可用组件中存在的组件。add_node函数无法设置参数。
2.delete_node(node_id)
该函数可以删除当前机器学习工作流中的指定节点。
3.update_node_params(node_id, widget_name, node_params)
该函数可以更新当前机器学习工作流中的指定节点的参数信息。仅更新用户要求更新的参数，不要更新过多的参数。
4.add_edge(source_node_id, target_node_id)
该函数可以添加一条边到当前机器学习工作流中。
5.delete_edge(edge_id)
该函数可以删除当前机器学习工作流中的指定边。

# 当前任务可用组件
{widget_list}

# 约束
你需要遵循以下约束：
1.在使用add_node函数时，仅能添加以下组件：{widget_names}
2.仅能使用函数库中存在的函数：["add_node","delete_node","update_node_params","add_edge","delete_edge"]
3.你的每行代码必须是在调用函数，不得编写其他任何代码。
4.仅能通过update_node_params函数来设置节点参数，无特殊要求均使用默认参数。
5.如果用户没有明确要求进行评估、预测或可视化，请不要添加这些节点。

# 输出要求
你是输出应该有且只有一个Python代码块：
```python
生成的函数调用代码
```
不要有任何多余的信息！
"""


def _new_result(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "sub_task": task["description"],
        "widgets": [],
        "codes": "",
    }


def _append_widget_notes(
    prompt_or_widgets: str,
    widget_names: list[str],
) -> str:
    result = prompt_or_widgets
    for widget_name, note in SPECIAL_WIDGET_NOTES.items():
        if widget_name in widget_names:
            result += f"\n## 使用组件{widget_name}时的注意事项\n{note}"
    return result


def _load_data_info() -> dict[str, Any]:
    path = Path(get_project_root()) / "data" / "data_info.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _is_timeseries_task(
    executor: Executor,
    task: dict[str, Any],
    relevant_widgets: list[dict[str, Any]],
) -> bool:
    if task["type"] != "model":
        return False
    selected = {widget["widget_name"] for widget in relevant_widgets}
    return any(
        widget["widget_name"] in selected
        and widget.get("package") == "timeseries"
        for widget in executor.widgets
    )


def _task_user_prompt(
    executor: Executor,
    task: dict[str, Any],
    relevant_widgets: list[dict[str, Any]],
    workflow_info: str,
    *,
    react: bool = False,
) -> str:
    data_info = _load_data_info()
    if _is_timeseries_task(executor, task, relevant_widgets):
        return (
            f"我的总目标为：{data_info['instruction']}，现在需要你帮我完成的任务为："
            f"{task['description']}\n工作流状态\n{workflow_info}\n数据集信息\n"
            f"{data_info['dataset_info']}"
        )
    prefix = "我的目标为：" if react else ""
    return (
        f"{prefix}{task['description']}\n工作流状态\n{workflow_info}"
        f"\n数据集信息\n{data_info['dataset_info']}"
    )


def _prepare_current_retrieval(
    executor: Executor,
    task: dict[str, Any],
    retriever: str,
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], bool] | None]:
    """Use exactly the current MLPlatAgent tool-retrieval implementation."""
    begin_time = time.time()
    relevant_widgets = executor.get_relevant_widgets(task, retriever)
    executor.time_cost["widget_retrieval"] += time.time() - begin_time
    result["widgets"] = relevant_widgets

    if (
        not relevant_widgets
        and executor.user_intent != "Modify"
        and executor.widget_retrieval_status == "no_component_needed"
        and task["type"] in {"preprocess", "feature engineering"}
    ):
        result["skipped"] = True
        result["reason"] = "当前数据状态不需要执行该可选处理步骤"
        return relevant_widgets, (result, True)

    if not relevant_widgets and executor.user_intent != "Modify":
        result["error"] = (
            f"当前平台 {executor.action_agent.platform_name} "
            f"没有找到可完成任务“{task['description']}”的组件。"
        )
        return relevant_widgets, (result, False)

    executor.action_agent.relevant_widgets_names = [
        widget["widget_name"] for widget in relevant_widgets
    ]
    return relevant_widgets, None


def _execute_recorded_command(
    executor: Executor,
    action: str,
    action_input: Any,
) -> Any:
    """Mirror ``Action.execute_command`` and record only successful edits."""
    if action not in {
        "add_node",
        "delete_node",
        "update_node_params",
        "add_edge",
        "delete_edge",
    }:
        return executor.action_agent.execute_command(action, action_input)
    if not isinstance(action_input, dict):
        return f"Error: action {action!r} requires an argument mapping"

    scope = _build_execution_scope(
        executor.action_agent,
        executor._workflow_edit_recorder,
    )
    try:
        return scope[action](**action_input)
    except Exception as exc:
        return f"Error: {exc}"


def _message_content(message: Any) -> Any:
    try:
        return message["content"]
    except (KeyError, TypeError):
        return getattr(message, "content", None)


class PlannerBackedAgent(MLAgent):
    """MLAgent with a replaceable Executor and an unchanged Planner."""

    executor_class: type[Executor] = Executor

    def __init__(
        self,
        tool_retrieve_type: int = 0,
        annotation: bool = False,
    ) -> None:
        self.planner = Planner(annotation)
        self.executor = self.executor_class(
            annotation=annotation,
            tool_retrieve_type=tool_retrieve_type,
        )


class AllWidgetsExecutor(Executor):
    """Executor used by the historical ``w/o tool retrieval`` ablation."""

    def get_relevant_widgets(
        self,
        task: dict[str, Any],
        retriever: str,
        topk: int = 5,
    ) -> list[dict[str, Any]]:
        del task, retriever, topk
        relevant_widgets = [
            {
                "widget_name": widget["widget_name"],
                "description": widget["description"],
                "params": widget["params"],
            }
            for widget in self.widgets
        ]
        self.widget_retrieval_status = (
            "selected" if relevant_widgets else "no_platform_widgets"
        )
        return relevant_widgets

    def act(
        self,
        task: dict[str, Any],
        case_number: int,
        retriever: str,
        max_try: int = 3,
    ) -> tuple[dict[str, Any], bool]:
        """Run FCC while exposing all widgets, matching ``wo_WR.py``."""
        logger.info(f"current task: {task}")
        result = _new_result(task)
        workflow_info = json.dumps(
            self.action_agent.get_workflow(),
            ensure_ascii=False,
        )
        logger.info(f"current workflow: {workflow_info}")

        begin_retrieval = time.time()
        relevant_widgets = self.get_relevant_widgets(task, retriever)
        self.time_cost[
            "widget_retrieval"
        ] += time.time() - begin_retrieval
        result["widgets"] = relevant_widgets
        if not relevant_widgets:
            result["error"] = (
                f"当前平台 {self.action_agent.platform_name} 没有可用组件。"
            )
            return result, False

        self.action_agent.relevant_widgets_names = [
            widget["widget_name"] for widget in relevant_widgets
        ]
        widget_list = _append_widget_notes(
            json.dumps(relevant_widgets, ensure_ascii=False),
            self.action_agent.relevant_widgets_names,
        )
        cases = (
            self.retrieve_case(
                task=task["description"],
                case_number=case_number,
                retriever=retriever,
            )
            if case_number > 0
            else ""
        )
        system_prompt = SYSTEM_PROMPT_default.format(
            widget_list=widget_list,
            widget_names=json.dumps(
                self.action_agent.relevant_widgets_names,
                ensure_ascii=False,
            ),
            cases=cases,
        )
        user_prompt = _task_user_prompt(
            self,
            task,
            relevant_widgets,
            workflow_info,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        retry_time = 0
        begin_operation = time.time()
        while retry_time < max_try:
            retry_time += 1
            content, message, input_tokens, output_tokens, price = (
                self.llm.predict(messages=messages)
            )
            self.total_tokens += input_tokens + output_tokens
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_price += price
            messages.append(
                {"role": "assistant", "content": content}
            )

            is_succ, parsed_code = load_python(content)
            result["codes"] = parsed_code
            if is_succ:
                try:
                    exec(
                        parsed_code,
                        _build_execution_scope(
                            self.action_agent,
                            self._workflow_edit_recorder,
                        ),
                    )
                    self.time_cost[
                        "operation_sequence_generation"
                    ] += time.time() - begin_operation
                    return result, True
                except Exception as exc:
                    workflow_info = json.dumps(
                        self.action_agent.get_workflow(),
                        ensure_ascii=False,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"给出的函数调用执行出错了：{exc}。"
                                "***工作流状态已更新为："
                                f"{workflow_info}***\n你要先反思犯错的原因，"
                                "避免再犯同样的错。最后请*****基于新的工作流"
                                "******重新给出正确的函数调用代码！"
                            ),
                        }
                    )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"给出的函数调用代码格式错误：{parsed_code}。"
                            "你是输出应该有且只有一个Python代码块："
                            "```python\n生成的函数调用代码\n```"
                        ),
                    }
                )
            print(f"Retry {retry_time} times...")

        self.time_cost[
            "operation_sequence_generation"
        ] += time.time() - begin_operation
        return result, False


class ReActExecutor(Executor):
    """Historical ReAct construction loop with current Planner/retrieval."""

    def run(
        self,
        task: dict[str, Any],
        case_number: int,
        retriever: str,
    ) -> tuple[dict[str, Any], bool]:
        del case_number
        return self.react(task, retriever)

    def react(
        self,
        task: dict[str, Any],
        retriever: str,
        max_try: int = 10,
    ) -> tuple[dict[str, Any], bool]:
        logger.info(f"current task: {task}")
        result = _new_result(task)
        workflow_info = json.dumps(
            self.action_agent.get_workflow(),
            ensure_ascii=False,
        )
        logger.info(f"current workflow: {workflow_info}")

        relevant_widgets, early_result = _prepare_current_retrieval(
            self,
            task,
            retriever,
            result,
        )
        if early_result is not None:
            return early_result

        widget_names = self.action_agent.relevant_widgets_names
        widget_list = _append_widget_notes(
            json.dumps(relevant_widgets, ensure_ascii=False),
            widget_names,
        )
        system_prompt = SYSTEM_PROMPT_react.format(
            widget_list=widget_list,
            widget_names=json.dumps(widget_names, ensure_ascii=False),
        )
        user_prompt = _task_user_prompt(
            self,
            task,
            relevant_widgets,
            workflow_info,
            react=True,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    user_prompt
                    + "\n现在，请以严格的JSON格式，给出下一步要采取的行动！"
                ),
            },
        ]

        retry_time = 0
        begin_time = time.time()
        while retry_time < max_try:
            retry_time += 1
            content, message, input_tokens, output_tokens, price = (
                self.llm.predict(messages=messages)
            )
            self.total_tokens += input_tokens + output_tokens
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_price += price

            parsed_content = load_json(content)
            if parsed_content != {}:
                action = parsed_content["action"]["name"]
                action_input = parsed_content["action"]["args"]
                result["codes"] = json.dumps(
                    parsed_content,
                    ensure_ascii=False,
                )
                if action == "end_task":
                    self.time_cost[
                        "operation_sequence_generation"
                    ] += time.time() - begin_time
                    return result, True
                action_result = _execute_recorded_command(
                    self,
                    action,
                    action_input,
                )
                feedback = (
                    f"{action}的执行结果为：{action_result}。\n工作流状态已更新为："
                    + json.dumps(
                        self.action_agent.get_workflow(),
                        ensure_ascii=False,
                    )
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            parsed_content,
                            ensure_ascii=False,
                        ),
                    }
                )
            else:
                feedback = (
                    "输出的JSON存在格式错误。请检查是否有多余的符号："
                    "']', '[', '}', '{'"
                )
                messages.append(message)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        feedback
                        + "\n现在，请以严格的JSON格式，给出下一步要采取的行动！"
                    ),
                }
            )

        self.time_cost[
            "operation_sequence_generation"
        ] += time.time() - begin_time
        return result, False


class FunctionCallExecutor(Executor):
    """Historical native Function Call loop with current retrieval."""

    @staticmethod
    def parse_function_input(
        action_input: Any,
    ) -> tuple[Any, bool]:
        try:
            parsed_input = json.loads(action_input)
        except Exception:
            return "你给出的参数无法被json.loads解析，请修正！", False
        if parsed_input is None:
            return "", True
        if "node_params" in parsed_input:
            try:
                parsed_input["node_params"] = json.loads(
                    parsed_input["node_params"]
                )
            except Exception:
                return (
                    '你给出的参数"node_params"无法被json.loads解析，请修正！',
                    False,
                )
        return parsed_input, True

    def run(
        self,
        task: dict[str, Any],
        case_number: int,
        retriever: str,
    ) -> tuple[dict[str, Any], bool]:
        del case_number
        return self.function_call(task, retriever)

    def function_call(
        self,
        task: dict[str, Any],
        retriever: str,
        max_try: int = 10,
    ) -> tuple[dict[str, Any], bool]:
        logger.info(f"current task: {task}")
        result = _new_result(task)
        workflow_info = json.dumps(
            self.action_agent.get_workflow(),
            ensure_ascii=False,
        )
        logger.info(f"current workflow: {workflow_info}")

        relevant_widgets, early_result = _prepare_current_retrieval(
            self,
            task,
            retriever,
            result,
        )
        if early_result is not None:
            return early_result

        user_prompt = _task_user_prompt(
            self,
            task,
            relevant_widgets,
            workflow_info,
        )
        finish_func = {
            "type": "function",
            "function": {
                "name": "end_task",
                "description": (
                    "结束当前任务，如果您完成了当前任务时，请执行该行动"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "任务执行结果",
                        }
                    },
                    "required": ["result"],
                },
            },
        }
        # Keep the original five workflow tools and one terminal tool for each
        # subtask without accumulating duplicate terminal schemas.
        self.tools = list(self.actions) + [finish_func]
        widget_names = self.action_agent.relevant_widgets_names
        system_prompt = SYSTEM_PROMPT_function_call.format(
            widget_list=json.dumps(
                relevant_widgets,
                ensure_ascii=False,
            ),
            widget_names=json.dumps(widget_names, ensure_ascii=False),
        )
        system_prompt = _append_widget_notes(system_prompt, widget_names)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        retry_time = 0
        begin_time = time.time()
        while retry_time < max_try:
            retry_time += 1
            content, message, input_tokens, output_tokens, price = (
                self.llm.predict(messages=messages, tools=self.tools)
            )
            self.total_tokens += input_tokens + output_tokens
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_price += price

            if "tool_calls" in message:
                logger.info(
                    "llm output:\n"
                    + json.dumps(message, indent=4, ensure_ascii=False)
                )
                tool_call = message["tool_calls"][0]
                action = tool_call["function"]["name"]
                action_input, is_json = self.parse_function_input(
                    tool_call["function"]["arguments"]
                )
                if is_json:
                    action_result = _execute_recorded_command(
                        self,
                        action,
                        action_input,
                    )
                else:
                    action_result = action_input

                result["codes"] = json.dumps(
                    {
                        "action": action,
                        "action_input": action_input,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": _message_content(message),
                        "tool_calls": [tool_call],
                    }
                )
                messages.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": action,
                        "content": (
                            f"{action}的执行结果为：{action_result}。"
                            "\n工作流状态已更新为："
                            + json.dumps(
                                self.action_agent.get_workflow(),
                                ensure_ascii=False,
                            )
                            + "\n现在，请进行下一步函数调用！"
                        ),
                    }
                )
                if action == "end_task":
                    self.time_cost[
                        "operation_sequence_generation"
                    ] += time.time() - begin_time
                    return result, True
            else:
                messages.append(message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你上一步回答错误！你应该总是执行函数调用，不应该"
                            "输出文本。如果要结束任务请调用end_task函数。"
                            "\n现在，请进行下一步函数调用！"
                        ),
                    }
                )

        self.time_cost[
            "operation_sequence_generation"
        ] += time.time() - begin_time
        return result, False


class DFSDTExecutor(FunctionCallExecutor):
    """Historical DFSDT tree search with current Planner and retrieval."""

    def run(
        self,
        task: dict[str, Any],
        case_number: int,
        retriever: str,
    ) -> tuple[dict[str, Any], bool]:
        del case_number
        return self.dfsdt(task, retriever)

    def dfsdt(
        self,
        task: dict[str, Any],
        retriever: str,
    ) -> tuple[dict[str, Any], bool]:
        logger.info(f"current task: {task}")
        result = _new_result(task)
        workflow_info = json.dumps(
            self.action_agent.get_workflow(),
            ensure_ascii=False,
        )
        logger.info(f"current workflow: {workflow_info}")

        relevant_widgets, early_result = _prepare_current_retrieval(
            self,
            task,
            retriever,
            result,
        )
        if early_result is not None:
            return early_result

        user_prompt = _task_user_prompt(
            self,
            task,
            relevant_widgets,
            workflow_info,
        )
        widget_names = self.action_agent.relevant_widgets_names
        system_prompt = SYSTEM_PROMPT_dfsdt.format(
            widget_list=json.dumps(
                relevant_widgets,
                ensure_ascii=False,
            ),
            widget_names=json.dumps(widget_names, ensure_ascii=False),
        )
        system_prompt = _append_widget_notes(system_prompt, widget_names)
        finish_func = {
            "type": "function",
            "function": {
                "name": "finish",
                "description": (
                    "如果你认为当前任务已完成，请调用此工具结束当前任务。"
                    "或者，如果你认为在当前状态下已经无法完成任务，请调用"
                    "此工具重新开始尝试。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "return_type": {
                            "type": "string",
                            "enum": [
                                "end_task",
                                "give_up_and_restart",
                            ],
                            "description": (
                                "end_task：结束当前任务，"
                                "give_up_and_restart：重新开始尝试"
                            ),
                        },
                        "result": {
                            "type": "string",
                            "description": "任务执行结果",
                        },
                    },
                    "required": ["return_type", "result"],
                },
            },
        }
        # DFSDT uses the same tool set on every search node.  Rebuild it for
        # each subtask so sequential plans do not accumulate ``finish`` tools.
        self.tools = list(self.actions) + [finish_func]

        root_node = TreeNode()
        root_node.node_type = "root"
        root_node.messages.append(
            {"role": "system", "content": system_prompt}
        )
        root_node.messages.append(
            {"role": "user", "content": user_prompt}
        )
        begin_time = time.time()
        steps_back, is_success = self.DFS(root_node, 0)
        self.time_cost[
            "operation_sequence_generation"
        ] += time.time() - begin_time
        result["codes"] = json.dumps(
            {"steps_back": steps_back},
            ensure_ascii=False,
        )
        return result, is_success

    def DFS(
        self,
        now_node: "TreeNode",
        query_count: int,
        single_chain_max_step: int = 10,
        tree_beam_size: int = 2,
        max_query_count: int = 20,
    ) -> tuple[int, bool]:
        if now_node.finished:
            return 10000, True
        if now_node.pruned:
            return 2, False
        if now_node.get_depth() >= single_chain_max_step:
            return 1, False

        for _ in range(tree_beam_size):
            new_node = TreeNode()
            new_node.father = now_node
            new_node.messages = new_node.father.messages.copy()

            if now_node.children:
                previous_calls = []
                for child in now_node.children:
                    temp_node = child
                    while (
                        temp_node.node_type != "action"
                        and temp_node.children
                    ):
                        temp_node = temp_node.children[0]
                    if temp_node.node_type == "action":
                        previous_calls.append(
                            {
                                "name": temp_node.action_info["action"],
                                "arguments": temp_node.action_info[
                                    "action_input"
                                ],
                                "function_output": temp_node.action_info[
                                    "action_output"
                                ],
                            }
                        )
                if previous_calls:
                    new_node.messages.append(
                        {
                            "role": "user",
                            "content": DIVERSITY_PROMPT_dfsdt.format(
                                previous_calls=json.dumps(
                                    previous_calls,
                                    ensure_ascii=False,
                                )
                            ),
                        }
                    )
                else:
                    new_node.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "这已经不是你第一次尝试这项任务了，之前的尝试"
                                "都失败了。\n现在，你需要复述用户目标以明确要"
                                "完成的任务，然后分析当前任务完成进度，最后"
                                "进行函数调用。"
                            ),
                        }
                    )

            content, message, input_tokens, output_tokens, price = (
                self.llm.predict(
                    messages=new_node.messages,
                    tools=self.tools,
                )
            )
            self.total_tokens += input_tokens + output_tokens
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_price += price
            query_count += 1
            if query_count >= max_query_count:
                return 100000, False
            if self.total_tokens >= 100000:
                return 100000, False

            if "tool_calls" in message:
                new_node.node_type = "action"
                logger.info(
                    "llm output:\n"
                    + json.dumps(message, indent=4, ensure_ascii=False)
                )
                tool_call = message["tool_calls"][0]
                action = tool_call["function"]["name"]
                raw_action_input = tool_call["function"]["arguments"]
                new_node.action_info["action"] = action
                new_node.action_info["action_input"] = raw_action_input
                action_input, is_json = self.parse_function_input(
                    raw_action_input
                )
                action_result: Any = ""
                if is_json:
                    if action == "finish":
                        try:
                            if action_input["return_type"] == "end_task":
                                new_node.finished = True
                            elif (
                                action_input["return_type"]
                                == "give_up_and_restart"
                            ):
                                new_node.pruned = True
                            else:
                                action_result = (
                                    "finish函数的return_type参数值错误，"
                                    "return_type必须为'end_task'或"
                                    "'give_up_and_restart'。"
                                )
                        except Exception:
                            action_result = (
                                "finish函数的参数错误，请检查参数格式以及是否"
                                "包含了必要参数'return_type'!"
                            )
                    else:
                        action_result = _execute_recorded_command(
                            self,
                            action,
                            action_input,
                        )
                else:
                    action_result = action_input

                new_node.action_info["action_output"] = action_result
                new_node.messages.append(
                    {
                        "role": "assistant",
                        "content": _message_content(message),
                        "tool_calls": [tool_call],
                    }
                )
                new_node.messages.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": action,
                        "content": (
                            f"{action}的执行结果为：{action_result}。"
                            "\n工作流状态已更新为："
                            + json.dumps(
                                self.action_agent.get_workflow(),
                                ensure_ascii=False,
                            )
                            + "\n现在，请进行下一步函数调用！"
                        ),
                    }
                )
            else:
                new_node.node_type = "thought"
                new_node.messages.append(message)
                new_node.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你上一步回答错误！你应该总是执行函数调用，不应该"
                            "输出文本。如果要结束任务请调用finish函数。"
                            "\n现在，请进行下一步函数调用！"
                        ),
                    }
                )

            now_node.children.append(new_node)
            steps_back, is_success = self.DFS(new_node, query_count)
            if is_success:
                return 10000, True
            if steps_back > 1:
                return steps_back - 1, False
        return 1, False


class TreeNode:
    """Search node used by the historical DFSDT implementation."""

    def __init__(self) -> None:
        self.node_type: str | None = None
        self.father: TreeNode | None = None
        self.children: list[TreeNode] = []
        self.messages: list[Any] = []
        self.pruned = False
        self.finished = False
        self.action_info = {
            "action": "",
            "action_input": "",
            "action_output": "",
        }

    def get_depth(self) -> int:
        if self.father is None:
            return 0
        return self.father.get_depth() + 1


class DirectWorkflowExecutor(Executor):
    """FCC operation generation without Planner, with optional retrieval."""

    def __init__(
        self,
        *,
        retrieval_mode: str,
        annotation: bool = False,
        catch_llm_errors: bool = False,
    ) -> None:
        super().__init__(
            annotation=annotation,
            tool_retrieve_type=0,
        )
        if retrieval_mode not in {"semantic", "all"}:
            raise ValueError(
                "retrieval_mode must be either 'semantic' or 'all'"
            )
        self.retrieval_mode = retrieval_mode
        self.catch_llm_errors = catch_llm_errors

    def _direct_widgets(
        self,
        instruction: str,
        retriever: str,
        topk: int = 15,
    ) -> list[dict[str, Any]]:
        widgets = [
            {
                "widget_name": widget["widget_name"],
                "package": widget.get("package", widget["type"]),
                "image": widget.get("image", ""),
                "description": widget["description"],
                "params": widget["params"],
            }
            for widget in self.widgets
        ]
        if self.retrieval_mode == "semantic":
            corpus = [
                widget["widget_name"] + widget["description"]
                for widget in widgets
            ]
            model = Embedding_Model(retriever)
            pairs_sorted = model.get_scores(
                [instruction],
                corpus,
                topk,
            )
            widgets = [
                widgets[index] for _, index in pairs_sorted
            ]

        relevant_widgets = [
            {
                "widget_name": widget["widget_name"],
                "description": widget["description"],
                "params": widget["params"],
            }
            for widget in widgets
        ]
        logger.info(
            "最终检索到的相关组件为："
            + str(
                [
                    widget["widget_name"]
                    for widget in relevant_widgets
                ]
            )
        )
        return relevant_widgets

    def run_instruction(
        self,
        instruction: str,
        *,
        case_number: int = 3,
        retriever: str = "multilingual-e5-large",
        max_try: int = 3,
    ) -> tuple[dict[str, Any], bool]:
        task = {"description": instruction, "type": "direct"}
        result = _new_result(task)
        logger.info(f"Instruction: {instruction}")
        workflow_info = json.dumps(
            self.action_agent.get_workflow(),
            ensure_ascii=False,
        )
        logger.info(f"Workflow: {workflow_info}")

        begin_retrieval = time.time()
        relevant_widgets = self._direct_widgets(
            instruction,
            retriever,
        )
        self.time_cost[
            "widget_retrieval"
        ] += time.time() - begin_retrieval
        result["widgets"] = relevant_widgets
        self.action_agent.relevant_widgets_names = [
            widget["widget_name"] for widget in relevant_widgets
        ]

        widget_list = _append_widget_notes(
            json.dumps(relevant_widgets, ensure_ascii=False),
            self.action_agent.relevant_widgets_names,
        )
        # cases = self.retrieve_case(
        #     task=instruction,
        #     case_number=case_number,
        #     retriever=retriever,
        # )
        system_prompt = DIRECT_SYSTEM_PROMPT.format(
            widget_list=widget_list,
            widget_names=json.dumps(
                self.action_agent.relevant_widgets_names,
                ensure_ascii=False,
            )
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "#用户需求：\n"
                    + instruction
                    + "\n#工作流状态：\n"
                    + workflow_info
                ),
            },
        ]

        retry_time = 0
        begin_operation = time.time()
        while retry_time < max_try:
            retry_time += 1
            try:
                content, message, input_tokens, output_tokens, price = (
                    self.llm.predict(
                        messages=messages,
                        max_tokens=None,
                    )
                )
            except Exception as exc:
                if not self.catch_llm_errors:
                    raise
                logger.info(exc)
                break

            self.total_tokens += input_tokens + output_tokens
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_price += price
            messages.append(
                {"role": "assistant", "content": content}
            )
            is_succ, parsed_code = load_python(content)
            result["codes"] = parsed_code
            if is_succ:
                try:
                    exec(
                        parsed_code,
                        _build_execution_scope(
                            self.action_agent,
                            self._workflow_edit_recorder,
                        ),
                    )
                    self.time_cost[
                        "operation_sequence_generation"
                    ] += time.time() - begin_operation
                    return result, True
                except Exception as exc:
                    workflow_info = json.dumps(
                        self.action_agent.get_workflow(),
                        ensure_ascii=False,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"给出的函数调用执行出错了：{exc}。"
                                "***工作流状态已更新为："
                                f"{workflow_info}***\n你要先反思犯错的原因，"
                                "避免再犯同样的错。最后请*****基于新的工作流"
                                "******重新给出正确的函数调用代码！"
                            ),
                        }
                    )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"给出的函数调用代码格式错误：{parsed_code}。"
                            "你是输出应该有且只有一个Python代码块："
                            "```python\n生成的函数调用代码\n```"
                        ),
                    }
                )
            print(f"Retry {retry_time} times...")

        self.time_cost[
            "operation_sequence_generation"
        ] += time.time() - begin_operation
        return result, False


class DirectWorkflowAgent:
    """Common Agent contract for the two no-Planning ablations."""

    retrieval_mode = "semantic"
    catch_llm_errors = False

    def __init__(
        self,
        tool_retrieve_type: int = 0,
        annotation: bool = False,
    ) -> None:
        del tool_retrieve_type
        self.planner = SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            total_price=0,
        )
        self.executor = DirectWorkflowExecutor(
            retrieval_mode=self.retrieval_mode,
            annotation=annotation,
            catch_llm_errors=self.catch_llm_errors,
        )

    def run(
        self,
        Instruction: str,
        case_number: int = 3,
        retriever: str = "multilingual-e5-large",
    ) -> dict[str, Any]:
        update_data_info(instruction=Instruction)
        self.executor.action_agent.reset()
        self.executor.reset_workflow_edit_sets()
        result, is_success = self.executor.run_instruction(
            Instruction,
            case_number=case_number,
            retriever=retriever,
        )
        result["is_success"] = is_success
        return {
            "instruction": Instruction,
            "user_intent": None,
            "plans": [],
            "execute_results": [result],
            "workflow_edit_sets": (
                self.executor.get_workflow_edit_sets()
            ),
            "is_success": is_success,
        }


def run_standalone(agent_class: type[Any]) -> None:
    """Run one baseline like the repository root ``run.py``."""
    agent = agent_class()
    agent.executor.action_agent.clear_workflow()
    result = agent.run(Instruction=DEFAULT_REQUIREMENT)
    print(json.dumps(result, indent=4, ensure_ascii=False))
