# -*- coding: utf-8 -*-
import time
from agents.planner import Planner
from agents.executor import Executor
from utils.utils import update_data_info


class MLAgent:

    def __init__(self, tool_retrieve_type=0, annotation=False):
        self.planner = Planner(annotation)
        self.executor = Executor(annotation=annotation, tool_retrieve_type=tool_retrieve_type)

    def run(self, Instruction, case_number=3, retriever="multilingual-e5-large"):
        time_costs = {
            "task_decomposition": 0,
            "widget_retrieval": 0,
            "data_retrieval": 0,
            "operation_sequence_generation": 0
        }

        # 每次调用都重置统计与本次编辑记录，但不改变当前工作流。
        self.executor.action_agent.reset()
        self.executor.reset_workflow_edit_sets()
        update_data_info(instruction=Instruction)

        base_results = {
            "instruction": Instruction,
            "user_intent": None,
            "plans": [],
            "execute_results": [],
            "workflow_edit_sets": [],
            "is_success": False,
            "status": "failed",
            "response": "",
        }

        # 意图识别与任务规划
        begin_time = time.time()
        user_intent, plans = self.planner.run(Instruction)
        end_time = time.time()
        time_costs["task_decomposition"] = end_time - begin_time
        results = base_results
        results["user_intent"] = user_intent
        results["plans"] = plans
        if plans == []:
            results["error"] = "未生成可执行的工作流计划"
            results["response"] = results["error"]
            results[
                "workflow_edit_sets"
            ] = self.executor.get_workflow_edit_sets()
            return results
        self.executor.user_intent = user_intent

        # 任务执行
        for task in plans:
            result, is_success = self.executor.run(task, case_number, retriever)
            time_costs["widget_retrieval"] = self.executor.time_cost["widget_retrieval"]
            time_costs["operation_sequence_generation"] = self.executor.time_cost["operation_sequence_generation"]
            time_costs["data_retrieval"] = self.executor.action_agent.data_retrieval_time
            result["is_success"] = is_success
            results["execute_results"].append(result)
            if not is_success:
                break
        results["is_success"] = (
            len(results["execute_results"]) == len(plans)
            and all(
                item.get("is_success") is True
                for item in results["execute_results"]
            )
        )
        results[
            "workflow_edit_sets"
        ] = self.executor.get_workflow_edit_sets()
        results["status"] = (
            "completed" if results["is_success"] else "failed"
        )
        if not results["is_success"]:
            failed_result = next(
                (
                    item
                    for item in results["execute_results"]
                    if item.get("is_success") is False
                ),
                {},
            )
            if failed_result.get("error"):
                results["error"] = failed_result["error"]
                results["response"] = failed_result["error"]
        return results
