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

        # 数据初始化
        update_data_info(instruction=Instruction)
        self.executor.action_agent.clear_workflow()
        self.executor.action_agent.reset_XY()
        
        # 意图识别与任务规划
        begin_time = time.time()
        user_intent, plans = self.planner.run(Instruction)
        end_time = time.time()
        time_costs["task_decomposition"] = end_time - begin_time
        results = {
            "instruction": Instruction,
            "user_intent": user_intent,
            "plans": plans,
            "execute_results": []
        }
        if plans == []:
            total_input_tokens = self.planner.input_tokens + self.executor.input_tokens + self.executor.action_agent.input_tokens
            total_output_tokens = self.planner.output_tokens + self.executor.output_tokens + self.executor.action_agent.output_tokens
            return False, total_input_tokens, total_output_tokens, time_costs, results
        self.executor.user_intent = user_intent

        # 任务执行
        for task in plans:
            result, is_success = self.executor.run(task, case_number, retriever)
            time_costs["widget_retrieval"] = self.executor.time_cost["widget_retrieval"]
            time_costs["operation_sequence_generation"] = self.executor.time_cost["operation_sequence_generation"]
            results["execute_results"].append(result)
        return results