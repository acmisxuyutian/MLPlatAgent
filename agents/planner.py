# -*- coding: utf-8 -*-
import json
import os

from llm.llm import Qwen_Model
from utils.utils import load_json, get_project_root
from prompts.planner_prompts import INTENT_PROMPT, PLAN_PROMPT, Experiences
from utils.logs import logger
from ml_platform.actions import action_agent
from embedding_models.embedding_model import Embedding_Model

class Planner:

    def __init__(self, annotation=False):
        self.llm = Qwen_Model()
        self.plan_messages = []
        self.annotation = annotation
        self.action_agent = action_agent
        self.current_workflow = None
        self.total_price = 0
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.time_cost = 0

    def run(self, Instruction):

        logger.info(f"User Instruction: {Instruction}")

        user_intent = self.intent(Instruction)
        logger.info(f"用户意图: {user_intent}")
        if user_intent == "Other":
            return user_intent, []
        elif user_intent in ["Traditional Machine Learning", "Deep Learning", "Modify"]:
            plans = self.plan(Instruction, user_intent)
            return user_intent, plans
        else:
            return user_intent, []

    def intent(self, Instruction, max_tyr_times=3):
        """
        返回当前的对话意图
        """
        self.current_workflow = self.action_agent.get_workflow()
        messages = [
            {
                "role": "system",
                "content": INTENT_PROMPT
            },
            {
                "role": "user",
                "content": Instruction
            }
        ]
        tyr_times = 0
        while tyr_times < max_tyr_times:
            content, message, input_tokens, output_tokens, price = self.llm.predict(messages=messages)
            self.total_price += price
            self.total_tokens += (input_tokens + output_tokens)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

            """
            ```json
            {
                "thought": "中文描述的思考",
                "intent": "Traditional Machine Learning,Deep Learning,Modify,Other中的一个"
            }
            ```
            """
            content = load_json(content)
            if content != {}:
                if content["intent"] in ["Traditional Machine Learning", "Deep Learning", "Modify", "Other"]:
                    return content["intent"]
                else:
                    feedback = "意图类型应该是Traditional Machine Learning,Deep Learning,Modify,Other中的一个，请重新输入。"
            else:
                feedback = "请返回严格的JSON格式！确保能够被python的json.loads()函数解析，且包含intent字段。"
            tyr_times += 1
            logger.info(f"第{tyr_times}次尝试")
            messages.append(message)
            messages.append({
                "role": "user",
                "content": feedback
            })
        return ""

    def plan(self, Instruction, user_intent, max_tyr_times=3):
        """
        负责生成任务列表
        :return: list:任务列表
        """
        if user_intent != "Modify":
            self.action_agent.clear_workflow()
        self.current_workflow = json.dumps(self.current_workflow, ensure_ascii=False)
        logger.info(f"current workflow: {self.current_workflow}")

        system_prompt = PLAN_PROMPT.format(
            experiences=Experiences[user_intent],
            case=self.retrieve_cases(Instruction, user_intent)
        )
        user_prompt = Instruction + "\n当前工作流状态为：" + self.current_workflow
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
        tyr_times = 0
        while tyr_times < max_tyr_times:
            content, message, input_tokens, output_tokens, price = self.llm.predict(messages=messages)
            self.total_price += price
            self.total_tokens += (input_tokens + output_tokens)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

            if self.annotation:
                # 用于存储输入的文本
                content = ""

                # 提示用户开始输入
                print("输入计划列表，以 '*' 结束：")

                # 循环读取输入，直到用户输入'*'
                while True:
                    line = input()
                    if line == '*':
                        break
                    content += line + '\n'  # 添加换行符以保持文本的格式

                planner_path = os.path.join(get_project_root(), r"data/cases_library/planner.json")
                with open(planner_path, encoding="utf-8") as f:
                    planner_data = json.load(f)
                planner_data.append(self.plan_messages)
                with open(planner_path, "w", encoding="utf-8") as f:
                    json.dump(planner_data, f, ensure_ascii=False, indent=4)

            plan = load_json(content)
            if plan != {}:
                if user_intent in ["Traditional Machine Learning", "Modify"]:
                    types = ["io", "preprocess", "feature engineering", "model", "evaluate", "predict", "visualize"]
                else:
                    types = ["io", "model", "predict", "visualize"]
                errors = []
                for p in plan:
                    if "type" not in p:
                        errors.append("你给出的任务列表缺少'type'信息，请仔细查看输出格式要求！")
                        break
                    else:
                        if p["type"] not in types:
                            errors.append(f"任务{p['id']}的类型：{p['type']}不在可选类型{types}中")
                if len(errors) > 0:
                    feedback = "\n".join(errors)
                else:
                    return self.merger_plan(plan)
            else:
                feedback = "请返回严格的JSON格式！确保能够被python的json.loads()函数解析。"

            tyr_times += 1
            logger.info(f"第{tyr_times}次尝试")
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": feedback
            })

        return []

    def merger_plan(self, plans):
        """
        如果连续多个任务的任务类型相同，且依赖关系是顺序依赖的，则将这些任务合并为一个任务
        """
        merged_plans = []
        i = 0
        while i < len(plans):
            if i != len(plans)-1 and plans[i]["type"] == plans[i+1]["type"]:
                j = i
                while j != len(plans)-1 and plans[j]["type"] == plans[j+1]["type"]:
                    j += 1
                merged_plan_dependencies = f"你需要完成以下{j-i+1}个任务："
                k = 0
                while i <= j:
                    k += 1
                    merged_plan_dependencies += f"{k}.{plans[i]['description']}"
                    i += 1
                merged_plans.append({
                    "id": len(merged_plans)+1,
                    "description":  merged_plan_dependencies,
                    "type": plans[i-1]["type"]
                })
            else:
                merged_plans.append({
                    "id": len(merged_plans)+1,
                    "description": plans[i]["description"],
                    "type": plans[i]["type"]
                })
                i += 1
        return merged_plans

    def retrieve_cases(self, Instruction, user_intent, case_number=3):
        case_file = os.path.join(get_project_root(), r"data/cases_library/planner_cases.json")
        with open(case_file, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
        candidate_cases = [case for case in case_data if case["type"] == user_intent]

        if len(candidate_cases) > case_number:
            model_name_path = os.path.join(get_project_root(), r"embedding_models/stella-large-zh-v3-1792d")
            embedding_model = Embedding_Model(model_name_path)
            corpus = [case["Instruction"] for case in candidate_cases]
            pairs_sorted = embedding_model.get_scores(Instruction, corpus, case_number)
            relevant_cases = [candidate_cases[index] for score, index in pairs_sorted]
        else:
            relevant_cases = candidate_cases

        result = ""
        for i in range(len(relevant_cases)):
            case = relevant_cases[i]
            result += f"## 示例{i+1}\n"
            result += f"User：\n" + case["Instruction"] + "\nAssistant：\n```json\n" + json.dumps(case["plan"], ensure_ascii=False) + "\n```\n\n"
        return result