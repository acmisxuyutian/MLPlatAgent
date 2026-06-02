# -*- coding: utf-8 -*-
import json
import os
import jieba
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from prompts.data_loader_prompts import SYSTEM_PROMPT, USER_PROMPT, EDA_RESULT_TEMPLATE, RETR_DATASET_SYSTEM_PROMPT
from llm.llm import Qwen_Model
from utils.utils import load_json, get_project_root,update_data_info
from ml_platform.ai_studio import AI_Studio
from config import MySQL_Config
from utils.mysql_utils import MySQLDatabase
from utils.logs import logger

class Data_Loader:

    def __init__(self, ai_studio, annotation=False):
        self.ai_studio = ai_studio
        self.annotation = annotation
        self.db = MySQLDatabase(host=MySQL_Config["server"], port=MySQL_Config["port"], user=MySQL_Config["username"], password=MySQL_Config["password"], database=MySQL_Config["database"])

    def run(self, data_description, node_id):
        self.db.connect()
        total_input_tokens = 0
        total_output_tokens = 0
        table_name, input_tokens, output_tokens = self.retrieve_dataset(data_description)
        if table_name == None:
            update_data_info(dataset_info="")
            return "", 0, total_input_tokens, total_output_tokens

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        logger.info(f"llm检索到的数据集为: {table_name}")
        logger.info("正在从数据库加载数据...")
        data = self.load_data(table_name)
        logger.info("数据加载完成")

        logger.info("正在对数据集的列进行配置...")
        mapping, result_json, price, input_tokens, output_tokens, sample_value = self.get_columns(data)
        logger.info(f"配置完成")
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        self.update_node_params({
            "table_name": table_name,
            "attr_mapping": mapping,
            "node_id": node_id
        })

        eda_result = self.eda(data, result_json, sample_value)

        self.db.close_connection()

        update_data_info(dataset_info=eda_result)

        return eda_result, price, total_input_tokens, total_output_tokens

    def load_data(self, table_name):

        # 查询示例
        select_query = f"SELECT * FROM {table_name}"

        # 执行查询并获取结果和列名
        results, column_names = self.db.read_query(select_query)
        data = None
        # 检查results是否为None
        if results is not None:
            # 将查询结果转换为DataFrame
            data = pd.DataFrame(results, columns=column_names)

        else:
            print("查询没有返回任何结果")

        return data

    def get_columns(self, data):
        """
        {
            "PassengerId": {
                "key": 0,
                "name": "PassengerId",
                "type": 2,  // 属性类型编号，1：离散属性，2：数值属性，3：文本属性，4：日期属性
                "role": 0   // 属性类别编号，0：特征属性，1：目标属性，2：描述属性，-1：忽略该属性
            }
            ...
        }
        """
        ############################################################## 1.获取数据集的字段配置 ##############################################################
        import random
        total_input_tokens = 0
        total_output_tokens = 0
        # 去除含有缺失值的行
        df_clean = data.dropna()
        if len(df_clean) < 3:
            df_clean = data
        # 生成3个随机索引
        random_indices = random.sample(range(len(df_clean)), 3)

        # 选择这些随机索引对应的行
        random_rows = df_clean.iloc[random_indices]

        random_rows = random_rows.to_dict(orient='records')
        sample_value = {}
        for col in data.columns:
            sample_value[col] = []
            for  i in range(len(random_rows)):
                sample_value[col].append(random_rows[i][col])

        columns_info = data.columns.to_list()
        data_path = os.path.join(get_project_root(), r"data/data_info.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            data_info = json.load(f)
        prompt = USER_PROMPT.format(user_requirement=data_info["instruction"], columns_info=columns_info)
        llm = Qwen_Model()
        msgs = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        max_try = 3
        retry_time = 0
        while retry_time < max_try:
            content, message, input_tokens, output_tokens, price = llm.predict(messages=msgs)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            result_json = load_json(content)
            feedback = ""
            # 检查JSON解析是否成功
            if content != {}:
                if len(result_json) != len(columns_info):
                    feedback = "请重新输入，确保配置的列与数据集存在的列一致。不要猜测数据集存在的列，我给你的列就是对的！"
                else:
                    break
            else:
                feedback = """JSON解析错误！请返回严格的JSON格式，并用```json和```括起来
```json
{
"col_name"{
        "role": <必须是["feature", "target", "skip", "meta"]中的一个>,
        "type": <必须是["numeric", "categorical", "datetime", "text"]中的一个>
    },
    ...
}
```"""
            retry_time += 1
            logger.info(f"第{retry_time}次尝试")
            msgs.append(message)
            msgs.append({
                "role": "user",
                "content": feedback
            })
        # 将结果处理为file组件需要的数据
        role_map = {
            "feature": 0,
            "target": 1,
            "meta": 2,
            "skip": -1
        }
        type_map = {
            "numeric": 2,
            "categorical": 1,
            "text": 3,
            "datetime": 4
        }
        mapping = {}
        count = 0
        for key, value in result_json.items():
            mapping[key] = {
                "key": count,
                "name": key,
                "type": type_map[value["type"]],
                "role": role_map[value["role"]]
            }
            count += 1
        return mapping, result_json, price, total_input_tokens, total_output_tokens, sample_value

    def update_node_params(self, args):

        widget_params = {
            "type": "2",
            "server": MySQL_Config["server"],
            "database": MySQL_Config["database"],
            "schema": "public",
            "port": MySQL_Config["port"],
            "username": MySQL_Config["username"],
            "password": MySQL_Config["password"],
            "dbType": None,
            "load_type": "table",
            "getDatabaseFlag": False,
            "selectedDatasource": None,
            "selectedDatasourceUserId": -1,
            "isNewIDIS": True,
            "table_name": args["table_name"],
            "save2db": False,
            "sql_code": "",
            "dump_table": "my_table",
            "selectedDataset": "",
            "dataset_name": None,
            "dataset_sql": None,
            "field": None,
            "interact_type": 0,
            "attr_mapping": args["attr_mapping"]
        }

        data = {
            "workflow_id": self.ai_studio.workflow_id,
            "param_info": {str(args["node_id"]): widget_params},
        }

        data = json.dumps(data, ensure_ascii=False)

        result = self.ai_studio.update_node_params(data)
        return result

    def eda(self, data, result_json, sample_value):
        df = data
        ############################################################## 3.整理EDA的结果 ##############################################################
        eda_result = EDA_RESULT_TEMPLATE.replace("{shape}", str(df.shape))
        columns = df.columns.tolist()
        categorical_feature = []
        numerical_feature = []
        target = {"name": "", "type": ""}
        for i in range(len(columns)):
            name = columns[i]
            role = result_json[name]["role"]
            type = result_json[name]["type"]
            if role == "target":
                target["name"] = name
                target["type"] = type
            elif role == "feature":
                if type == "categorical":
                    categorical_feature.append(name)
                elif type == "numeric":
                    numerical_feature.append(name)

        if target["type"] == "categorical":
            target_distribution = df[target["name"]].value_counts().to_string()
        else:
            target_distribution = df[target["name"]].describe().to_string()
        eda_result = eda_result.replace("{target_distribution}", target_distribution)

        columns_info = []
        sample_data = []
        for c in df.columns:
            if result_json[c]["role"] == "feature":
                sample_data.append({
                    "column name": c,
                    "sample data": sample_value[c]
                })
                # "data type": result_json[c]["type"]
                columns_info.append(
                    {
                        "column name": c,
                        "data type": result_json[c]["type"]
                    }
                )
        eda_result = eda_result.replace("{sample_data}", json.dumps(sample_data, ensure_ascii=False))
        # categorical_analysis = ""
        # for cf in categorical_feature:
        #     categorical_analysis += df[cf].value_counts().to_string() + "\n"
        # eda_result = eda_result.replace("{categorical_analysis}", categorical_analysis)
        #
        # numerical_analysis = df[numerical_feature].describe().to_string()
        # eda_result = eda_result.replace("{numerical_analysis}", numerical_analysis)
        #
        # correlation_check = df[numerical_feature + [target["name"]]].corr().to_string()
        # eda_result = eda_result.replace("{correlation_check}", correlation_check)

        missing_value_check = df.isnull().sum()
        missing_value_check = missing_value_check[missing_value_check > 0].to_string()
        eda_result = eda_result.replace("{missing_value_check}", missing_value_check)
        return eda_result

    def retrieve_dataset(self, data_description):
        """
        使用了HyDE检索技术：LLM生成的data_description包含了假设文档 + BM25/语义相似度检索
        """

        datasets_info = self.db.get_database_info()

        # ######################## BM25 ReCall ########################
        # corpus = [datasets["dataset_name"]+json.dumps(datasets["columns"], ensure_ascii=False) for datasets in datasets_info]
        #
        # tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
        #
        # bm25 = BM25Okapi(tokenized_corpus)
        #
        # task_des_tokens = jieba.lcut(data_description)
        # doc_scores = bm25.get_scores(task_des_tokens)
        # top_indexes = np.argsort(doc_scores)[::-1][:5]
        #
        # recalled_dataset = [datasets_info[index] for index in top_indexes]
        #
        # dataset_names = [d["dataset_name"] for d in recalled_dataset]
        # logger.info("BM25检索的数据集: {}".format(dataset_names))

        ######################## 语义相似度 ReCall ########################
        from embedding_models.embedding_model import Embedding_Model
        corpus = [datasets["dataset_name"] + json.dumps(datasets["columns"], ensure_ascii=False) for datasets in
                  datasets_info]
        model = Embedding_Model("all-mpnet-base-v2")
        pairs_sorted = model.get_scores([data_description], corpus,topk=7)
        dataset_names = [datasets_info[index]["dataset_name"] for s, index in pairs_sorted]
        logger.info("语义相似度检索的数据集: {}".format(dataset_names))

        ######################## LLM ReRank ########################
        system_prompt = RETR_DATASET_SYSTEM_PROMPT.format(datasets=dataset_names)
        llm = get_llm()
        msgs = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": data_description
            }
        ]
        total_input_tokens = 0
        total_output_tokens = 0
        max_try = 3
        retry_time = 0
        while retry_time < max_try:
            retry_time += 1
            content, message, input_tokens, output_tokens, price = llm.predict(messages=msgs)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            content = load_json(content)
            if content != {}:
                if content["dataset_name"] in dataset_names:
                    return content["dataset_name"], total_input_tokens,  total_output_tokens
                else:
                    feedback = f"你选择的数据集{content['dataset_name']}不合法，请从以下数据集中选择：{dataset_names}"
            else:
                feedback = "JSON解析错误！你的输出应该是一个严格的JSON，确保你的输出能够被python的json.loads()函数解析。重新选择一个数据集"

            msgs.append(message)
            msgs.append({
                "role": "user",
                "content": feedback
            })

        return None, total_input_tokens,  total_output_tokens

def load_data(data_description, node_id):
    data_loader = Data_Loader(AI_Studio())
    return data_loader.run(data_description, node_id)