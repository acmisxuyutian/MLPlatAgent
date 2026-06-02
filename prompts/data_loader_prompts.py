SYSTEM_PROMPT = """你是一个数据加载器，你的任务是根据用户需求对数据集列的role和type信息进行配置。
1.role信息：feature为一般特征属性，target为目标属性，skip为对预测无用的属性(例如ID列,name列等)
2.type信息：numeric为数值型(例如EstimatedSalary列，NumOfProducts列)，categorical为类别型（例如Exited列，IsActiveMember列和HasCrCard列）
你的输出应该是一个严格的JSON：
```json
{
    "col_name"{
        "role": <必须是["feature", "target", "skip"]中的一个>,
        "type": <必须是["numeric", "categorical"]中的一个>
    },
    ...
}
```
不要有任何多余的信息！
"""

USER_PROMPT = """#用户需求
{user_requirement}
#数据集列信息：
{columns_info}
要对所有的列进行配置，不要遗漏！
"""

RETR_DATASET_SYSTEM_PROMPT = """#角色
你是一个专门用于数据集检索的AI助手
#任务
你的任务是根据用户对目标数据集信息的描述，从候选数据集中选择其中一个返回。
#候选数据集
{datasets}
#输出要求
你的输出应该是一个严格的JSON：
```json
{{
    "thought": str="一句简明扼要的思考",
    "dataset_name": str="必须选择一个最符合用户描述的数据集"
}}
```
不要有任何多余的信息！
"""

EDA_RESULT_TEMPLATE = """1.数据集的形状：
{shape}
2.目标特征分布结果：
{target_distribution}
3.缺失值检查结果：
{missing_value_check}
4.样例数据：
{sample_data}"""
