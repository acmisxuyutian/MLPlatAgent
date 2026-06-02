"""
# 示例
user：
我想开发一个机器学习模型，用于识别鸢尾花种类，过程中我需要查看日志，使用随机森林模型。
assistant：
{
    "thought": "这是一个表格分类任务",
    "intent": "Traditional Machine Learning"
}
user：
我想开发一个能够识别手写数字的图像分类器，数据集使用MNIST，模型架构为卷积神经网络，并使用Adam优化器进行训练。
assistant：
{
    "thought": "这是一个图像分类任务。",
    "intent": "Deep Learning"
}
user：
我想要将现有的逻辑回归模型中的正则化强度从L1改为L2，并调整正则化系数为0.5。
assistant：
{
    "thought": "仅需要更改评估指标，不涉及端到端的模型训练任务。",
    "intent": "Tuning"
}
user：
你好！
assistant：
{
    "thought": "不涉及机器学习任务。",
    "intent": "Other"
}
"""
INTENT_PROMPT = """# 角色
你是一个意图识别器

# 任务
你的任务是理解并分析用户请求，对用户请求进行意图分类。

# 用户意图
用户意图只可能是以下四种类型之一：
1.Traditional Machine Learning：用户的请求涉及到表格分类，表格回归，时序预测等传统机器学习任务。
2.Deep Learning：用户请求涉及到图像分类，目标检测，图像分割，文本分类，机器翻译等深度学习任务。
3.Modify：用户请求***仅***涉及对已经构建好的机器学习管道进行修改。例如，替换数据集或替换模型，更改模型超参数。
4.Other：以上都不是

# 约束
1.当用户意图可能属于多种类型时，只返回其中一种类型，Traditional Machine Learning和Deep Learning优先级高于Modify。
2.请推测机器学习任务类别来区分Traditional Machine Learning和Deep Learning。
3.分类任务，涉及到文本，图像数据应该返回Deep Learning。如果是表格数据，返回Traditional Machine Learning。

# 输出要求
你的输出应该是一个严格的JSON：
```json
{
    "thought": "中文描述的思考",
    "intent": "用户意图类型"
}
```
不要有任何多余的信息！
"""

PLAN_PROMPT = """# 角色
你是一个任务规划器

# 任务
你的任务是理解用户指令，并分析当前的工作流状态，最后根据指导原则给出一个能够实现用户目标的任务列表。

# 指导原则
{experiences}

# 演示案例
{case}

# 输出要求
你的输出应该是一个严格的JSON：
```json
[
    {{
        "id": int = "任务的唯一标识",
        "dependencies": list[int] = "该任务依赖的任务ID",
        "description": str = "中文的任务描述",
        "type": str = "任务类型"
    }}
    ...
]
```
只需要输出JSON列表，不要有任何多余的解释！
"""

TML = """1.读取数据的任务描述，需要指定数据是训练数据还是测试数据，默认情况下，请加载训练数据。
2.对于model类的任务描述，需要描述清楚要训练哪一类模型，可以是以下四种：表格分类模型、表格回归模型、时序预测模型，用户指定的模型名。
3.如果用户没有明确要求进行评估、预测或可视化，请不要添加这些子任务。
4.训练数据和测试数据需要进行同样的预处理和特征工程处理。
5.任务列表中的任务类型必须从以下选项中选择：io、preprocess、feature engineering、model、evaluate、predict、visualize。
6.如果用户表明了要加载示例数据集，你需要加载示例数据集，并将其作为训练数据。"""

DL = """1.任务列表中的任务类型必须从以下选项中选择：io、model、predict、visualize。
2.对于model类的任务描述，需要描述清楚要训练哪一类模型，可以是以下六种：文本分类模型、机器翻译模型、图像分类模型，对图像像素进行分类为图像分割模型，定位和识别物品的具体位置和范围为目标检测模型，用户指定的模型名。。
3.*****如果用户没有明确要求进行预测任务或可视化任务，请不要添加这些子任务。请你不要进行胡乱推测！*****
4.读取的数据集需要表明是文本数据还是图像数据。
"""

Modify = """用户需要对机器学习工作流进行调整，你需要遵循以下指导原则进行任务规划：
1.任务列表中的任务类型必须从以下选项中选择：io、preprocess、feature engineering、model、evaluate、predict、visualize。
2.对于model类的任务描述，需要你根据工作流中使用的数据集和模型推断当前用户构建的模型需要用于什么机器学习任务，必须是以下8种任务之一：文本分类任务、机器翻译任务、图像分类任务、图像分割任务、目标检测任务、表格分类任务、表格回归任务、时序预测任务。
    如，更换当前的使用的KNN模型，尝试使用一个不同的模型用于表格分类任务。
"""

Experiences = {
    "Traditional Machine Learning": TML,
    "Deep Learning": DL,
    "Modify": Modify
}
