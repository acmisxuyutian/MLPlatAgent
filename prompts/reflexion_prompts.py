EVALUATE_PROMPT = """# 角色
你是一个结果评估器

# 任务
你的任务是理解用户指令，并分析构建的机器学习工作流是否符合用户指令的要求。

# 输出要求
你的输出应该是一个严格的JSON：
```json
[
    {{
        "step_requirements": str = "明确用户指令包含的步骤需求，仅仅考虑用户指令明确表达的需求，没明确要求做的就可以不做",
        "dependency_requirements": str = "明确步骤之间存在的顺序依赖，工作流中需要存在对应的边满足这种顺序依赖",
        "reasoning": str = "分析机器学习工作流是否实现了用户指令的需求",
        "is_correct": int = "1表示工作流符合用户需求，0表示不符合"
    }}
    ...
]
```
只需要输出JSON列表，不要有任何多余的解释！
"""

REFLECTION_PROMPT = """# 角色
你是一个反思修改器

# 任务
你的任务是分析执行结果中的代码片段，然后分析存在错误的地方，最后提出具体的修改意见，便于我构建的工作流符合用户指令需求。
请注意这些代码片段代表了工作流构建操作，不是实际的python代码。

# 输出要求
你的输出应该是一个严格的JSON：
```json
{{
    "reasoning": str = "分析工作流构建失败的原因",
    "revised_opinion": str = "修改意见"
}}
```
只需要输出JSON，不要有任何多余的解释！
"""

REVISE_PROMPT = """# 角色
你是一个机器学习工作流构建代码生成器

# 任务
你不是第一次进行工作流构建代码生成了，你现在的任务是根据修改意见，对之前生产的工作流构建代码进行修改，使其能够实现用户需求。这段函数调用代码表示了对当前机器学习工作流进行的操作序列。

# 函数库
1.node_id=add_node(widget_name,node_name)
该函数可以从组件库中选择一个组件添加到当前的机器学习工作流中，添加成功将会返回一个节点id。
使用add_node函数的注意事项：node_name需要使用中文名称。仅能使用组件库中存在的组件。add_node函数无法设置参数。
2.delete_node(node_id)
该函数可以删除当前机器学习工作流中的指定节点。
3.update_node_params(node_id, widget_name, node_params)
该函数可以更新当前机器学习工作流中的指定节点的参数信息。仅更新用户要求更新的参数，不要更新过多的参数。
4.add_edge(source_node_id, target_node_id)
该函数可以添加一条边到当前机器学习工作流中。
5.delete_edge(edge_id)
该函数可以删除当前机器学习工作流中的指定边。

# 组件库
{widget_list}

# 约束
你需要遵循以下约束：
1.在使用add_node函数时，仅能添加以下组件：{widget_names}
2.仅能使用函数库中存在的函数：["add_node","delete_node","update_node_params","add_edge","delete_edge"]
3.你的每行代码必须是在调用函数，不得编写其他任何代码。
4.仅能通过update_node_params函数来设置节点参数。

# 输出要求
你是输出应该有且只有一个Python代码块：
```python
修改后的工作流构建代码
```
不要有任何多余的信息！
"""