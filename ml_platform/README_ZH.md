# MLPlatAgent 平台扩展指南

[English](README.md)

`ml_platform` 用于隔离 MLPlatAgent 核心逻辑与不同机器学习平台之间的实现差异。

Planner 将用户需求拆解为机器学习任务，Executor 根据当前平台的组件目录选择组件并生成工作流操作，平台 Action 再将这些统一操作转换为平台原生的 API、SDK 或工作流文件操作。

新增平台时，开发者只需要提供：

- 平台组件目录 `widgets.json`；
- 平台工作流适配器 `PlatformAction`；
- 平台配置 `<PLATFORM>_CONFIG`；
- 平台运行环境和依赖。

Agent 核心代码始终通过同一个全局对象工作：

```python
from ml_platform.actions import action_agent
```

通常情况下，新增平台不需要修改 Planner、Executor、MLAgent 或公共 Action 工厂。

## 扩展边界

`PlatformAction` 是 Agent 与机器学习平台之间的适配层，负责：

- 校验本平台配置；
- 连接或加载当前工作流；
- 将统一工作流操作转换为平台原生操作；
- 将 Agent 可见参数转换为平台原生组件参数；
- 创建平台所需的辅助节点、连接或元数据；
- 将平台原生工作流转换为统一工作流快照；
- 隔离平台凭据、连接对象及其他私有状态；
- 对不支持的组件、参数或连接返回明确错误。

平台适配器可以通过多个原生步骤完成一个 Agent 操作，但不能改变 Agent 使用的公共 Action 契约。

## 快速接入新平台

以下步骤以平台名 `my_platform` 为例：

1. 创建 `ml_platform/my_platform/`。
2. 添加 `widgets.json`，完整描述希望开放给 Agent 的组件和参数。
3. 添加 `actions.py`，实现并导出 `PlatformAction(Action)`。
4. 在根目录 `config.py` 中添加 `MY_PLATFORM_CONFIG`。
5. 校验平台配置、加载当前工作流，并实现七个 Action 方法。
6. 将平台专属依赖和状态限制在平台目录内部。
7. 设置 `PLATFORM = "my_platform"`，并通过根目录 `run.py` 启动验证。

完成以上约定后，新平台即可在不修改 Agent 核心编排逻辑的前提下接入 MLPlatAgent。

## 目录与发现约定

平台名称必须与 `ml_platform` 下的目录名一致，并且只能包含小写字母、数字或下划线。

```text
ml_platform/
├── action.py
├── actions.py
├── README.md
├── README_ZH.md
└── my_platform/
    ├── actions.py
    ├── widgets.json
    └── 其他平台内部模块
```

公共文件职责：

- `action.py`：定义平台必须遵循的抽象接口和公共状态；
- `actions.py`：读取 `config.py`，动态加载当前平台并创建全局 `action_agent`；
- `<platform>/actions.py`：实现平台原生工作流操作；
- `<platform>/widgets.json`：声明当前平台向 Agent 开放的组件能力。

每个平台模块必须导出：

```python
class PlatformAction(Action):
    ...
```

公共工厂按照以下约定发现实现类：

```python
ml_platform.<platform>.actions.PlatformAction
```

因此，无需在工厂中增加平台专属的判断分支。

## Action 公共契约

每个平台的 `PlatformAction` 必须继承 `ml_platform.action.Action` 并实现以下方法。变更工作流的方法统一接收一个参数字典，与 Executor 生成的调用格式保持一致。

| 方法 | 必需输入 | 职责 |
| --- | --- | --- |
| `add_node` | `widget_name`、`node_name` | 创建组件节点并返回节点 ID |
| `delete_node` | `node_id` | 删除节点及平台要求的关联状态 |
| `update_node_params` | `node_id`、`widget_name`、`node_params` | 更新 Agent 可见的节点参数 |
| `add_edge` | `source_node_id`、`target_node_id` | 创建连接并返回边 ID |
| `delete_edge` | `edge_id` | 删除连接 |
| `get_workflow` | 无 | 返回统一工作流快照 |
| `clear_workflow` | 无 | 清空平台配置所指向的工作流 |

`Action` 基类已经提供组件加载与校验、组件索引、任务级组件白名单、Token 与耗时统计、布局状态、`reset()` 以及受控的 `execute_command()` 分发。平台实现应优先复用这些公共能力。

最小实现骨架：

```python
from typing import Any, Mapping

from ml_platform.action import Action


class PlatformAction(Action):
    def __init__(self, platform_config: Mapping[str, Any]):
        super().__init__(
            platform_config,
            platform_name="my_platform",
        )
        # 校验配置并加载当前工作流。

    def add_node(self, args):
        ...

    def delete_node(self, args):
        ...

    def update_node_params(self, args):
        ...

    def add_edge(self, args):
        ...

    def delete_edge(self, args):
        ...

    def get_workflow(self):
        ...

    def clear_workflow(self):
        ...
```

操作失败时，应抛出包含明确原因的 `ActionError` 或平台专属子类。平台适配器不能静默忽略失败、自动切换平台或替换为未声明的组件。

## 组件目录

Executor 将当前平台的 `widgets.json` 作为组件能力和参数定义的来源。

每个组件至少包含：

```json
{
  "widget_name": "Classifier",
  "widget_id": "platform-native-component-id",
  "description": "训练一个分类模型。",
  "type": "model",
  "params": [
    {
      "name": "max_depth",
      "type": "int",
      "default": 5,
      "description": "模型最大深度。"
    }
  ]
}
```

字段要求：

- `widget_name`：Agent 使用的唯一组件名；
- `widget_id`：由 `PlatformAction` 解释的平台原生组件标识；
- `description`：组件检索和选择使用的能力说明；
- `type`：组件对应的机器学习任务类型；
- `params`：Agent 可设置的公共参数列表；
- 每个参数必须定义 `name`、`type`、`default` 和 `description`。

`widget_name` 必须与 `PlatformAction` 接受及 `get_workflow()` 返回的名称一致。可以按需补充平台原生元数据，但不能在组件目录中保存用户凭据。

平台原生参数与 Agent 公共参数不一致时，应由 `PlatformAction` 完成转换，Planner 和 Executor 不需要理解平台原生对象。

## 统一工作流快照

`get_workflow()` 必须返回可由 `json.dumps` 序列化的结构：

```json
{
  "nodes": [
    {
      "node_id": "1",
      "widget_name": "Data Source",
      "node_name": "训练数据",
      "node_params": {
        "dataset": "customer_data"
      }
    },
    {
      "node_id": "2",
      "widget_name": "Classifier",
      "node_name": "分类模型",
      "node_params": {}
    }
  ],
  "edges": [
    {
      "edge_id": "1",
      "source_node_id": "1",
      "target_node_id": "2"
    }
  ]
}
```

快照要求：

- `nodes` 和 `edges` 必须始终存在且类型为列表；
- 节点和边 ID 必须在同一快照中稳定对应；
- `widget_name` 必须使用 `widgets.json` 中声明的 Agent 可见名称；
- `node_params` 必须使用 Agent 可见的公共参数；
- 不能包含 Token、密码、平台连接对象或不可序列化值；
- 平台适配器创建的辅助节点如果会影响后续工作流编排，应准确体现在快照中。

## 配置与启动

平台选择和用户配置统一放在根目录 `config.py` 中，不使用命令行参数：

```python
PLATFORM = "my_platform"

MY_PLATFORM_CONFIG = {
    "workspace": "",
}
```

配置名称必须遵循以下约定：

```text
<PLATFORM.upper()>_CONFIG
```

工厂只读取当前平台的配置，并将配置字典传给 `PlatformAction`。平台适配器负责校验自身必需字段。未选中平台的配置和依赖不应参与初始化。

`action_agent` 是进程级单例。修改 `PLATFORM` 或对应配置后，需要重新启动 Python 进程。

项目必须通过根目录入口启动：

```powershell
python .\run.py
```
