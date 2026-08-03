# Orange3 工作流构建操作

本目录封装了便携版 Orange3-3.40.0 的工作流编辑能力，供 Agent 通过固定动作构建、修改和读取 Orange 可视化工作流。

操作对象是一个已经存在的 `.ows` 文件。每次成功的增删改操作都会立即保存到该文件，因此 Agent 生成的工作流可以直接用 Orange 画布打开。

## 一、运行环境

默认使用项目中的便携版 Orange：

```text
Orange3-3.40.0/
└── Orange/
    ├── python.exe
    └── pythonw.exe
```

建议在项目根目录使用便携版 Python 运行代码：

```powershell
.\Orange3-3.40.0\Orange\python.exe your_script.py
```

公开接口可以从包入口导入：

```python
from workflow_graph_edits.orange3 import (
    add_edge,
    add_node,
    configure_workflow,
    delete_edge,
    delete_node,
    get_test_and_score_results,
    get_workflow,
    open_canvas,
    update_node_params,
)
```

## 二、为 Agent 指定当前工作流

Agent 调用工作流操作前，必须先使用 `configure_workflow` 指定：

- 当前任务允许使用的组件；
- 当前需要编辑的现有 `.ows` 工作流文件。

```python
from workflow_graph_edits.orange3 import configure_workflow

WORKFLOW_PATH = r"D:\workflows\current.ows"

configure_workflow(
    relevant_widgets_names=[
        "File",
        "Select Columns",
        "Random Forest",
        "Test and Score",
        "Predictions",
        "Confusion Matrix",
        "Save Data",
    ],
    workflow_path=WORKFLOW_PATH,
)
```

`workflow_path` 是必填参数。文件不存在或无法被 Orange 加载时会抛出 `WorkflowActionError`，不会隐式创建其他工作流。

再次调用 `configure_workflow` 并传入另一个 `.ows` 文件，即可切换当前工作流。模块级操作在同一进程中共享这一个“当前工作流”，因此并发操作多个工作流时应分别使用 `Actions` 实例。

## 三、提供给 Agent 的六个操作

以下操作名称和参数规则是固定的。

### 1. 添加节点

```python
node_id = add_node(widget_name, node_name)
```

用途：从当前任务允许使用的组件中选择一个组件，添加到当前工作流。

规则：

- `widget_name` 必须存在于 `relevant_widgets_names`；
- `widget_name` 必须使用 `widgets.json` 中的真实 Orange 组件名称；
- `node_name` 必须是非空中文名称；
- 本操作不能设置组件参数；
- 成功后返回字符串形式的节点 ID。

示例：

```python
file_node = add_node("File", "读取训练数据")
forest_node = add_node("Random Forest", "随机森林模型")
```

### 2. 删除节点

```python
delete_node(node_id)
```

用途：删除指定节点。与该节点连接的边也会一并删除。

示例：

```python
delete_node(forest_node)
```

### 3. 更新节点参数

```python
update_node_params(node_id, widget_name, node_params)
```

用途：更新指定节点的组件参数。

规则：

- `widget_name` 必须与该节点实际使用的组件一致；
- `node_params` 必须是标准 JSON 对象；
- 参数名必须存在于该组件的 `widgets.json` 定义中；
- 参数值必须符合定义的类型；
- 只更新调用方明确提交的字段，其他参数和默认值保持不变。

示例：

```python
update_node_params(
    forest_node,
    "Random Forest",
    {
        "n_estimators": 100,
        "use_max_depth": True,
        "max_depth": 8,
    },
)
```

### 4. 添加边

```python
edge_id = add_edge(source_node_id, target_node_id)
```

用途：连接两个工作流节点。

封装层会根据 Orange 的输入输出信号自动选择语义最合适的兼容端口。节点之间没有兼容端口、重复连接或连接自身时会报错。

示例：

```python
edge_id = add_edge(file_node, forest_node)
```

### 5. 删除边

```python
delete_edge(edge_id)
```

用途：根据边 ID 删除连接。

示例：

```python
delete_edge(edge_id)
```

### 6. 获取工作流

```python
workflow = get_workflow()
```

用途：返回当前工作流的标准 JSON-ready Python 字典。

```json
{
  "edges": [
    {
      "edge_id": "936445958711154875",
      "source_node_id": "1",
      "target_node_id": "3"
    }
  ],
  "nodes": [
    {
      "node_id": "1",
      "widget_name": "File",
      "node_name": "读取训练数据",
      "node_params": {
        "source": 0,
        "url": ""
      }
    }
  ]
}
```

新建的空工作流应返回：

```json
{"edges": [], "nodes": []}
```

节点参数包含 `widgets.json` 中定义的默认值。节点 ID、边 ID、中文节点名称、组件名称和参数会持久化到 `.ows` 文件，并在重新加载后保持稳定。

### 获取 Test and Score 运行结果

```python
result = get_test_and_score_results(
    node_id="5",
    timeout=60,
)
```

该 API 会实际运行当前 `.ows` 工作流，等待 Test and Score 的异步评估
完成，并返回标准 JSON。`node_id` 可省略；省略时返回工作流中全部
Test and Score 节点的结果。

返回内容包括：

- 评估方式、折数和是否分层抽样；
- 实际参与评估的数据行数；
- 目标变量名称、类型和离散取值；
- 每个学习器的名称和指标；
- 执行失败的学习器及错误信息；
- 无法计算的指标及原因。

分类任务返回 AUC、CA、F1、Precision、Recall、LogLoss、
Specificity 和 MCC；回归任务返回 MSE、RMSE、MAE、MAPE、SMAPE、
R2 和 CVRMSE。

SQL Table 的用户名和密码仍从 Orange 系统凭据库读取，不会写入
工作流 JSON。数据库不可访问、节点输入不完整或超过 `timeout`
时会抛出 `WorkflowActionError`。

## 四、组件和参数定义

[widgets.json](./widgets.json) 由便携版 Orange3-3.40.0 的运行时组件注册表生成，当前包含：

- 33 个能够映射到 UniPlore 的 Orange 组件；
- 组件注册 ID、Python 类、类别、图标和基于 Orange 官方文档的用途描述；
- 每个组件对应的 `orange3-master/doc/visual-programming/source/widgets` 官方文档路径；
- 输入与输出信号；
- 221 个可用标准 JSON 表示的有效参数，以及参数类型、默认值和中文含义。

组件的 `description_source` 和参数的 `description_source` 用于区分说明来源：

- `official_document`：参数或功能能够在对应 Orange 官方组件文档中定位；
- `source_inference`：官方文档没有逐项说明，根据 Orange3-3.40.0 组件源码中的 `Setting`、参数名称、默认值和界面用途推断。

例如：

```json
{
  "widget_name": "Random Forest",
  "description": "Random Forest 组件，属于“机器学习模型构建”类别。Orange 官方说明：Predict using an ensemble of decision trees.",
  "description_source": "official_document",
  "official_document": "orange3-master/doc/visual-programming/source/widgets/model/randomforest.md",
  "params": [
    {
      "name": "n_estimators",
      "type": "int",
      "default": 10,
      "description": "集成模型中使用的基学习器数量；对树模型通常表示树的数量。",
      "description_source": "official_document"
    }
  ]
}
```

`controlAreaVisible`、`savedWidgetGeometry`、`selection` 等参数属于 Orange 画布或组件界面状态，不一定影响算法结果；其描述中会明确指出这一点。

Agent 只能使用当前任务允许并且存在于该文件中的组件和参数。组件名称应使用 Orange 的真实名称，例如：

```text
File
Select Columns
Random Forest
Test and Score
Predictions
Confusion Matrix
Save Data
k-Means
```

更换 Orange 版本或安装组件扩展后，可以重新生成定义：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\Orange3-3.40.0\Orange\python.exe `
  .\workflow_graph_edits\orange3\generate_widgets.py
```

## 五、Agent 构建工作流示例

```python
import json

from workflow_graph_edits.orange3 import (
    add_edge,
    add_node,
    configure_workflow,
    get_workflow,
    update_node_params,
)

WORKFLOW_PATH = r"D:\workflows\classification.ows"

configure_workflow(
    relevant_widgets_names=[
        "File",
        "Select Columns",
        "Random Forest",
        "Test and Score",
    ],
    workflow_path=WORKFLOW_PATH,
)

before = get_workflow()
print(json.dumps(before, ensure_ascii=False))

file_node = add_node("File", "读取分类数据")
select_node = add_node("Select Columns", "选择特征与目标变量")
forest_node = add_node("Random Forest", "随机森林模型")
score_node = add_node("Test and Score", "交叉验证模型评估")

update_node_params(
    forest_node,
    "Random Forest",
    {
        "n_estimators": 100,
        "use_max_depth": True,
        "max_depth": 8,
    },
)

add_edge(file_node, select_node)
add_edge(select_node, forest_node)
add_edge(select_node, score_node)
add_edge(forest_node, score_node)

after = get_workflow()
print(json.dumps(after, ensure_ascii=False, indent=2))
```

推荐 Agent 按以下顺序工作：

1. 调用 `configure_workflow` 指定当前工作流和允许组件；
2. 调用 `get_workflow` 获取修改前状态；
3. 根据返回的节点和边 ID 执行增删改操作；
4. 每个节点创建后，根据需要调用 `update_node_params`；
5. 调用 `add_edge` 连接兼容节点；
6. 再次调用 `get_workflow` 检查最终结果。

## 六、转换为 UniPlore 工作流 JSON

### 6.1 能否全部一一对应

不能全量一一对应，并且不能只根据组件名称判断两个工作流相同。

当前便携版 Orange3-3.40.0 安装了 104 个非废弃组件，但
`orange3/widgets.json` 仅暴露其中能够映射到 UniPlore 的 33 个组件。
UniPlore 参考定义有 51 个组件。对照结果如下：

- 26 个组件名称完全相同；
- 7 个组件功能对应，但名称不同；
- 合计 33 个组件可以进行组件级转换；
- 18 个 UniPlore 组件在当前 Orange 中没有等价实现；
- Orange 其余 71 个组件在 UniPlore 中没有对应项。

7 个异名组件映射为：

| Orange3 | UniPlore |
| --- | --- |
| `Correlations` | `Correlogram` |
| `Formula` | `Feature Constructor` |
| `Gradient Boosting` | `Gradient Boosting Decision Tree` |
| `k-Means` | `K-Means` |
| `kNN` | `KNN` |
| `Line Plot` | `Line Chart` |
| `Test and Score` | `Test Score` |

当前 Orange 没有等价实现的 18 个 UniPlore 组件为：

```text
ARIMA Model
AutoML Training
Change Domain
Image
Image Classification
Infer
Interpolate
LightGBM
Object Detection
One Hot Encoder
Segmentation
Select Best N Attributes
Text
Text Classification
Train Log
Translation
VAR Model
XGBoost
```

组件能够对应不代表参数能够完全对应。两个平台使用的参数名称、枚举值和默认值可能不同，例如：

- Orange `Test and Score` 对应 UniPlore `Test Score`；
- Orange `SVM.kernel_type=2` 对应 UniPlore `kernel="rbf"`；
- Orange `kNN.metric_index=0` 对应 UniPlore `metric="euclidean"`；
- Orange `Data Sampler.sampleSizePercentage` 对应 UniPlore `percentage`；
- Orange `Random Forest.max_features` 是整数，UniPlore 中定义为字符串；
- `File`、`SQL Table`、`Select Columns` 等组件的两端参数模型不同，不能无损转换。

UniPlore 是只读参考，转换实现不会修改 `workflow_graph_edits/uniplore` 中的任何代码或组件定义。

### 6.2 转换函数

使用 `convert_to_uniplore_workflow` 将 `get_workflow()` 返回的 Orange JSON 转换为 UniPlore 组件名称和参数：

```python
import json

from workflow_graph_edits.orange3 import (
    convert_to_uniplore_workflow,
    get_workflow,
)

orange_workflow = get_workflow()
uniplore_workflow = convert_to_uniplore_workflow(orange_workflow)

print(json.dumps(uniplore_workflow, ensure_ascii=False, indent=2))
```

函数会：

- 保留节点 ID、边 ID、中文节点名称和图拓扑；
- 将 Orange 组件名转换成 UniPlore 组件名；
- 将参数名、枚举索引和参数值转换成 UniPlore 定义；
- 补齐 UniPlore `widgets.json` 中定义的默认参数；
- 不修改传入的 Orange 工作流对象；
- 遇到没有 UniPlore 对应项的 Orange 组件时抛出 `WorkflowConversionError`。

默认情况下，每个转换后的节点会增加 `matched` 字段：

```json
{
  "node_id": "4",
  "widget_name": "Random Forest",
  "node_name": "随机森林模型",
  "node_params": {
    "n_estimators": 80,
    "use_max_features": true,
    "max_features": "6",
    "use_random_state": true,
    "use_max_depth": true,
    "max_depth": 9,
    "use_min_samples_split": true,
    "min_samples_split": 4
  },
  "matched": true
}
```

`matched` 的含义：

- `true`：该节点的 UniPlore 参数可以从 Orange 参数语义化转换；
- `false`：组件可以对应，但至少有一个 UniPlore 参数无法从 Orange 状态无损推导，转换结果使用了 UniPlore 默认值。

当前固定为有损参数映射的组件包括：

```text
AdaBoost
Continuize
Discretize
Edit Domain
File
Formula / Feature Constructor
Predictions
SQL Table
Select Columns
```

部分组件只在特定参数值下能够无损转换。例如，Orange `Gradient Boosting` 只有 `method_index=0` 能对应 UniPlore 的 `Gradient Boosting Decision Tree`。

如果需要得到与 UniPlore `get_workflow()` 完全相同的节点字段结构，可以关闭匹配标记：

```python
uniplore_workflow = convert_to_uniplore_workflow(
    orange_workflow,
    include_matched=False,
)
```

如果两端工作流的节点 ID 和边 ID 来自不同系统，即使组件、参数和拓扑相同，两个 JSON 也不会直接相等。此时应根据唯一的 `node_name` 对齐节点，再比较：

1. 转换后的 `widget_name`；
2. 完整 `node_params`；
3. 以节点名称表示的边连接关系。

显式组件映射表也可以直接导入：

```python
from workflow_graph_edits.orange3 import ORANGE_TO_UNIPLORE_WIDGETS
```

## 七、打开 Orange 画布

工作流构建完成后，使用 `open_canvas` 在 Orange GUI 中打开指定 `.ows` 文件：

```python
from workflow_graph_edits.orange3 import open_canvas

WORKFLOW_PATH = r"D:\workflows\classification.ows"

process_id = open_canvas(WORKFLOW_PATH)
print(f"Orange 进程 PID：{process_id}")
```

`open_canvas` 会：

- 检查文件是否存在；
- 检查扩展名是否为 `.ows`；
- 检查文件是否具有 Orange 工作流的 `scheme` 根节点；
- 使用便携版 `Orange\pythonw.exe -Psm Orange.canvas` 启动画布；
- 强制子进程使用 Windows 图形后端，避免继承测试环境中的 `QT_QPA_PLATFORM=offscreen`；
- 立即返回 Orange 进程 PID，不等待用户关闭窗口。

该函数与 `configure_workflow` 相互独立，因此既可以打开当前正在编辑的工作流，也可以打开其他有效 `.ows` 文件。

如需指定另一套便携版 Orange：

```python
process_id = open_canvas(
    workflow_path=r"D:\workflows\classification.ows",
    orange_directory=r"D:\tools\Orange3-3.40.0",
)
```

如果在 PyCharm Console、Jupyter 等长期运行的解释器中更新了本项目代码，需要重新启动解释器或重新加载模块后再调用，否则可能仍在使用旧版函数。

## 八、测试

测试六个工作流操作、持久化及重新加载：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONDONTWRITEBYTECODE = "1"
.\Orange3-3.40.0\Orange\python.exe `
  .\workflow_graph_edits\orange3\test_existing_workflow.py
```

## Agent 参数与 UniPlore 参数对应表

`widgets.json` 只保留会影响数据源、数据变换、算法、评估、输出或
可视化语义的参数。画布几何、自动提交开关、临时选区、对话框状态、
图形渲染缓存和脚本编辑器布局不会暴露给 Agent。

组件名称和逐参数转换规则可以直接导入：

```python
from workflow_graph_edits.orange3 import (
    ORANGE_TO_UNIPLORE_PARAMETER_MAP,
    ORANGE_TO_UNIPLORE_WIDGETS,
)
```

`ORANGE_TO_UNIPLORE_PARAMETER_MAP` 以 Orange 组件名为第一层键，以
UniPlore 参数名为第二层键。每条规则会明确记录直接复制、重命名、
枚举转换、多参数合成、固定值或使用 UniPlore 默认值。

例如，SQL Table 会将 `selected_backend`、`database`、`schema` 和
`table` 合成为 UniPlore 的 `data_description`：

```text
PostgreSQL table mlagent.public.abalone_test
```

运行参数精简与转换测试：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\Orange3-3.40.0\Orange\python.exe `
  .\workflow_graph_edits\orange3\test_conversion.py
```

运行 Test and Score PostgreSQL 集成测试：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\Orange3-3.40.0\Orange\python.exe `
  .\workflow_graph_edits\orange3\test_test_and_score_results.py
```

工作流操作规则不满足时会抛出 `WorkflowActionError`；转换遇到不支持的组件、参数或无效图结构时会抛出 `WorkflowConversionError`。调用 Agent 应捕获异常并根据错误信息修正输入。
