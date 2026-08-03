"""基于 Orange3 Canvas Core API 的工作流动作实现。

本模块负责发现 Orange 组件、加载和持久化 ``.ows`` 工作流、增删节点与边，
以及运行 Test and Score 组件并将评估结果转换成可 JSON 序列化的数据。
"""

from __future__ import annotations  # 推迟类型注解求值，避免运行时解析前向引用。

import copy  # 深拷贝节点参数，防止调用方后续修改影响内部状态。
import hashlib  # 根据边的端点信息生成稳定且可复现的边 ID。
import itertools  # 生成连续递增的节点 ID。
import json  # 读取组件定义并校验动作参数是否可 JSON 序列化。
import math  # 判断评估分数和超时时间是否为有限数值。
import os  # 拼接跨平台路径并执行原子文件替换。
import re  # 校验节点名称中是否包含中文字符。
import time  # 计算执行截止时间并在事件循环中短暂休眠。
import uuid  # 为临时工作流文件生成无冲突名称。
import warnings  # 在评估期间局部过滤已知的第三方库兼容性警告。
from pathlib import Path  # 以对象方式解析和操作工作流及组件定义路径。
from typing import Any, Iterable, Mapping  # 描述动态 Orange 对象和可迭代组件名称。
from xml.etree import ElementTree  # 从 OWS XML 中恢复序列化节点 ID。

# The portable Orange runtime must load the embedding/PyTorch stack before
# Orange's Qt stack on Windows, otherwise torch's native DLLs can fail to load.
from agents.data_loader import resolve_dataset
from AnyQt.QtWidgets import QApplication  # 创建并驱动 Orange 控件依赖的 Qt 事件循环。
from Orange.canvas.config import Config  # 获取 Orange 官方组件的发现入口。
from Orange.evaluation import (
    AUC,  # 分类任务的 ROC 曲线下面积。
    CA,  # 分类准确率。
    F1,  # 分类任务的 F1 分数。
    LogLoss,  # 分类任务的对数损失。
    MAE,  # 回归任务的平均绝对误差。
    MAPE,  # 回归任务的平均绝对百分比误差。
    MatthewsCorrCoefficient,  # 分类任务的 Matthews 相关系数。
    MSE,  # 回归任务的均方误差。
    Precision,  # 分类任务的精确率。
    R2,  # 回归任务的决定系数。
    Recall,  # 分类任务的召回率。
    RMSE,  # 回归任务的均方根误差。
    SMAPE,  # 回归任务的对称平均绝对百分比误差。
)
from Orange.evaluation.scoring import CVRMSE, Specificity  # 补充变异系数 RMSE 和特异度。
from Orange.widgets.evaluate.owtestandscore import (  # 复用 Test and Score 的结果工具。
    learner_name,  # 从学习器对象中取得供用户阅读的名称。
    results_merge,  # 将多个学习器的独立评估结果合并为一个 Results。
)
from Orange.widgets.data.owsql import OWSql
from Orange.widgets.utils.filedialogs import RecentPath  # 把 JSON 文件路径转换为 File 控件要求的原生路径对象。
from orangecanvas.registry import (  # 描述并索引 Orange Canvas 组件。
    WidgetDescription,  # 单个组件的注册信息和输入输出通道定义。
    WidgetRegistry,  # 保存所有已发现组件的注册表。
)
from orangecanvas.scheme import Scheme, SchemeLink, SchemeNode  # 操作工作流图、边和节点。
from orangewidget.workflow.widgetsscheme import WidgetsScheme  # 创建可实例化并执行控件的运行时图。
from sklearn.exceptions import UndefinedMetricWarning  # 标识无预测样本时的指标定义警告。
import config as project_config
from ml_platform.action import Action, ActionConfigurationError, ActionError
from utils.utils import get_project_root  # 获取项目根目录以构造默认资源路径。
import logging

logging.getLogger("orangecanvas.scheme.readwrite").setLevel(logging.ERROR)


# 将平台自定义信息放在独立属性键下，避免与 Orange 控件自身属性冲突。
_NODE_METADATA_KEY = "__ml_platform__"
# 指向声明平台可用组件及其参数模式的 JSON 文件。
WIDGETS_PATH = os.path.join(get_project_root(), r"ml_platform/orange3/widgets.json")

class WorkflowActionError(ActionError):
    """表示 Orange 工作流动作因输入或图状态不合法而无法完成。

    该异常把底层 Orange、XML 和参数校验错误统一成动作层可识别的失败类型。
    """


def _json_score(value: Any) -> float | None:
    """把单个评估值转换为 JSON 可安全表示的有限浮点数。

    Args:
        value: Orange 或 NumPy 返回的数值对象。

    Returns:
        转换后的有限 ``float``；若值为 NaN 或无穷大则返回 ``None``。
    """
    number = float(value)  # 统一 NumPy 标量、整数和浮点数的输出类型。
    return number if math.isfinite(number) else None  # JSON 不允许标准 NaN/Infinity。

def _evaluation_scores(results: Any) -> tuple[dict[str, list[float | None]], dict[str, str]]:
    """根据目标变量类型计算一组标准 Orange 评估指标。

    Args:
        results: 合并后的 Orange ``Results`` 对象，可包含多个学习器的预测。

    Returns:
        二元组：第一项为“指标名 -> 各学习器分数”，第二项为计算失败的指标及错误。
    """
    class_var = results.domain.class_var  # 目标变量决定使用分类指标还是回归指标。
    if class_var.is_discrete:  # 离散目标表示当前为分类任务。
        scorers = {  # 为分类任务构造统一的指标调用表。
            "AUC": lambda value: AUC(value),  # 计算每个学习器的 AUC。
            "CA": lambda value: CA(value),  # 计算每个学习器的分类准确率。
            "F1": lambda value: F1(value, average="weighted"),  # 按类别样本量加权。
            "Precision": lambda value: Precision(
                value, average="weighted"  # 多分类精确率按类别样本量加权。
            ),
            "Recall": lambda value: Recall(value, average="weighted"),  # 加权召回率。
            "LogLoss": lambda value: LogLoss(value),  # 基于预测概率计算对数损失。
            "Specificity": lambda value: Specificity(
                value, average="weighted"  # 多分类特异度按类别样本量加权。
            ),
            "MCC": lambda value: MatthewsCorrCoefficient(value),  # 计算 MCC。
        }
    else:  # 连续目标表示当前为回归任务。
        scorers = {  # 为回归任务构造统一的指标调用表。
            "MSE": lambda value: MSE(value),  # 均方误差。
            "RMSE": lambda value: RMSE(value),  # 均方根误差。
            "MAE": lambda value: MAE(value),  # 平均绝对误差。
            "MAPE": lambda value: MAPE(value),  # 平均绝对百分比误差。
            "SMAPE": lambda value: SMAPE(value),  # 对称平均绝对百分比误差。
            "R2": lambda value: R2(value),  # 决定系数。
            "CVRMSE": lambda value: CVRMSE(value),  # 经目标均值归一化的 RMSE。
        }

    scores: dict[str, list[float | None]] = {}  # 收集成功计算的全部指标列。
    errors: dict[str, str] = {}  # 单独记录失败指标，不中断其他指标的计算。
    for name, scorer in scorers.items():  # 依次执行当前任务类型支持的指标。
        try:  # 某些指标可能因类别数量或目标分布而没有定义。
            scores[name] = [  # 将一个指标下的每个学习器分数转成 JSON 安全值。
                _json_score(item) for item in scorer(results)
            ]
        except Exception as exc:  # A metric can be undefined for some targets.
            errors[name] = f"{type(exc).__name__}: {exc}"  # 保留异常类型与说明。
    return scores, errors  # 同时返回可用分数和非致命指标错误。

def _resampling_description(widget: Any) -> dict[str, Any]:
    """读取 Test and Score 控件配置并生成可 JSON 序列化的重采样说明。

    Args:
        widget: 已实例化的 Orange Test and Score 控件。

    Returns:
        包含重采样方法、折数、重复次数、抽样比例及分层设置的字典。
    """
    names = {  # 把控件内部枚举值映射成人类可读的评估方法名称。
        widget.KFold: "Cross Validation",  # 普通 K 折交叉验证。
        widget.FeatureFold: "Cross Validation by Feature",  # 按特征分组交叉验证。
        widget.ShuffleSplit: "Random Sampling",  # 重复随机抽样。
        widget.LeaveOneOut: "Leave One Out",  # 留一法验证。
        widget.TestOnTrain: "Test on Training Data",  # 在训练数据上测试。
        widget.TestOnTest: "Test on Test Data",  # 使用独立测试数据。
    }
    description: dict[str, Any] = {  # 所有重采样方式都具备的基础字段。
        "method": names.get(widget.resampling, str(widget.resampling)),  # 展示名称。
        "resampling": widget.resampling,  # 保留 Orange 原始枚举值。
    }
    if widget.resampling == widget.KFold:  # K 折方式需要补充折数和分层开关。
        description.update(  # 把仅适用于 K 折的字段合并进结果。
            {
                "folds": widget.NFolds[widget.n_folds],  # 将选项索引换算成真实折数。
                "stratified": widget.cv_stratified,  # 是否保持各折类别比例。
            }
        )
    elif widget.resampling == widget.ShuffleSplit:  # 随机抽样有独立的配置字段。
        description.update(  # 把随机抽样专用设置合并进结果。
            {
                "repeats": widget.NRepeats[widget.n_repeats],  # 实际重复次数。
                "sample_size_percent": widget.SampleSizes[
                    widget.sample_size  # 将样本比例选项索引换成百分比。
                ],
                "stratified": widget.shuffle_stratified,  # 是否执行分层随机抽样。
            }
        )
    return description  # 返回适配当前重采样方式的完整说明。

def _serialize_test_and_score_widget(
    node_id: str,
    node: SchemeNode,
    widget: Any,
) -> dict[str, Any]:
    """把一个 Test and Score 控件的运行状态与指标序列化成字典。

    Args:
        node_id: 平台对外使用的 Test and Score 节点 ID。
        node: 对应的运行时 Orange SchemeNode。
        widget: 已完成或部分完成评估的 Test and Score 控件实例。

    Returns:
        包含评估状态、目标信息、学习器分数和失败详情的 JSON 就绪字典。
    """
    completed = [  # 仅保留已经产生结果对象的学习器插槽。
        slot  # 返回满足条件的学习器插槽本身。
        for slot in widget.learners.values()  # 遍历控件接收到的全部学习器。
        if slot.results is not None  # 排除尚未完成的学习器。
    ]
    successful = [  # 从已完成插槽中筛出评估成功的学习器。
        slot for slot in completed if slot.results.success  # success 表示未抛训练异常。
    ]
    failed = [  # 将失败学习器的异常转换为可展示、可序列化的信息。
        {
            "learner_name": learner_name(slot.learner),  # 失败学习器的名称。
            "error_type": type(slot.results.exception).__name__,  # 异常类型名。
            "error": str(slot.results.exception),  # 异常文本内容。
        }
        for slot in completed  # 检查全部已完成插槽。
        if not slot.results.success  # 只序列化失败的插槽。
    ]
    if not successful:  # 没有任何学习器成功时直接返回整体失败结果。
        return {
            "node_id": node_id,  # 对外节点 ID。
            "node_name": node.title,  # 工作流画布中的节点标题。
            "status": "failed",  # 明确标识整体失败。
            "evaluation": _resampling_description(widget),  # 本次评估配置。
            "rows": 0,  # 没有可用预测结果时行数记为零。
            "target": None,  # 无成功结果时无法可靠读取目标信息。
            "learners": [],  # 成功学习器列表为空。
            "failed_learners": failed,  # 返回每个失败学习器的原因。
            "score_errors": {},  # 未进入指标计算，因此没有指标级错误。
        }

    merged = results_merge(  # 合并成功学习器的结果以便一次计算全部指标。
        [slot.results.value for slot in successful]  # 提取异步结果包装器中的 Results。
    )
    names = [  # 生成与合并结果列顺序严格一致的学习器名称列表。
        learner_name(slot.learner) for slot in successful
    ]
    merged.learner_names = names  # 把名称写入 Results 供下游 Orange API 使用。
    score_columns, score_errors = _evaluation_scores(merged)  # 计算标准指标。

    learners = []  # 最终按学习器组织指标，而不是按指标组织学习器列。
    for index, name in enumerate(names):  # 使用索引关联名称与每个指标的对应列。
        learners.append(  # 追加一个学习器的完整分数记录。
            {
                "learner_name": name,  # 当前学习器的显示名称。
                "scores": {  # 把各指标同一列的值聚合到当前学习器下。
                    score_name: values[index]  # 读取当前学习器的指标值。
                    for score_name, values in score_columns.items()  # 遍历全部指标。
                },
            }
        )

    class_var = merged.domain.class_var  # 从成功结果的数据域取得目标变量。
    target = {  # 构造所有目标类型都具备的基础说明。
        "name": class_var.name,  # 目标字段名。
        "type": (
            "discrete" if class_var.is_discrete else "continuous"  # 分类或连续目标。
        ),
    }
    if class_var.is_discrete:  # 分类目标还需要暴露可能的类别取值。
        target["values"] = list(class_var.values)  # 转为普通列表以支持 JSON。

    return {  # 汇总节点级评估结果。
        "node_id": node_id,  # 对外节点 ID。
        "node_name": node.title,  # 画布节点标题。
        "status": "partial" if failed else "completed",  # 区分部分成功和全部成功。
        "evaluation": _resampling_description(widget),  # 重采样设置。
        "rows": len(merged.actual),  # 实际参与评估的目标值数量。
        "target": target,  # 目标字段信息。
        "learners": learners,  # 成功学习器及其指标。
        "failed_learners": failed,  # 失败学习器及异常。
        "score_errors": score_errors,  # 非致命的指标级计算错误。
    }

class PlatformAction(Action):
    """加载、编辑、执行并持久化一个明确指定的 Orange 工作流。"""

    def __init__(
        self,
        platform_config: Mapping[str, Any],
        *,
        relevant_widgets_names: Iterable[str] | None = None,
        workflow_path: str | os.PathLike[str] | None = None,
        title: str = "LLM Generated Workflow",
        widgets_path: str | os.PathLike[str] = WIDGETS_PATH,
    ) -> None:
        """初始化动作代理及其内存工作流。

        Args:
            relevant_widgets_names: 当前任务允许使用的组件名称；默认没有授权组件。
            workflow_path: 要加载或新建的 ``.ows`` 工作流路径。
            title: 新工作流标题，或为无标题工作流补充的标题。
            widgets_path: 平台组件与参数定义 JSON 的路径。

        Raises:
            Exception: 组件定义无效或现有工作流无法加载。
        """
        configured_workflow_path = platform_config.get("workflow_path")
        selected_workflow_path = (
            workflow_path if workflow_path is not None else configured_workflow_path
        )
        if not isinstance(selected_workflow_path, (str, os.PathLike)) or not str(
            selected_workflow_path
        ).strip():
            raise ActionConfigurationError(
                "ORANGE3_CONFIG['workflow_path'] must be a non-empty path"
            )
        selected_workflow_path = Path(
            selected_workflow_path
        ).expanduser()
        if selected_workflow_path.suffix.lower() != ".ows":
            raise ActionConfigurationError(
                "ORANGE3_CONFIG['workflow_path'] must point to an .ows file"
            )
        super().__init__(
            platform_config,
            platform_name="orange3",
            widgets_path=widgets_path,
            relevant_widgets_names=list(relevant_widgets_names or []),
        )
        # Orange 控件必须依附 Qt 应用；进程已有实例时直接复用。
        self._app = QApplication.instance() or QApplication([])
        # 发现当前 Orange 环境中真实安装并可用的组件。
        self.registry = self._discover_widgets()
        # 解析绝对路径，避免工作目录变化影响后续读写。
        self.widgets_path = Path(widgets_path).resolve()
        # 加载平台层声明的组件参数模式。
        self._widget_definitions = self._load_widget(self.widgets_path)
        # 保留与 Uniplore 动作对象一致的组件定义列表接口。
        self.widgets = list(self._widget_definitions.values())
        # 缓存组件名列表，便于调用方做组件检索。
        self.widgets_name = list(self._widget_definitions)
        # 按唯一显示名称索引 Orange 注册信息。
        self._widgets_by_name = self._index_widgets_by_name(self.registry)
        # 建立 Orange Python 限定名到平台外部组件名的反向映射。
        self._external_name_by_qualified_name = {
            description.qualified_name: external_name  # 每个组件的稳定限定名。
            for external_name, description in self._widgets_by_name.items()  # 遍历唯一组件。
        }
        # 按首次出现顺序去重，同时把 None 规范为空列表。
        self.relevant_widgets_names = list(dict.fromkeys(relevant_widgets_names or []))
        self.workflow_path = selected_workflow_path.resolve()
        self.workflow_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.workflow_path.is_file():  # 文件不存在时创建最小空工作流。
            scheme = Scheme(title=title)  # 使用调用方标题创建临时图对象。
            scheme.save_to(  # 立即落盘，确保后续加载路径存在。
                str(self.workflow_path),  # Orange API 接受字符串路径。
                pretty=True,  # 输出易读的格式化 XML。
                pickle_fallback=False,  # 新空图无需不安全的 pickle 回退。
            )
        self.scheme = Scheme()  # 创建负责编辑的内存工作流图。
        self._nodes: dict[str, SchemeNode] = {}  # 平台节点 ID 到 Orange 节点。
        self._edges: dict[str, SchemeLink] = {}  # 平台边 ID 到 Orange 连线。
        self._node_widget_names: dict[str, str] = {}  # 节点 ID 到外部组件名。
        self._node_params: dict[str, dict[str, Any]] = {}  # 节点 ID 到完整参数。
        self._next_position_index = 0  # 新节点自动布局序号。
        self.pg_config = dict(getattr(project_config, "PG_CONFIG", {}))
        self._load_current_workflow()  # 从磁盘恢复图及以上索引。
        if any(
            widget_name == "SQL Table"
            for widget_name in self._node_widget_names.values()
        ):
            self._store_postgres_credentials(require_complete=False)
        if not self.scheme.title and title:  # 仅在原文件无标题时应用默认标题。
            self.scheme.title = title  # 不覆盖用户已经设置的标题。

    def execute_command(
        self,
        command_name: str,
        args: dict[str, Any] | None = None,
    ) -> Any:
        """按名称分发动作，并把异常转换成与 Uniplore 一致的文本结果。

        Args:
            command_name: 要调用的 ``Actions`` 方法名。
            args: 原样传给目标方法的动作参数字典。

        Returns:
            目标方法返回值；方法不存在或执行失败时返回错误文本。
        """
        return super().execute_command(command_name, args)

    @staticmethod
    def _discover_widgets() -> WidgetRegistry:
        """发现当前 Orange 环境中的组件并返回注册表。

        Returns:
            已加载所有 Orange 入口点组件的 ``WidgetRegistry``。
        """
        registry = WidgetRegistry()  # 创建空注册表接收发现结果。
        discovery = Config.widget_discovery(registry)  # 创建 Orange 发现器。
        discovery.run(Config.widgets_entry_points())  # 扫描已安装组件入口点。
        return registry  # 返回供节点创建和工作流加载使用的注册表。

    @staticmethod
    def _load_widget(
        widgets_path: Path,
    ) -> dict[str, dict[str, Any]]:
        """读取并验证平台组件定义文件。

        Args:
            widgets_path: UTF-8 编码的组件 JSON 文件路径。

        Returns:
            以 ``widget_name`` 为键的组件定义字典。

        Raises:
            Exception: 文件不存在、JSON 无效、名称重复或参数模式不合法。
        """
        try:  # 把文件读取和 JSON 解析错误转换为更明确的配置错误。
            data = json.loads(widgets_path.read_text(encoding="utf-8"))  # 解析 JSON。
        except FileNotFoundError:  # 给出缺失文件的完整路径。
            raise Exception(f"组件定义文件不存在：{widgets_path}")
        except json.JSONDecodeError:  # 区分文件存在但内容不是合法 JSON。
            raise Exception(f"组件定义文件不是有效 JSON：{widgets_path}")
        definitions: dict[str, dict[str, Any]] = {}  # 收集去重后的组件。
        for widget in data:  # 逐个验证顶层数组元素。
            if not isinstance(widget, dict) or not isinstance(
                widget.get("widget_name"), str  # 名称必须存在且为字符串。
            ):
                raise Exception("widgets.json 中的每个组件都必须包含 widget_name")
            name = widget["widget_name"]  # 读取后续索引使用的外部名称。
            if name in definitions:  # 重名会使动作无法唯一解析组件。
                raise Exception(f"widgets.json 中存在重复组件：{name}")
            params = widget.get("params", [])  # 无参数组件默认使用空列表。
            if not isinstance(params, list):  # 参数集合必须可有序遍历。
                raise Exception(f"组件 {name} 的 params 必须是数组")
            for param in params:  # 校验每一个参数模式。
                if (
                    not isinstance(param, dict)  # 参数定义本身必须是对象。
                    or not isinstance(param.get("name"), str)  # 必须有字符串名称。
                    or "default" not in param  # 必须提供完整参数所需的默认值。
                ):
                    raise Exception(f"组件 {name} 包含无效参数定义")
            definitions[name] = widget  # 验证通过后按名称保存。
        return definitions  # 返回供参数默认值和类型校验使用的索引。

    @staticmethod
    def _index_widgets_by_name(
        registry: WidgetRegistry,
    ) -> dict[str, WidgetDescription]:
        """按唯一显示名称索引注册组件并排除废弃或重名项。

        Args:
            registry: Orange 组件发现得到的注册表。

        Returns:
            只包含非废弃且名称唯一组件的映射。
        """
        by_name: dict[str, WidgetDescription] = {}  # 暂存首次出现的名称。
        ambiguous: set[str] = set()  # 记录出现多次、无法唯一解析的名称。

        for description in registry.widgets():  # 遍历所有已发现组件。
            if description.category == "Orange Obsolete":  # 排除官方废弃组件。
                continue  # 不允许新工作流继续创建废弃节点。
            if description.name in by_name:  # 同一显示名再次出现即有歧义。
                ambiguous.add(description.name)  # 记录待删除名称。
            else:  # 首次出现时先保存候选描述。
                by_name[description.name] = description

        for name in ambiguous:  # 清理所有无法唯一定位的名称。
            by_name.pop(name, None)  # 使用默认值避免重复清理报错。
        return by_name  # 返回安全的名称索引。

    @staticmethod
    def _as_id(value: Any, field_name: str) -> str:
        """把外部节点或边 ID 规范成非空字符串。

        Args:
            value: 调用方提供的 ID 值。
            field_name: 用于错误消息的字段名。

        Returns:
            规范化后的字符串 ID。

        Raises:
            WorkflowActionError: 值为布尔值、None 或空字符串。
        """
        if isinstance(value, bool) or value is None:  # bool 不应被当作整数 ID。
            raise WorkflowActionError(f"{field_name} 必须是有效的节点或边 ID")
        value = str(value)  # 兼容数字 ID 和字符串 ID。
        if not value:  # 空字符串无法索引图元素。
            raise WorkflowActionError(f"{field_name} 不能为空")
        return value  # 后续索引统一使用字符串键。

    def _resolve_node(self, node_id: Any) -> tuple[str, SchemeNode]:
        """规范节点 ID 并返回对应的 Orange 节点。

        Args:
            node_id: 外部节点 ID。

        Returns:
            ``(规范化 ID, SchemeNode)`` 二元组。

        Raises:
            WorkflowActionError: ID 无效或节点不存在。
        """
        normalized = self._as_id(node_id, "node_id")  # 统一键类型。
        try:  # 字典查找失败时转换成动作层异常。
            return normalized, self._nodes[normalized]  # 同时返回 ID 与节点。
        except KeyError as exc:  # 对调用方隐藏内部字典实现。
            raise WorkflowActionError(f"节点 ID {normalized} 不存在") from exc

    def _resolve_widget(self, widget_name: str) -> WidgetDescription:
        """校验组件授权并解析唯一的 Orange 组件描述。

        Args:
            widget_name: 平台组件显示名称。

        Returns:
            可传给 ``Scheme.new_node`` 的组件描述。

        Raises:
            WorkflowActionError: 组件未授权、未安装或名称不唯一。
        """
        if widget_name not in self.relevant_widgets_names:  # 执行任务级组件白名单。
            raise WorkflowActionError(
                f"组件 {widget_name!r} 不在当前任务可用组件中；"
                f"可用组件为：{self.relevant_widgets_names}"
            )
        try:  # 查找发现阶段构造的唯一组件描述。
            return self._widgets_by_name[widget_name]  # 返回真实 Orange 注册对象。
        except KeyError as exc:  # 未安装和重名组件都会被排除在索引外。
            raise WorkflowActionError(
                f"组件名 {widget_name!r} 不能唯一对应一个 Orange3 组件"
            ) from exc

    def _default_params(self, widget_name: str) -> dict[str, Any]:
        """返回指定组件的一份独立默认参数字典。

        Args:
            widget_name: 已在平台 JSON 中声明的组件名。

        Returns:
            参数名到深拷贝默认值的映射。
        """
        return {  # 每个节点必须拥有独立的可变默认值。
            param["name"]: copy.deepcopy(param["default"])  # 复制列表和字典。
            for param in self._widget_definitions[widget_name]["params"]  # 遍历模式。
        }

    def _param_definitions(self, widget_name: str) -> dict[str, dict[str, Any]]:
        """按参数名索引指定组件的参数模式。

        Args:
            widget_name: 已在平台 JSON 中声明的组件名。

        Returns:
            参数名到完整参数定义的映射。
        """
        return {  # 将列表转成便于校验未知字段的字典。
            param["name"]: param  # 保留类型、默认值和描述等完整信息。
            for param in self._widget_definitions[widget_name]["params"]  # 遍历参数。
        }

    @staticmethod
    def _validate_param_type(
        widget_name: str,
        param_name: str,
        value: Any,
        expected_type: str | None,
    ) -> None:
        """根据 widgets.json 中的简单类型声明校验一个参数值。

        未识别的类型声明不做限制，以兼容 Orange 扩展组件的自定义模式。
        """
        validators = {  # JSON 类型名到 Python 运行时判断函数。
            "str": lambda item: isinstance(item, str),  # 只接受字符串。
            "int": lambda item: isinstance(item, int)
            and not isinstance(item, bool),  # bool 是 int 子类，需显式排除。
            "bool": lambda item: isinstance(item, bool),  # 只接受布尔值。
            "float": lambda item: isinstance(item, (int, float))
            and not isinstance(item, bool),  # 浮点配置允许整数但不允许 bool。
            "list": lambda item: isinstance(item, list),  # 只接受数组。
            "dict": lambda item: isinstance(item, dict),  # 只接受对象。
        }
        validator = validators.get(expected_type)  # 未声明或扩展类型得到 None。
        if validator is not None and not validator(value):  # 仅执行已知规则。
            raise WorkflowActionError(
                f"组件 {widget_name} 的参数 {param_name} "
                f"必须符合 widgets.json 中定义的 {expected_type} 类型"
            )

    @staticmethod
    def _column_descriptor_name(descriptor: Any) -> str:
        """从 Orange 上下文字段描述或 Variable 对象中读取字段名。"""
        if isinstance(descriptor, tuple):  # 持久化上下文使用 (name, vartype)。
            return str(descriptor[0])  # 元组首项就是原始字段名。
        return str(descriptor.name)  # 运行时 Variable 直接暴露 name。

    @staticmethod
    def _column_descriptor_type(descriptor: Any) -> int:
        """把上下文字段描述或 Variable 对象转换成 Orange vartype 编码。"""
        if isinstance(descriptor, tuple):  # 已编码描述的第二项就是类型。
            return int(descriptor[1]) % 100  # 同时兼容旧的 100+ 类型编码。
        if descriptor.is_discrete:  # Orange 离散变量编码为 1。
            return 1
        if descriptor.is_continuous:  # 普通连续变量和时间变量均可用作数值列。
            return 4 if descriptor.is_time else 2  # 时间变量使用专用编码 4。
        if descriptor.is_string:  # 字符串变量编码为 3，只能放入 Meta 或 Ignored。
            return 3
        return 0  # 未知扩展类型按不可用类型处理。

    @classmethod
    def _resolve_select_columns_hints(
        cls,
        hints: dict[Any, tuple[str, int]],
        params: dict[str, Any],
    ) -> tuple[dict[Any, tuple[str, int]], dict[str, Any]]:
        """把三个公开参数解析成 Select Columns 的完整数据域角色映射。

        ``features`` 精确指定特征，``targets`` 指定唯一目标，
        ``ignores`` 显式指定要移除的字段。输入数据原有的 Meta 字段会自动
        保留，其余未被选择的普通字段自动进入 Orange 的 Ignored 区域。
        """
        descriptors = {  # 建立字段名到上下文描述对象的稳定索引。
            cls._column_descriptor_name(descriptor): descriptor
            for descriptor in hints
        }
        ordered_hints = sorted(  # 使用控件保存的角色索引恢复稳定顺序。
            hints.items(),
            key=lambda item: (
                ("attribute", "class", "meta", "available").index(
                    item[1][0]
                ),
                item[1][1],
            ),
        )
        descriptor_order = [  # 保留输入域的总体字段顺序。
            cls._column_descriptor_name(descriptor)
            for descriptor, _ in ordered_hints
        ]
        original_metas = [  # Meta 不作为公开参数，默认按原角色自动保留。
            cls._column_descriptor_name(descriptor)
            for descriptor, (role, _) in ordered_hints
            if role == "meta"
        ]

        features = params["features"]  # Features 必须是字段名数组。
        ignores = params["ignores"]  # Ignored 必须是字段名数组。
        target = params["targets"]  # Target 必须是唯一字段名字符串。
        claimed_by: dict[str, str] = {}  # 记录每个字段的公开参数归属。
        for parameter_name, names in (  # 检查数组内部及跨参数重复。
            ("features", features),
            ("ignores", ignores),
            ("targets", [target]),
        ):
            for name in names:
                if name in claimed_by:
                    raise WorkflowActionError(
                        f"Select Columns 字段 {name!r} 同时出现在 "
                        f"{claimed_by[name]} 和 {parameter_name} 中"
                    )
                claimed_by[name] = parameter_name

        unknown = sorted(set(claimed_by) - set(descriptors))  # 查找不存在的列。
        if unknown:
            raise WorkflowActionError(
                "Select Columns 输入数据中不存在以下字段："
                + ", ".join(unknown)
            )

        for role_name, names in (  # Features 与 Target 不能使用字符串变量。
            ("Features", features),
            ("Target", [target]),
        ):
            for name in names:
                descriptor_type = cls._column_descriptor_type(descriptors[name])
                if descriptor_type not in (1, 2, 4):
                    raise WorkflowActionError(
                        f"Select Columns 字段 {name!r} 不能作为 {role_name}；"
                        "字符串字段只能保留为 Meta 或放入 Ignored"
                    )

        metas = [  # 仅自动保留没有被三个公开参数重新分配的原始 Meta。
            name for name in original_metas if name not in claimed_by
        ]
        selected = set(features) | {target} | set(metas)  # 已进入输出域的字段。
        automatic_ignores = [  # 普通字段未被选中时自动移入 Ignored。
            name
            for name in descriptor_order
            if name not in selected and name not in ignores
        ]
        all_ignores = list(ignores) + automatic_ignores  # 显式顺序优先。
        ordered_names = {  # 构造 Orange 四个列表区域的最终顺序。
            "attribute": list(features),
            "class": [target],
            "meta": metas,
            "available": all_ignores,
        }
        resolved_hints = {  # 编码成控件 ContextSetting 使用的原生映射。
            descriptors[name]: (role, index)
            for role, names in ordered_names.items()
            for index, name in enumerate(names)
        }
        readable = {  # 只返回用户要求展示的三个关键参数。
            "ignores": all_ignores,
            "features": list(features),
            "targets": target,
        }
        return resolved_hints, readable

    @classmethod
    def _select_columns_params_from_properties(
        cls,
        properties: Any,
    ) -> dict[str, Any] | None:
        """从最新数据域上下文恢复三个可读的 Select Columns 参数。"""
        if not isinstance(properties, dict):  # 非字典属性没有上下文。
            return None
        contexts = properties.get("context_settings")  # 上下文按最近使用优先。
        if not isinstance(contexts, list) or not contexts:
            return None
        for context in contexts:  # 找到第一个包含字段角色的有效上下文。
            values = getattr(context, "values", None)  # Context 使用 values 属性。
            if not isinstance(values, dict):
                continue
            packed = values.get("domain_role_hints")  # 读取编码后的 ContextSetting。
            hints = (
                packed[0]
                if isinstance(packed, tuple)
                and len(packed) == 2
                and isinstance(packed[0], dict)
                else packed
            )
            if not isinstance(hints, dict):
                continue
            groups = {  # 收集并按角色内部索引排序。
                role: [
                    cls._column_descriptor_name(descriptor)
                    for descriptor, (stored_role, index) in sorted(
                        hints.items(), key=lambda item: item[1][1]
                    )
                    if stored_role == role
                ]
                for role in ("attribute", "class", "available")
            }
            return {
                "ignores": groups["available"],
                "features": groups["attribute"],
                "targets": (
                    groups["class"][0]
                    if len(groups["class"]) == 1
                    else ""
                ),
            }
        return None  # 没有可解码上下文时返回空状态。

    def _to_native_node_params(
        self,
        widget_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """把 Uniplore 风格公开参数转换成 Orange 控件原生设置。

        Args:
            widget_name: 当前节点对应的平台组件名称。
            params: 已合并默认值和历史值的完整公开参数。

        Returns:
            可直接合并进 ``SchemeNode.properties`` 的 Orange 原生属性。

        Raises:
            WorkflowActionError: 参数取值无法由对应 Orange 控件准确表达。
        """
        if widget_name == "SQL Table":
            table_name = params.get("table_name")
            if not isinstance(table_name, str) or not table_name.strip():
                raise WorkflowActionError(
                    "SQL Table 必须先通过 data_description 确认具体数据表"
                )
            required = ("host", "port", "database", "schema", "user", "password")
            missing = [
                name
                for name in required
                if self.pg_config.get(name) in (None, "")
            ]
            if missing:
                raise WorkflowActionError(
                    "PG_CONFIG 缺少以下配置：" + ", ".join(missing)
                )
            self._store_postgres_credentials(require_complete=True)
            return {
                "selected_backend": "PostgreSQL",
                "host": str(self.pg_config["host"]),
                "port": str(self.pg_config["port"]),
                "database": str(self.pg_config["database"]),
                "schema": str(self.pg_config["schema"]),
                "data_source": 0,
                "table": table_name.strip(),
                "sql": "",
                "guess_values": True,
                "materialize": False,
                "materialize_table_name": "",
                "__version__": 2,
            }

        if widget_name == "File":  # Uniplore File 只公开内置示例数据集。
            if params["type"].strip().lower() != "sample":
                raise WorkflowActionError(
                    "File 的 type 当前只能是 'sample'"
                )
            filename = params["filename"].strip()  # 示例名使用 csv 风格公开名称。
            if not filename or Path(filename).name != filename:
                raise WorkflowActionError(
                    "File 的 filename 必须是不含目录的示例数据集文件名"
                )
            if Path(filename).suffix.lower() != ".csv":
                raise WorkflowActionError(
                    "File 的 filename 必须使用 .csv 后缀"
                )
            declared_samples = {
                "heart_disease.csv",
                "housing.csv",
                "iris.csv",
                "titanic.csv",
                "zoo.csv",
            }
            if filename not in declared_samples:
                raise WorkflowActionError(
                    f"File 的 filename 必须是 widgets.json 声明的示例名之一"
                )
            import Orange  # 延迟导入仅用于定位便携环境的数据集目录。

            dataset_directory = (
                Path(Orange.__file__).resolve().parent / "datasets"
            )
            dataset_path = (  # Orange 内置样例实际以 tab 格式随包分发。
                dataset_directory / f"{Path(filename).stem}.tab"
            )
            if not dataset_path.is_file():
                available = sorted(  # 把真实可用 tab 文件转换回公开 csv 名。
                    sample
                    for sample in declared_samples
                    if (
                        dataset_directory
                        / f"{Path(sample).stem}.tab"
                    ).is_file()
                )
                raise WorkflowActionError(
                    f"File 示例数据集 {filename!r} 未随当前 Orange 安装提供；"
                    f"当前可用：{available}"
                )
            return {
                "source": 0,  # OWFile 使用本地 recent_paths 来源。
                "recent_paths": [
                    RecentPath(
                        str(dataset_path),  # 指向已安装的真实 tab 文件。
                        None,  # 内置绝对路径不依赖搜索前缀。
                        None,  # 绝对路径不需要相对路径。
                        title=filename,  # 界面继续展示 Uniplore 的 csv 名称。
                    )
                ],
                "recent_urls": [],  # 清理旧 URL 历史。
                "url": "",  # 同步清理旧 URL Setting。
                "sheet_names": {},  # 清理其他文件残留的工作表缓存。
                "context_settings": [],  # 清理旧文件的数据域编辑上下文。
            }

        if widget_name == "Select Columns":  # 字段列表需在数据上下文中单独转换。
            claimed_by: dict[str, str] = {}  # 提前校验无输入节点的参数冲突。
            for parameter_name in ("features", "ignores"):
                value = params[parameter_name]  # 两个多值参数必须是数组。
                if not isinstance(value, list):
                    raise WorkflowActionError(
                        f"Select Columns 的 {parameter_name} "
                        "必须是字段名数组"
                    )
                if any(
                    not isinstance(name, str) or not name.strip()
                    for name in value
                ):
                    raise WorkflowActionError(
                        f"Select Columns 的 {parameter_name} "
                        "只能包含非空字段名"
                    )
                for name in value:  # 同列表重复也视为角色冲突。
                    if name in claimed_by:
                        raise WorkflowActionError(
                            f"Select Columns 字段 {name!r} 同时出现在 "
                            f"{claimed_by[name]} 和 {parameter_name} 中"
                        )
                    claimed_by[name] = parameter_name
            target = params["targets"]  # Target 只接受一个非空字段名。
            if not isinstance(target, str) or not target.strip():
                raise WorkflowActionError(
                    "Select Columns 的 targets 必须配置为一个非空字段名"
                )
            if target in claimed_by:  # Target 不能同时属于 Features 或 Ignored。
                raise WorkflowActionError(
                    f"Select Columns 字段 {target!r} 同时出现在 "
                    f"{claimed_by[target]} 和 targets 中"
                )
            return {
                "ignore_new_features": False,  # 未列入 Features 的普通字段自动忽略。
                "use_input_features": False,  # 只采用本动作明确指定的 Features。
                "auto_commit": True,  # 参数更新后立即提交新的输出数据域。
                "__version__": 1,  # 当前 Select Columns 设置结构版本。
            }

        if widget_name == "Data Sampler":  # 公开名称与原生名称逐项转换。
            sampling_type = params["sampling_type"]
            if sampling_type not in range(4):
                raise WorkflowActionError(
                    "Data Sampler 的 sampling_type 必须是 0、1、2 或 3"
                )
            percentage = params["percentage"]
            if not 0 <= percentage <= 100:
                raise WorkflowActionError(
                    "Data Sampler 的 percentage 必须在 0 到 100 之间"
                )
            if params["n_samples"] < 0:
                raise WorkflowActionError(
                    "Data Sampler 的 n_samples 不能小于 0"
                )
            if params["n_samples"] > 2 ** 31 - 1:
                raise WorkflowActionError(
                    "Data Sampler 的 n_samples 不能超过 2147483647"
                )
            if not 2 <= params["n_folds"] <= 100:
                raise WorkflowActionError(
                    "Data Sampler 的 n_folds 必须在 2 到 100 之间"
                )
            if not 1 <= params["selected_folds"] <= params["n_folds"]:
                raise WorkflowActionError(
                    "Data Sampler 的 selected_folds 必须在 1 到 n_folds 之间"
                )
            return {
                "sampling_type": sampling_type,
                "sampleSizePercentage": percentage,
                # 非固定数量模式下，该值本来不会参与抽样；但若它大于输入
                # 行数，Orange 在加载数据时会截断 SpinBox，并由其回调把
                # sampling_type 意外切回 FixedSize。因此写入安全占位值 1，
                # 平台公开值仍完整保存在 _node_params，切回 FixedSize 时再恢复。
                "sampleSizeNumber": (
                    params["n_samples"] if sampling_type == 1 else 1
                ),
                "replacement": params["replacement"],
                "number_of_folds": params["n_folds"],
                "selectedFold": params["selected_folds"],
                "use_seed": params["replicable"],
                "stratify": params["stratified"],
                "compatibility_mode": False,
                "__version__": 2,
            }

        if widget_name == "Unique":  # 字段本身稍后写入 DomainContext。
            from Orange.widgets.data.owunique import OWUnique

            native_tiebreakers = tuple(OWUnique.TIEBREAKERS)
            tiebreakers = {
                "Last instance": native_tiebreakers[0],
                "First instance": native_tiebreakers[1],
                "Drop": native_tiebreakers[4],
            }
            try:
                tiebreaker = tiebreakers[params["configure"]]
            except KeyError as exc:
                raise WorkflowActionError(
                    "Unique 的 configure 只能是 Last instance、"
                    "First instance 或 Drop"
                ) from exc
            names = params["selGroupByVariables"]
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise WorkflowActionError(
                    "Unique 的 selGroupByVariables 只能包含非空字段名"
                )
            if len(names) != len(set(names)):
                raise WorkflowActionError(
                    "Unique 的 selGroupByVariables 不能包含重复字段"
                )
            return {
                "tiebreaker": tiebreaker,
                "autocommit": True,
                "__version__": 1,
            }

        if widget_name == "Impute":  # Uniplore 枚举需跳过 Orange 的内部状态项。
            method_map = {
                1: 1,  # Leave：不填补。
                2: 2,  # Average：均值或众数。
                3: 4,  # Model：基于模型。
                4: 5,  # Random：随机值。
                5: 6,  # Drop：删除含缺失值的行。
            }
            try:
                method = method_map[params["method"]]
            except KeyError as exc:
                raise WorkflowActionError(
                    "Impute 的 method 必须是 1、2、3、4 或 5"
                ) from exc
            return {
                "_default_method_index": method,
                "context_settings": [],
                "autocommit": True,
                "__version__": 1,
            }

        if widget_name == "Continuize":  # 具体字段提示需收到输入域后生成。
            choices = {
                "multi_attr": range(6),
                "cont_attr": range(3),
                "disc_cls_attr": range(4),
                "range": range(2),
            }
            for name, valid_values in choices.items():
                if params[name] not in valid_values:
                    raise WorkflowActionError(
                        f"Continuize 的 {name} 取值无效"
                    )
            return {"autosend": True, "__version__": 3}

        if widget_name == "Edit Domain":  # 映射对象需结合真实离散变量构造。
            column_name = params["column_name"]
            if not isinstance(column_name, str):
                raise WorkflowActionError(
                    "Edit Domain 的 column_name 必须是字符串"
                )
            mappings = params["valueMappings"]
            origins: set[str] = set()
            keys: set[int] = set()
            for index, mapping in enumerate(mappings):
                if not isinstance(mapping, dict):
                    raise WorkflowActionError(
                        f"Edit Domain 的 valueMappings[{index}] 必须是对象"
                    )
                if "originValue" not in mapping or "currentValue" not in mapping:
                    raise WorkflowActionError(
                        "Edit Domain 的每个映射必须包含 originValue 和 currentValue"
                    )
                key = mapping.get("key")
                if (
                    not isinstance(key, int)
                    or isinstance(key, bool)
                    or key < 0
                    or key in keys
                ):
                    raise WorkflowActionError(
                        "Edit Domain 映射的 key 必须是互不重复的非负整数"
                    )
                keys.add(key)
                origin = str(mapping["originValue"])
                if origin in origins:
                    raise WorkflowActionError(
                        f"Edit Domain 的原始值 {origin!r} 重复"
                    )
                origins.add(origin)
                current = mapping["currentValue"]
                if current is not None and not isinstance(
                    current, (str, int, float, bool)
                ):
                    raise WorkflowActionError(
                        "Edit Domain 的 currentValue 必须是标量或 null"
                    )
            if keys != set(range(len(mappings))):
                raise WorkflowActionError(
                    "Edit Domain 映射的 key 必须从 0 开始连续编号"
                )
            if mappings and not column_name.strip():
                raise WorkflowActionError(
                    "Edit Domain 配置 valueMappings 时必须指定 column_name"
                )
            return {"_domain_change_hints": {}, "__version__": 5}

        if widget_name == "Formula":  # JSON 表达式转换成 Orange 描述对象。
            import ast  # 使用 Orange 相同的 AST 白名单提前验证表达式。
            from Orange.widgets.data.owfeatureconstructor import (
                ContinuousDescriptor,
                DiscreteDescriptor,
                validate_exp,
            )

            descriptors = []
            used_names: set[str] = set()
            used_keys: set[int] = set()
            ordered_items = []
            for index, item in enumerate(params["expressions"]):
                if not isinstance(item, dict):
                    raise WorkflowActionError(
                        f"Formula 的 expressions[{index}] 必须是对象"
                    )
                key = item.get("key")
                if (
                    not isinstance(key, int)
                    or isinstance(key, bool)
                    or key < 0
                    or key in used_keys
                ):
                    raise WorkflowActionError(
                        "Formula 表达式的 key 必须是互不重复的非负整数"
                    )
                used_keys.add(key)
                ordered_items.append((key, index, item))
            if used_keys != set(range(len(ordered_items))):
                raise WorkflowActionError(
                    "Formula 表达式的 key 必须从 0 开始连续编号"
                )
            for _, index, item in sorted(ordered_items):
                name = item.get("name")
                expression = item.get("expression")
                if not isinstance(name, str) or not name.strip():
                    raise WorkflowActionError(
                        f"Formula 的 expressions[{index}].name 必须是非空字符串"
                    )
                if name in used_names:
                    raise WorkflowActionError(
                        f"Formula 的构造字段名 {name!r} 重复"
                    )
                used_names.add(name)
                if not isinstance(expression, str) or not expression.strip():
                    raise WorkflowActionError(
                        f"Formula 的 expressions[{index}].expression "
                        "必须是非空字符串"
                    )
                try:
                    valid_expression = validate_exp(
                        ast.parse(expression, mode="eval")
                    )
                except (SyntaxError, TypeError, ValueError) as exc:
                    raise WorkflowActionError(
                        f"Formula 表达式 {name!r} 不是 Orange 支持的表达式"
                    ) from exc
                if not valid_expression:
                    raise WorkflowActionError(
                        f"Formula 表达式 {name!r} 不是 Orange 支持的表达式"
                    )
                descriptor_type = str(item.get("type", ""))
                if descriptor_type == "2":
                    if item.get("image") not in (None, "N"):
                        raise WorkflowActionError(
                            "Formula 数值属性的 image 必须是 N"
                        )
                    descriptors.append(
                        ContinuousDescriptor(name, expression, 3, False)
                    )
                elif descriptor_type == "1":
                    if item.get("image") not in (None, "C"):
                        raise WorkflowActionError(
                            "Formula 离散属性的 image 必须是 C"
                        )
                    categories = item.get("categories", "")
                    if not isinstance(categories, str):
                        raise WorkflowActionError(
                            "Formula 离散属性的 categories 必须是逗号分隔字符串"
                        )
                    values = tuple(
                        value.strip()
                        for value in categories.split(",")
                        if value.strip()
                    )
                    if len(values) < 2:
                        raise WorkflowActionError(
                            "Formula 离散属性至少需要两个 categories"
                        )
                    descriptors.append(
                        DiscreteDescriptor(
                            name,
                            expression,
                            values,
                            False,
                            False,
                        )
                    )
                else:
                    raise WorkflowActionError(
                        "Formula 表达式的 type 只能是 1（离散）或 2（数值）"
                    )
            return {
                "descriptors": descriptors,
                "expressions_with_values": False,
                "currentIndex": -1,
                "__version__": 4,
            }

        if widget_name == "kNN":  # 字符串选项转换成组合框索引。
            if not 1 <= params["n_neighbors"] <= 100:
                raise WorkflowActionError(
                    "kNN 的 n_neighbors 必须在 1 到 100 之间"
                )
            metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]
            weights = ["uniform", "distance"]
            try:
                metric_index = metrics.index(params["metric"].lower())
            except ValueError as exc:
                raise WorkflowActionError(
                    f"kNN 的 metric 必须是 {metrics} 之一"
                ) from exc
            try:
                weight_index = weights.index(params["weights"].lower())
            except ValueError as exc:
                raise WorkflowActionError(
                    f"kNN 的 weights 必须是 {weights} 之一"
                ) from exc
            return {
                "n_neighbors": params["n_neighbors"],
                "metric_index": metric_index,
                "weight_index": weight_index,
            }

        if widget_name == "Tree":  # 六个公开设置与 Orange 原生设置同名。
            if params["min_leaf"] < 1:
                raise WorkflowActionError("Tree 的 min_leaf 必须大于 0")
            if params["min_internal"] < 1:
                raise WorkflowActionError("Tree 的 min_internal 必须大于 0")
            if params["max_depth"] < 1:
                raise WorkflowActionError("Tree 的 max_depth 必须大于 0")
            return copy.deepcopy(params)

        if widget_name == "Random Forest":
            if not 1 <= params["n_estimators"] <= 10000:
                raise WorkflowActionError(
                    "Random Forest 的 n_estimators 必须在 1 到 10000 之间"
                )
            # Orange 的 max_features 是整数 SpinBox，无法表达 Uniplore 的
            # "auto" 等非整数字符串。此时只在平台元数据中保留公开值，并在
            # 原生控件中关闭该限制，避免更新动作因无法转换而失败。
            native_use_max_features = False
            max_features = None
            if params["use_max_features"]:
                try:
                    max_features = int(params["max_features"])
                except (TypeError, ValueError):
                    max_features = None  # 不可表示的值由转换层静默屏蔽。
                if max_features is not None and max_features < 1:
                    raise WorkflowActionError(
                        "Random Forest 的 max_features 必须大于 0"
                    )
                native_use_max_features = max_features is not None
            if params["max_depth"] < 1:
                raise WorkflowActionError(
                    "Random Forest 的 max_depth 必须大于 0"
                )
            if params["min_samples_split"] < 2:
                raise WorkflowActionError(
                    "Random Forest 的 min_samples_split 必须至少为 2"
                )
            native_params = {
                "n_estimators": params["n_estimators"],
                "use_max_features": native_use_max_features,
                "use_random_state": params["use_random_state"],
                "use_max_depth": params["use_max_depth"],
                "max_depth": params["max_depth"],
                "use_min_samples_split": params["use_min_samples_split"],
                "min_samples_split": params["min_samples_split"],
            }
            if max_features is not None:
                native_params["max_features"] = max_features
            return native_params

        if widget_name == "Gradient Boosting":
            # loss_function 没有对应 Orange Setting；允许平台保存任意整数，
            # 但不把它写入原生属性，控件继续使用自身默认损失函数。
            if not 1 <= params["n_estimators"] <= 1000:
                raise WorkflowActionError(
                    "Gradient Boosting 的 n_estimators 必须在 1 到 1000 之间"
                )
            if not 0 < float(params["learning_rate"]) <= 1:
                raise WorkflowActionError(
                    "Gradient Boosting 的 learning_rate 必须大于 0 且不超过 1"
                )
            if params["max_depth"] < 1 or params["min_samples"] < 2:
                raise WorkflowActionError(
                    "Gradient Boosting 的 max_depth 必须大于 0，"
                    "min_samples 必须至少为 2"
                )
            return {
                "method_index": 0,  # Uniplore 对应 sklearn Gradient Boosting。
                "gb_editor": {
                    "n_estimators": params["n_estimators"],
                    "learning_rate": float(params["learning_rate"]),
                    "random_state": True,
                    "max_depth": (
                        params["max_depth"] if params["use_max_depth"] else 3
                    ),
                    "subsample": 1,
                    "min_samples_split": (
                        params["min_samples"]
                        if params["use_min_samples"]
                        else 2
                    ),
                },
            }

        if widget_name == "Linear Regression":
            regularization = {
                "0": 0,  # 无正则。
                "1": 2,  # Uniplore Lasso 对应 Orange Lasso。
                "2": 1,  # Uniplore Ridge 对应 Orange Ridge。
                "3": 3,  # Elastic Net。
            }
            try:
                reg_type = regularization[params["regression_type"]]
            except KeyError as exc:
                raise WorkflowActionError(
                    "Linear Regression 的 regression_type "
                    "只能是 '0'、'1'、'2' 或 '3'"
                ) from exc
            return {"reg_type": reg_type}

        if widget_name == "Logistic Regression":
            from Orange.widgets.model.owlogisticregression import (
                OWLogisticRegression,
            )

            penalty_map = {"l1": 0, "l2": 1}
            try:
                penalty_type = penalty_map[params["penalty"].lower()]
            except KeyError as exc:
                raise WorkflowActionError(
                    "Logistic Regression 的 penalty 只能是 l1 或 l2"
                ) from exc
            c_value = float(params["C"])
            if not 0.001 <= c_value <= 1000:
                raise WorkflowActionError(
                    "Logistic Regression 的 C 必须在 0.001 到 1000 之间"
                )
            matching = [
                index
                for index, value in enumerate(OWLogisticRegression.C_s)
                if math.isclose(float(value), c_value, rel_tol=0, abs_tol=1e-12)
            ]
            native_params = {"penalty_type": penalty_type}
            if matching:
                native_params["C_index"] = matching[0]
            # 非网格 C 只保留在平台元数据；原生 C_index 保持原值或默认值。
            return native_params

        if widget_name == "Naive Bayes":  # 模型名来自 OWBaseLearner Setting。
            if not params["name"].strip():
                raise WorkflowActionError(
                    "Naive Bayes 的 name 必须是非空字符串"
                )
            return {"learner_name": params["name"].strip()}

        if widget_name == "Neural Network":
            from Orange.widgets.model.owneuralnetwork import OWNNLearner

            if not re.fullmatch(
                r"\s*(?:\d+\s*,\s*)*\d+\s*,?\s*",
                params["hidden_layers_input"],
            ):
                raise WorkflowActionError(
                    "Neural Network 的 hidden_layers_input "
                    "必须是逗号分隔的正整数"
                )
            layers = [
                int(value)
                for value in re.findall(r"\d+", params["hidden_layers_input"])
            ]
            if any(value < 1 for value in layers):
                raise WorkflowActionError(
                    "Neural Network 的隐藏层节点数必须大于 0"
                )
            activation = int(params["activation"])
            solver = int(params["solver"])
            if activation not in range(4) or solver not in range(3):
                raise WorkflowActionError(
                    "Neural Network 的 activation 必须为 0~3，"
                    "solver 必须为 0~2"
                )
            if not 1 <= params["maxItera"] <= 1000000:
                raise WorkflowActionError(
                    "Neural Network 的 maxItera 必须在 1 到 1000000 之间"
                )
            alpha = float(params["alpha"])
            if not 0 <= alpha <= 1:
                raise WorkflowActionError(
                    "Neural Network 的 alpha 必须在 0 到 1 之间"
                )
            matching = [
                index
                for index, value in enumerate(OWNNLearner.alphas)
                if math.isclose(float(value), alpha, rel_tol=0, abs_tol=1e-12)
            ]
            native_params = {
                "hidden_layers_input": params["hidden_layers_input"],
                "activation_index": activation,
                "solver_index": solver,
                "max_iterations": params["maxItera"],
                "replicable": params["replicable"],
            }
            if matching:
                native_params["alpha_index"] = matching[0]
            # 非网格 alpha 只保留在平台元数据；原生滑块保持原值或默认值。
            return native_params

        if widget_name == "Predictions":  # 只有概率列具备对应原生 Setting。
            # output_attrs 与 output_predictions 没有原生 Setting。公开值仍会
            # 写入平台元数据并能由 get_workflow 查询，但不改变 Orange 输出。
            return {"show_scores": True}

        return copy.deepcopy(params)  # 空参数组件无需额外转换。

    def _store_postgres_credentials(self, *, require_complete: bool) -> None:
        """Store PostgreSQL secrets outside the serialized workflow."""
        required = ("host", "port", "user", "password")
        missing = [
            name
            for name in required
            if self.pg_config.get(name) in (None, "")
        ]
        if missing:
            if require_complete:
                raise WorkflowActionError(
                    "PG_CONFIG 缺少以下配置：" + ", ".join(missing)
                )
            return
        manager = OWSql._credential_manager(
            str(self.pg_config["host"]),
            str(self.pg_config["port"]),
        )
        manager.username = str(self.pg_config["user"])
        manager.password = str(self.pg_config["password"])

    def _apply_select_columns_existing_context(
        self,
        node: SchemeNode,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """直接更新 Select Columns 节点最近保存的数据域上下文。"""
        properties = (
            copy.deepcopy(node.properties)
            if isinstance(node.properties, dict)
            else {}
        )
        contexts = properties.get("context_settings")  # 读取最近使用优先的上下文。
        if not isinstance(contexts, list) or not contexts:
            return None  # 新节点尚未收到数据时没有可修改上下文。

        for context in contexts:  # 只修改最新的有效数据域上下文。
            values = getattr(context, "values", None)  # Context 把设置放在 values。
            if not isinstance(values, dict):
                continue
            packed = values.get("domain_role_hints")  # 取得编码后的角色映射。
            hints = (
                packed[0]
                if isinstance(packed, tuple)
                and len(packed) == 2
                and isinstance(packed[0], dict)
                else packed
            )
            if not isinstance(hints, dict):
                continue
            try:
                resolved_hints, readable = self._resolve_select_columns_hints(
                    hints,
                    params,
                )
            except WorkflowActionError:
                # The upstream SQL table may have changed since this context
                # was saved. Try another stored domain and, if none match,
                # let the caller materialize a fresh context from current data.
                continue
            values["domain_role_hints"] = (resolved_hints, -2)  # 使用原编码标记。
            properties["context_settings"] = contexts  # 放回深拷贝后的上下文。
            node.properties = properties  # 原子替换节点属性引用。
            return readable  # 返回设置后的四个可读区域。
        return None  # 没有包含 domain_role_hints 的上下文。

    def _materialize_select_columns_context(
        self,
        node_id: str,
        node: SchemeNode,
        params: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """运行上游数据并为新 Select Columns 节点生成原生 ContextSetting。"""
        node_index = list(self.scheme.nodes).index(node)  # 记录运行图中的节点顺序。
        temporary = self.workflow_path.with_name(  # 临时 OWS 与正式文件位于同目录。
            f".{self.workflow_path.name}.select-columns.{uuid.uuid4().hex}.tmp"
        )
        runtime_scheme = WidgetsScheme()  # 创建会实例化控件的独立执行图。
        runtime_scheme.widget_manager.set_creation_policy(
            runtime_scheme.widget_manager.Immediate  # 加载时立即创建所有控件。
        )
        try:
            self.scheme.save_to(  # 保存尚未提交的内存图供运行时读取。
                str(temporary),
                pretty=True,
                pickle_fallback=True,
            )
            runtime_scheme.load_from(  # 恢复完整上游工作流和信号连接。
                str(temporary),
                registry=self.registry,
            )
            runtime_node = runtime_scheme.nodes[node_index]  # 使用稳定节点顺序定位。
            widget = runtime_scheme.widget_for_node(runtime_node)  # 取得真实控件。
            deadline = time.monotonic() + timeout  # 限制文件、SQL 等上游等待时间。
            while widget.data is None:  # 等待 Select Columns 收到输入数据。
                self._app.processEvents()  # 处理 File 的零延迟加载定时器。
                if runtime_scheme.signal_manager.has_pending():
                    runtime_scheme.signal_manager.process_queued()  # 传播上游输出。
                self._app.processEvents()  # 处理传播产生的控件事件。
                if time.monotonic() >= deadline:
                    raise WorkflowActionError(
                        f"Select Columns 节点 {node_id} 在 {timeout:g} 秒内"
                        "没有收到输入数据，无法解析字段名"
                    )
                time.sleep(0.01)  # 避免等待时占满 CPU。

            widget.update_domain_role_hints()  # 从四个列表取得当前完整角色。
            resolved_hints, readable = self._resolve_select_columns_hints(
                widget.domain_role_hints,
                params,
            )
            variables = {  # 把字段名映射回当前数据域的 Variable 对象。
                variable.name: variable
                for variable in (
                    widget.data.domain.variables + widget.data.domain.metas
                )
            }
            names_by_role = {  # 从解析结果恢复每个内部角色的有序字段名。
                role: [
                    self._column_descriptor_name(descriptor)
                    for descriptor, (stored_role, index) in sorted(
                        resolved_hints.items(), key=lambda item: item[1][1]
                    )
                    if stored_role == role
                ]
                for role in ("attribute", "class", "meta", "available")
            }
            widget.used_attrs[:] = [
                variables[name] for name in names_by_role["attribute"]
            ]
            widget.class_attrs[:] = [
                variables[name] for name in names_by_role["class"]
            ]
            widget.meta_attrs[:] = [
                variables[name] for name in names_by_role["meta"]
            ]
            widget.available_attrs[:] = [
                variables[name] for name in names_by_role["available"]
            ]
            widget.update_domain_role_hints()  # 把列表重新写入 ContextSetting。
            widget.commit.now()  # 立即产生按新字段角色转换的输出数据。
            packed = widget.settingsHandler.pack_data(widget)  # 编码真实数据域上下文。
            metadata = (
                node.properties.get(_NODE_METADATA_KEY)
                if isinstance(node.properties, dict)
                else None
            )
            if isinstance(metadata, dict):  # 保留平台节点 ID 和 JSON 参数。
                packed[_NODE_METADATA_KEY] = copy.deepcopy(metadata)
            node.properties = packed  # 把可被 Orange Canvas 恢复的设置写回编辑图。
            return readable  # 供 get_workflow 展示解析后的实际角色。
        finally:
            runtime_scheme.clear()  # 断开运行时节点和信号。
            runtime_scheme.deleteLater()  # 交给 Qt 延迟销毁控件。
            self._app.processEvents()  # 完成延迟销毁。
            if temporary.exists():  # 清理同目录内明确创建的临时文件。
                temporary.unlink()

    def _materialize_select_columns_context_from_roles(
        self,
        node: SchemeNode,
        params: dict[str, Any],
        column_roles: Mapping[str, Mapping[str, str]],
    ) -> dict[str, Any]:
        """Build Select Columns ContextSetting directly from LLM role/type data.

        Loading a large SQL table through ``OWSql`` can open modal discovery
        dialogs. The resolver has already inspected every column, so construct
        an equivalent synthetic Orange domain and pack the real Select Columns
        widget context without querying the table a second time.
        """
        from Orange.data import (
            ContinuousVariable,
            DiscreteVariable,
            Domain,
            StringVariable,
            Table,
            TimeVariable,
        )
        from Orange.widgets.data.owselectcolumns import OWSelectAttributes

        attributes = []
        metas = []
        for column_name, column_config in column_roles.items():
            column_type = column_config["type"]
            if column_type == "numeric":
                variable = ContinuousVariable(column_name)
            elif column_type == "categorical":
                # Domain context matching only uses the variable type here;
                # concrete category values will come from the SQL Table.
                variable = DiscreteVariable(
                    column_name,
                    values=("0", "1"),
                )
            elif column_type == "datetime":
                variable = TimeVariable(column_name)
            elif column_type == "text":
                variable = StringVariable(column_name)
            else:  # Defensive guard; the resolver validates this beforehand.
                raise WorkflowActionError(
                    f"列 {column_name!r} 的 type {column_type!r} 不合法"
                )
            if variable.is_string:
                metas.append(variable)
            else:
                attributes.append(variable)

        domain = Domain(attributes, metas=metas)
        data = Table.from_domain(domain, 0)
        widget = OWSelectAttributes()
        try:
            widget.set_data(data)
            widget.update_domain_role_hints()
            resolved_hints, readable = self._resolve_select_columns_hints(
                widget.domain_role_hints,
                params,
            )
            variables = {
                variable.name: variable
                for variable in domain.variables + domain.metas
            }
            names_by_role = {
                role: [
                    self._column_descriptor_name(descriptor)
                    for descriptor, (stored_role, index) in sorted(
                        resolved_hints.items(),
                        key=lambda item: item[1][1],
                    )
                    if stored_role == role
                ]
                for role in ("attribute", "class", "meta", "available")
            }
            widget.used_attrs[:] = [
                variables[name] for name in names_by_role["attribute"]
            ]
            widget.class_attrs[:] = [
                variables[name] for name in names_by_role["class"]
            ]
            widget.meta_attrs[:] = [
                variables[name] for name in names_by_role["meta"]
            ]
            widget.available_attrs[:] = [
                variables[name] for name in names_by_role["available"]
            ]
            widget.update_domain_role_hints()
            packed = widget.settingsHandler.pack_data(widget)
            metadata = (
                node.properties.get(_NODE_METADATA_KEY)
                if isinstance(node.properties, dict)
                else None
            )
            if isinstance(metadata, dict):
                packed[_NODE_METADATA_KEY] = copy.deepcopy(metadata)
            node.properties = packed
            return readable
        finally:
            widget.set_data(None)
            widget.deleteLater()
            self._app.processEvents()

    def _configure_select_columns_node(
        self,
        node_id: str,
        node: SchemeNode,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """优先修改现有上下文，必要时运行上游数据创建新上下文。"""
        readable = self._apply_select_columns_existing_context(node, params)
        if readable is not None:  # 已保存上下文无需重复执行工作流。
            return readable
        has_input = any(  # 新上下文只有连接了数据输入后才能创建。
            link.sink_node is node for link in self.scheme.links
        )
        if not has_input:
            return None  # 允许先设参数、稍后再连接数据输入。
        return self._materialize_select_columns_context(
            node_id,
            node,
            params,
        )

    @staticmethod
    def _select_columns_params_from_resolution(
        resolution: Any,
    ) -> dict[str, Any]:
        """Convert resolved LLM column roles into Select Columns parameters."""
        column_roles = getattr(resolution, "column_roles", None)
        if not isinstance(column_roles, Mapping) or not column_roles:
            raise WorkflowActionError(
                "数据集列配置缺少有效的 column_roles，"
                "无法自动配置 Select Columns"
            )

        features: list[str] = []
        targets: list[str] = []
        ignores: list[str] = []
        valid_types = {"numeric", "categorical", "datetime", "text"}
        for column_name, column_config in column_roles.items():
            if (
                not isinstance(column_name, str)
                or not column_name
                or not isinstance(column_config, Mapping)
            ):
                raise WorkflowActionError(
                    "数据集 column_roles 包含无效的列配置"
                )
            column_type = column_config.get("type")
            if column_type not in valid_types:
                raise WorkflowActionError(
                    f"列 {column_name!r} 的 type {column_type!r} 不合法"
                )
            role = column_config.get("role")
            if role == "feature":
                features.append(column_name)
            elif role == "target":
                targets.append(column_name)
            elif role in {"skip", "meta"}:
                # Select Columns 的公共契约只有 ignores/features/targets；
                # 不参与建模的 skip/meta 列统一移入 Ignored。
                ignores.append(column_name)
            else:
                raise WorkflowActionError(
                    f"列 {column_name!r} 的 role {role!r} 不合法"
                )

        if len(targets) != 1:
            raise WorkflowActionError(
                "Orange3 Select Columns 要求数据集列配置中有且只有一个"
                f" target，当前识别到 {len(targets)} 个"
            )
        if not features:
            raise WorkflowActionError(
                "Orange3 Select Columns 至少需要一个 feature 列"
            )
        return {
            "ignores": ignores,
            "features": features,
            "targets": targets[0],
        }

    def _direct_select_columns_nodes(
        self,
        sql_node: SchemeNode,
    ) -> list[tuple[str, str, SchemeNode]]:
        """Return direct SQL -> Select Columns links in public-ID form."""
        node_ids = {
            id(node): node_id for node_id, node in self._nodes.items()
        }
        edge_ids = {
            id(edge): edge_id for edge_id, edge in self._edges.items()
        }
        matches: list[tuple[str, str, SchemeNode]] = []
        for link in self.scheme.links:
            if link.source_node is not sql_node:
                continue
            target_id = node_ids.get(id(link.sink_node))
            edge_id = edge_ids.get(id(link))
            if (
                target_id is not None
                and edge_id is not None
                and self._node_widget_names.get(target_id)
                == "Select Columns"
            ):
                matches.append((edge_id, target_id, link.sink_node))
        return matches

    def _ensure_sql_select_columns(
        self,
        sql_node_id: str,
        sql_node: SchemeNode,
        params: dict[str, Any],
        column_roles: Mapping[str, Mapping[str, str]],
    ) -> dict[str, Any]:
        """Create or update the Select Columns node owned by a SQL load."""
        matches = self._direct_select_columns_nodes(sql_node)
        if len(matches) > 1:
            raise WorkflowActionError(
                f"SQL Table 节点 {sql_node_id} 连接了多个 Select Columns，"
                "无法确定要自动配置的节点"
            )

        if matches:
            edge_id, select_id, select_node = matches[0]
            previous_params = copy.deepcopy(
                self._node_params[select_id]
            )
            previous_properties = copy.deepcopy(select_node.properties)
            merged_params = copy.deepcopy(previous_params)
            merged_params.update(copy.deepcopy(params))
            native_params = self._to_native_node_params(
                "Select Columns",
                merged_params,
            )
            self._node_params[select_id] = merged_params
            patched = (
                dict(select_node.properties)
                if isinstance(select_node.properties, dict)
                else {}
            )
            patched.update(native_params)
            select_node.properties = patched
            try:
                resolved_params = (
                    self._materialize_select_columns_context_from_roles(
                    select_node,
                    merged_params,
                    column_roles,
                )
                )
                if resolved_params is not None:
                    merged_params.update(resolved_params)
                    self._node_params[select_id] = merged_params
            except Exception:
                self._node_params[select_id] = previous_params
                select_node.properties = previous_properties
                raise
            return {
                "created": False,
                "edge_id": edge_id,
                "select_id": select_id,
                "previous_params": previous_params,
                "previous_properties": previous_properties,
            }

        try:
            description = self._widgets_by_name["Select Columns"]
            self._widget_definitions["Select Columns"]
        except KeyError as exc:
            raise WorkflowActionError(
                "当前 Orange3 环境未提供可用的 Select Columns 组件"
            ) from exc

        previous_position_index = self._next_position_index
        select_node = self.scheme.new_node(
            description,
            title=f"{sql_node.title}列配置",
            position=self._next_position(),
            properties={},
        )
        select_id = str(next(self._node_ids))
        self._nodes[select_id] = select_node
        self._node_widget_names[select_id] = "Select Columns"
        merged_params = self._default_params("Select Columns")
        merged_params.update(copy.deepcopy(params))
        self._node_params[select_id] = merged_params
        link = None
        edge_id = None
        try:
            native_params = self._to_native_node_params(
                "Select Columns",
                merged_params,
            )
            select_node.properties = native_params

            proposals = self.scheme.propose_links(sql_node, select_node)
            if not proposals:
                raise WorkflowActionError(
                    "SQL Table 与 Select Columns 之间没有兼容端口"
                )
            non_selection_proposals = [
                proposal
                for proposal in proposals
                if "selected" not in (proposal[0].id or "").lower()
                and "selected" not in proposal[0].name.lower()
            ]
            if non_selection_proposals:
                proposals = non_selection_proposals
            source_channel, target_channel, _ = max(
                proposals,
                key=self._link_semantic_score,
            )
            link = self.scheme.new_link(
                sql_node,
                source_channel,
                select_node,
                target_channel,
            )
            edge_id = self._stable_edge_id(
                sql_node_id,
                source_channel.id,
                select_id,
                target_channel.id,
            )
            self._edges[edge_id] = link

            resolved_params = (
                self._materialize_select_columns_context_from_roles(
                    select_node,
                    merged_params,
                    column_roles,
                )
            )
            if resolved_params is not None:
                merged_params.update(resolved_params)
                self._node_params[select_id] = merged_params
        except Exception:
            if edge_id is not None:
                self._edges.pop(edge_id, None)
            if link is not None and link in self.scheme.links:
                self.scheme.remove_link(link)
            if select_node in self.scheme.nodes:
                self.scheme.remove_node(select_node)
            self._nodes.pop(select_id, None)
            self._node_widget_names.pop(select_id, None)
            self._node_params.pop(select_id, None)
            self._next_position_index = previous_position_index
            raise
        return {
            "created": True,
            "edge_id": edge_id,
            "select_id": select_id,
            "previous_position_index": previous_position_index,
        }

    def _rollback_sql_select_columns(
        self,
        mutation: dict[str, Any] | None,
    ) -> None:
        """Restore Select Columns state if the enclosing SQL update fails."""
        if mutation is None:
            return
        select_id = mutation["select_id"]
        if not mutation["created"]:
            select_node = self._nodes.get(select_id)
            if select_node is not None:
                self._node_params[select_id] = mutation["previous_params"]
                select_node.properties = mutation["previous_properties"]
            return

        edge_id = mutation["edge_id"]
        link = self._edges.pop(edge_id, None)
        if link is not None and link in self.scheme.links:
            self.scheme.remove_link(link)
        select_node = self._nodes.pop(select_id, None)
        if select_node is not None and select_node in self.scheme.nodes:
            self.scheme.remove_node(select_node)
        self._node_widget_names.pop(select_id, None)
        self._node_params.pop(select_id, None)
        self._next_position_index = mutation["previous_position_index"]

    def _apply_data_dependent_params(
        self,
        widget_name: str,
        widget: Any,
        params: dict[str, Any],
    ) -> None:
        """把依赖输入数据域的公开参数应用到已实例化 Orange 控件。"""
        if widget_name == "Unique":  # selected_vars 是 DomainContextSetting。
            variables = {  # 同时允许属性、目标和 Meta 字段参与去重。
                variable.name: variable
                for variable in (
                    widget.data.domain.variables + widget.data.domain.metas
                )
            }
            requested = params["selGroupByVariables"]  # 空列表表示使用全部列。
            unknown = sorted(set(requested) - set(variables))
            if unknown:
                raise WorkflowActionError(
                    "Unique 输入数据中不存在以下字段：" + ", ".join(unknown)
                )
            widget.selected_vars = (
                [variables[name] for name in requested]
                if requested
                else list(widget.var_model)
            )
            widget.commit.now()  # 立即生成按指定字段去重的输出。
            return

        if widget_name == "Continuize":  # 角色不同的字段使用不同提示映射。
            from Orange.widgets.data.owcontinuize import (
                Continuize,
                DefaultKey,
                Normalize,
            )

            discrete_methods = {
                0: Continuize.FirstAsBase,
                1: Continuize.FrequentAsBase,
                2: Continuize.Indicators,
                3: Continuize.Remove,
                4: Continuize.AsOrdinal,
                5: Continuize.AsNormalizedOrdinal,
            }
            continuous_methods = {
                0: Normalize.Leave,
                1: (
                    Normalize.Normalize11
                    if params["range"] == 0
                    else Normalize.Normalize01
                ),
                2: Normalize.Standardize,
            }
            class_methods = {
                0: Continuize.Leave,
                1: Continuize.AsOrdinal,
                2: Continuize.AsNormalizedOrdinal,
                3: Continuize.Indicators,
            }
            widget.disc_var_hints = {
                DefaultKey: discrete_methods[params["multi_attr"]]
            }
            widget.cont_var_hints = {
                DefaultKey: continuous_methods[params["cont_attr"]]
            }
            class_var = widget.data.domain.class_var  # 单目标才有可配置类字段。
            if class_var is not None and class_var.is_discrete:
                widget.disc_var_hints[class_var.name] = class_methods[
                    params["disc_cls_attr"]
                ]
            elif params["disc_cls_attr"] != 0:
                raise WorkflowActionError(
                    "Continuize 的 disc_cls_attr 非 0 时，"
                    "输入数据必须有且只有一个离散目标字段"
                )
            data = widget.data  # 重新载入输入以刷新列表模型和字段提示。
            widget.set_data(data)
            widget.commit.now()  # 立即应用连续化后的输出域。
            return

        if widget_name == "Edit Domain":  # 类别映射依赖输入变量的真实取值。
            from Orange.widgets.data.oweditdomain import (
                CategoriesMapping,
                abstract,
            )

            variables = {  # Edit Domain 也允许修改目标与 Meta。
                variable.name: variable
                for variable in (
                    widget.data.domain.variables + widget.data.domain.metas
                )
            }
            column_name = params["column_name"]
            mappings = params["valueMappings"]
            widget._domain_change_hints.clear()  # 公开参数精确替换旧映射。
            if mappings:
                try:
                    variable = variables[column_name]
                except KeyError as exc:
                    raise WorkflowActionError(
                        f"Edit Domain 输入数据中不存在字段 {column_name!r}"
                    ) from exc
                if not variable.is_discrete:
                    raise WorkflowActionError(
                        "Orange Edit Domain 的 valueMappings "
                        "只能直接映射离散字段的类别标签"
                    )
                ordered_mappings = sorted(
                    mappings,
                    key=lambda mapping: mapping["key"],
                )
                replacement = {
                    str(mapping["originValue"]): (
                        None
                        if mapping["currentValue"] is None
                        else str(mapping["currentValue"])
                    )
                    for mapping in ordered_mappings
                }
                unknown = sorted(set(replacement) - set(variable.values))
                if unknown:
                    raise WorkflowActionError(
                        f"Edit Domain 字段 {column_name!r} 中不存在以下原始值："
                        + ", ".join(unknown)
                    )
                category_mapping = [  # 显式映射按 key 决定新类别顺序。
                    (
                        str(mapping["originValue"]),
                        (
                            None
                            if mapping["currentValue"] is None
                            else str(mapping["currentValue"])
                        ),
                    )
                    for mapping in ordered_mappings
                ]
                category_mapping.extend(  # 未配置类别保持原值和原相对顺序。
                    (value, value)
                    for value in variable.values
                    if value not in replacement
                )
                widget._store_transform(
                    abstract(variable),
                    [CategoriesMapping(category_mapping)],
                )
            data = widget.data  # 重载模型以从新 hints 恢复转换对象。
            widget.set_data(data)
            widget.commit()  # 立即输出重新标记类别后的数据。
            return

        if widget_name == "Predictions":  # shown_probs 是类别上下文设置。
            widget.shown_probs = (
                widget.MODEL_PROBS
                if params["output_probabilities"]
                else widget.NO_PROBS
            )
            widget.commit()  # 同步输出预测数据及可选概率列。
            return

        raise WorkflowActionError(
            f"组件 {widget_name} 没有数据域参数转换器"
        )

    def _materialize_data_dependent_context(
        self,
        node_id: str,
        node: SchemeNode,
        widget_name: str,
        params: dict[str, Any],
        timeout: float = 30.0,
    ) -> None:
        """运行上游图，将数据域参数打包回编辑工作流节点。"""
        node_index = list(self.scheme.nodes).index(node)  # 保存稳定节点位置。
        temporary = self.workflow_path.with_name(
            f".{self.workflow_path.name}.{widget_name.lower().replace(' ', '-')}"
            f".{uuid.uuid4().hex}.tmp"
        )
        runtime_scheme = WidgetsScheme()  # 使用真实控件恢复上下文 Setting。
        runtime_scheme.widget_manager.set_creation_policy(
            runtime_scheme.widget_manager.Immediate
        )
        try:
            self.scheme.save_to(
                str(temporary),
                pretty=True,
                pickle_fallback=True,
            )
            runtime_scheme.load_from(
                str(temporary),
                registry=self.registry,
            )
            runtime_node = runtime_scheme.nodes[node_index]
            widget = runtime_scheme.widget_for_node(runtime_node)
            deadline = time.monotonic() + timeout
            while getattr(widget, "data", None) is None:
                self._app.processEvents()
                if runtime_scheme.signal_manager.has_pending():
                    runtime_scheme.signal_manager.process_queued()
                self._app.processEvents()
                if time.monotonic() >= deadline:
                    raise WorkflowActionError(
                        f"{widget_name} 节点 {node_id} 在 {timeout:g} 秒内"
                        "没有收到输入数据，无法解析数据域参数"
                    )
                time.sleep(0.01)

            self._apply_data_dependent_params(
                widget_name,
                widget,
                params,
            )
            self._app.processEvents()  # 处理立即提交产生的信号。
            packed = widget.settingsHandler.pack_data(widget)
            metadata = (
                node.properties.get(_NODE_METADATA_KEY)
                if isinstance(node.properties, dict)
                else None
            )
            if isinstance(metadata, dict):
                packed[_NODE_METADATA_KEY] = copy.deepcopy(metadata)
            node.properties = packed  # 保存原生上下文供 Canvas 恢复。
        finally:
            runtime_scheme.clear()
            runtime_scheme.deleteLater()
            self._app.processEvents()
            if temporary.exists():
                temporary.unlink()

    def _configure_data_dependent_node(
        self,
        node_id: str,
        node: SchemeNode,
        widget_name: str,
        params: dict[str, Any],
    ) -> None:
        """节点有数据输入时物化上下文；无输入时保留参数等待后续配置。"""
        has_input = any(
            link.sink_node is node for link in self.scheme.links
        )
        if not has_input:
            return
        self._materialize_data_dependent_context(
            node_id,
            node,
            widget_name,
            params,
        )

    def _load_current_workflow(self) -> None:
        """从 OWS 文件恢复图、平台元数据、节点 ID 计数器和边索引。

        Raises:
            Exception: XML 或 Orange 工作流内容无法加载。
        """
        try:  # 同时读取原始 XML ID 和 Orange 图对象。
            xml_root = ElementTree.parse(self.workflow_path).getroot()  # 解析 OWS XML。
            serialized_node_ids = [  # 保存文件中节点顺序对应的原始 ID。
                element.attrib["id"]  # 读取每个 node 元素的 id 属性。
                for element in xml_root.findall("./nodes/node")  # 定位所有节点元素。
            ]
            self.scheme.load_from(  # 让 Orange 恢复节点、通道、边和属性。
                str(self.workflow_path),  # Orange API 使用字符串路径。
                registry=self.registry,  # 用已发现注册表解析组件类型。
            )
        except Exception as exc:  # 为所有底层加载错误补充工作流路径。
            raise Exception(f"无法加载当前工作流 {self.workflow_path}：{exc}") from exc

        used_node_ids: set[str] = set()  # 保证恢复后的公开 ID 唯一。
        for index, node in enumerate(self.scheme.nodes):  # 按 XML 节点顺序恢复元数据。
            properties = (
                node.properties if isinstance(node.properties, dict) else {}  # 防御旧格式。
            )
            metadata = properties.get(_NODE_METADATA_KEY, {})  # 读取平台私有元数据。
            if not isinstance(metadata, dict):  # 损坏元数据不能影响整个图加载。
                metadata = {}  # 回退到 XML 和组件描述推断。

            fallback_id = (
                serialized_node_ids[index]  # 优先沿用 OWS 原始 ID。
                if index < len(serialized_node_ids)  # 确认 XML 中存在对应项。
                else str(index)  # 异常情况下使用顺序索引。
            )
            node_id = str(metadata.get("node_id", fallback_id))  # 元数据 ID 优先。
            if not node_id or node_id in used_node_ids:  # 修复空值或重复值。
                numeric_id = 1  # 从最小正整数开始寻找空闲 ID。
                while str(numeric_id) in used_node_ids:  # 跳过已占用 ID。
                    numeric_id += 1  # 检查下一个候选值。
                node_id = str(numeric_id)  # 使用找到的唯一 ID。
            used_node_ids.add(node_id)  # 标记当前 ID 已占用。

            external_name = metadata.get("widget_name")  # 优先使用持久化的平台名称。
            if external_name not in self._widget_definitions:  # 旧文件可能没有元数据。
                external_name = self._external_name_by_qualified_name.get(
                    node.description.qualified_name,  # 使用稳定 Python 限定名查找。
                    node.description.name,  # 查不到时回退 Orange 显示名。
                )

            if external_name in self._widget_definitions:  # 只恢复平台认识的参数。
                params = self._default_params(external_name)  # 先建立完整默认参数。
                stored_params = metadata.get("node_params", {})  # 读取平台保存值。
                if not isinstance(stored_params, dict):  # 忽略损坏的非字典数据。
                    stored_params = {}
                if stored_params:
                    for name, value in stored_params.items():  # 合并已知参数。
                        if name in params:  # 丢弃配置中已删除的旧参数。
                            params[name] = copy.deepcopy(value)  # 隔离可变值。
                    if external_name == "SQL Table":
                        table_name = stored_params.get("table_name")
                        if isinstance(table_name, str) and table_name.strip():
                            params["table_name"] = table_name.strip()
                    if (  # 把上一版 ignored 参数迁移为现在的 ignores。
                        external_name == "Select Columns"
                        and "ignores" not in stored_params
                        and isinstance(stored_params.get("ignored"), list)
                    ):
                        params["ignores"] = copy.deepcopy(
                            stored_params["ignored"]
                        )
                for name in params:  # 无公开元数据时再从同名原生属性恢复。
                    if (  # 有公开元数据时不能被转换后的原生同名属性覆盖。
                        name not in stored_params
                        and name in properties
                    ):
                        params[name] = copy.deepcopy(properties[name])  # 隔离状态。
                if (
                    external_name == "SQL Table"
                    and "table_name" not in params
                    and isinstance(properties.get("table"), str)
                    and properties["table"].strip()
                ):
                    params["table_name"] = properties["table"].strip()
                if external_name == "Select Columns":  # 兼容旧版数组和空值模式。
                    if params["features"] is None:
                        params["features"] = []
                    if isinstance(params["targets"], list):
                        params["targets"] = (
                            params["targets"][0]
                            if params["targets"]
                            else ""
                        )
                    restored_params = (
                        self._select_columns_params_from_properties(properties)
                    )
                    if restored_params is not None:  # 原生上下文反映实际角色。
                        params.update(restored_params)
            else:  # 未在平台 JSON 中定义的旧组件无法验证参数。
                params = {}  # 保留节点但不暴露未知参数。

            self._nodes[node_id] = node  # 建立公开 ID 到 Orange 节点索引。
            self._node_widget_names[node_id] = external_name  # 保存平台名称。
            self._node_params[node_id] = params  # 保存合并后的完整参数。

        numeric_ids = [  # 收集现有纯数字 ID 以避免新节点冲突。
            int(node_id)  # 转为整数用于求最大值。
            for node_id in self._nodes  # 遍历所有公开节点 ID。
            if node_id.isdigit()  # 非数字历史 ID 不参与计数。
        ]
        self._node_ids = itertools.count(  # 创建无限递增 ID 生成器。
            max(numeric_ids, default=0) + 1  # 从最大现有数字 ID 的下一位开始。
        )
        self._next_position_index = len(self._nodes)  # 新节点排在现有节点之后。

        reverse_nodes = {node: node_id for node_id, node in self._nodes.items()}  # 反查 ID。
        for link in self.scheme.links:  # 为每条已加载连线恢复稳定公开 ID。
            edge_id = self._stable_edge_id(  # 根据完整端点身份计算边 ID。
                reverse_nodes[link.source_node],  # 源节点公开 ID。
                link.source_channel.id,  # 源输出通道 ID。
                reverse_nodes[link.sink_node],  # 目标节点公开 ID。
                link.sink_channel.id,  # 目标输入通道 ID。
            )
            self._edges[edge_id] = link  # 建立公开边索引。

    @staticmethod
    def _stable_edge_id(
        source_node_id: str,
        source_channel_id: str | None,
        target_node_id: str,
        target_channel_id: str | None,
    ) -> str:
        """根据节点与通道端点生成确定性的公开边 ID。

        Returns:
            SHA-256 摘要前 64 位对应的无符号十进制字符串。
        """
        identity = (
            f"{source_node_id}|{source_channel_id or ''}|"  # 编码源端点。
            f"{target_node_id}|{target_channel_id or ''}"  # 编码目标端点。
        ).encode("utf-8")  # 固定 UTF-8 确保跨环境一致。
        digest = hashlib.sha256(identity).digest()  # 计算低碰撞摘要。
        return str(int.from_bytes(digest[:8], "big"))  # 输出 64 位十进制 ID。

    def _next_position(self) -> tuple[float, float]:
        """按五列网格计算下一个新节点的画布坐标。"""
        index = self._next_position_index  # 读取当前布局序号。
        self._next_position_index += 1  # 提前推进，保证每次调用位置不同。
        column = index % 5  # 每行最多放置五个节点。
        row = index // 5  # 超过五个后换到下一行。
        return 30.0 + column * 190.0, 100.0 + row * 140.0  # 加入间距和边距。

    @staticmethod
    def _link_semantic_score(proposal: tuple[Any, Any, int]) -> int:
        """对候选通道配对评分，优先主数据输出并回避交互式选中输出。"""
        source_channel, target_channel, orange_weight = proposal  # 解包 Orange 候选。
        score = orange_weight * 100  # Orange 原生兼容权重作为主要排序依据。
        source_id = (source_channel.id or "").lower()  # 规范源通道 ID。
        target_id = (target_channel.id or "").lower()  # 规范目标通道 ID。
        source_name = source_channel.name.lower()  # 规范源显示名。
        target_name = target_channel.name.lower()  # 规范目标显示名。

        if source_id == target_id:  # 完全相同的机器通道 ID 通常语义最佳。
            score += 30  # 给予最高附加分。
        if source_name == target_name:  # 显示名一致也表明语义匹配。
            score += 20  # 给予次级附加分。
        if "selected" in source_id or "selected" in source_name:  # 选中数据依赖 UI。
            score -= 10  # 自动工作流优先完整数据输出。
        if source_id in {"annotated", "annotated_data"} and target_id == "data":
            score += 5  # 标注数据仍可安全连接普通数据输入。
        return score  # 供 max 选择最高分候选。

    def _persist(self) -> None:
        """把当前内存图和平台元数据原子保存到 OWS 文件。"""
        self.workflow_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
        for node_id, node in self._nodes.items():  # 写入每个节点的平台私有元数据。
            properties = (
                dict(node.properties)  # 复制原属性避免原地污染共享字典。
                if isinstance(node.properties, dict)  # 仅复制合法字典。
                else {}  # 异常旧值回退为空属性。
            )
            properties[_NODE_METADATA_KEY] = {  # 写入独立命名空间。
                "node_id": node_id,  # 保存稳定公开节点 ID。
                "widget_name": self._node_widget_names[node_id],  # 保存平台名称。
                "node_params": copy.deepcopy(self._node_params[node_id]),  # 保存参数快照。
            }
            node.properties = properties  # 交还 Orange 序列化器。

        temporary = self.workflow_path.with_name(  # 在目标目录创建唯一临时路径。
            f".{self.workflow_path.name}.{uuid.uuid4().hex}.tmp"  # 避免并发名称冲突。
        )
        try:  # 先完整写临时文件，成功后再替换正式文件。
            self.scheme.save_to(  # 使用 Orange 官方序列化器。
                str(temporary),  # 临时目标路径。
                pretty=True,  # 输出便于审阅的 XML。
                pickle_fallback=True,  # 允许 Orange 保存复杂控件属性。
            )
            os.replace(temporary, self.workflow_path)  # 同文件系统内原子替换。
        finally:  # 无论保存成功与否都清理残留临时文件。
            if temporary.exists():  # 替换成功后路径已不存在。
                temporary.unlink()  # 删除失败写入留下的临时文件。

    def add_node(self, args: dict[str, Any]) -> str:
        """新增一个已授权 Orange 组件节点并返回其公开 ID。

        ``args`` 必须包含 ``widget_name`` 和非空中文 ``node_name``。
        """
        widget_name = args["widget_name"]  # 读取要实例化的组件名。
        node_name = args["node_name"]  # 读取画布展示标题。
        description = self._resolve_widget(widget_name)  # 校验授权并解析组件。
        if not isinstance(node_name, str) or not node_name.strip():  # 拒绝空标题。
            raise WorkflowActionError("node_name 必须是非空名称")

        node = self.scheme.new_node(  # 在内存工作流中创建 Orange 节点。
            description,  # 指定真实组件描述。
            title=node_name.strip(),  # 去除标题两端空白。
            position=self._next_position(),  # 自动计算不重叠的网格位置。
            properties={},  # 新节点从空原生属性开始。
        )
        node_id = str(next(self._node_ids))  # 分配下一个数字公开 ID。
        self._nodes[node_id] = node  # 更新节点索引。
        self._node_widget_names[node_id] = widget_name  # 记录平台组件名。
        self._node_params[node_id] = self._default_params(widget_name)  # 初始化参数。
        self._persist()  # 立即持久化，保证动作完成即落盘。
        return node_id  # 供后续连边和更新参数使用。

    def delete_node(self, args: dict[str, Any]) -> None:
        """删除指定节点、所有关联边及其平台索引记录。"""
        node_id = args["node_id"]  # 读取外部节点 ID。
        normalized, node = self._resolve_node(node_id)  # 解析真实节点。
        connected_edge_ids = [  # 删除前记录会被 Orange 级联删除的边 ID。
            edge_id  # 保存待清理的公开边 ID。
            for edge_id, edge in self._edges.items()  # 遍历全部边索引。
            if edge.source_node is node or edge.sink_node is node  # 匹配任一端点。
        ]

        self.scheme.remove_node(node)  # 从 Orange 图删除节点并级联移除连线。
        del self._nodes[normalized]  # 清理节点对象索引。
        del self._node_widget_names[normalized]  # 清理组件名索引。
        del self._node_params[normalized]  # 清理参数索引。
        for edge_id in connected_edge_ids:  # 同步清理所有关联边索引。
            del self._edges[edge_id]  # 删除单条公开边记录。
        self._persist()  # 保存删除后的工作流。

    def update_node_params(
        self,
        args: dict[str, Any],
    ) -> None:
        """校验并局部更新节点参数，未提供的参数保持不变。

        ``args`` 必须包含 ``node_id``、``widget_name`` 和字典或 JSON 字符串
        ``node_params``；未知参数、类型错误和非标准 JSON 值会被拒绝。
        """
        node_id = args["node_id"]  # 读取目标节点 ID。
        widget_name = args["widget_name"]  # 读取调用方声明的组件名。
        node_params = args["node_params"]  # 读取增量参数。
        if not isinstance(node_params, dict):  # 兼容上层模型生成的 JSON 字符串。
            try:  # 尝试把字符串解析为对象。
                node_params = json.loads(node_params)  # 使用标准 JSON 解析器。
            except (TypeError, json.JSONDecodeError) as exc:
                raise WorkflowActionError(
                    "node_params 必须是 JSON 对象"
                ) from exc

        normalized, node = self._resolve_node(node_id)  # 解析真实节点。
        actual_widget_name = self._node_widget_names[normalized]  # 读取真实组件名。
        if widget_name != actual_widget_name:  # 防止用错误模式更新节点。
            raise WorkflowActionError(
                f"节点 {node_id} 的组件是 {actual_widget_name!r}，"
                f"不是 {widget_name!r}"
            )
        self._resolve_widget(widget_name)  # 再次执行当前任务授权校验。
        if not isinstance(node_params, dict):  # JSON 也可能解析成数组或标量。
            raise WorkflowActionError("node_params 必须是 JSON 对象")
        if any(not isinstance(key, str) or not key for key in node_params):  # 校验键。
            raise WorkflowActionError("node_params 的参数名必须是非空字符串")

        try:  # 确认值可安全进入 OWS 与外部 JSON。
            json.dumps(node_params, ensure_ascii=False, allow_nan=False)  # 严格序列化。
        except (TypeError, ValueError) as exc:  # 捕获对象类型及 NaN/Infinity。
            raise WorkflowActionError(
                "node_params 必须只包含标准 JSON 可序列化值"
            ) from exc

        definitions = self._param_definitions(widget_name)  # 取得参数模式。
        unknown_params = sorted(set(node_params) - set(definitions))  # 找未知字段。
        if unknown_params:  # 未声明参数可能不会被控件消费。
            raise WorkflowActionError(
                f"组件 {widget_name} 在 widgets.json 中没有以下参数："
                + ", ".join(unknown_params)
            )
        for param_name, value in node_params.items():  # 逐个执行模式类型校验。
            self._validate_param_type(  # 抛出包含组件和参数名的动作异常。
                widget_name,  # 当前组件。
                param_name,  # 当前参数。
                value,  # 待校验值。
                definitions[param_name].get("type"),  # JSON 声明的期望类型。
            )

        select_columns_params: dict[str, Any] | None = None
        resolved_column_roles: Mapping[str, Mapping[str, str]] | None = None
        if widget_name == "SQL Table":
            data_description = node_params.get("data_description")
            if (
                not isinstance(data_description, str)
                or not data_description.strip()
            ):
                raise WorkflowActionError(
                    "SQL Table 的 data_description 必须是非空字符串"
                )
            started_at = time.perf_counter()
            try:
                resolution = resolve_dataset(
                    data_description,
                    "postgresql",
                    self.pg_config,
                )
            except Exception as exc:
                raise WorkflowActionError(
                    "无法根据 data_description 从 PostgreSQL 确认数据表："
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            finally:
                self.data_retrieval_time += (
                    time.perf_counter() - started_at
                )
            node_params = copy.deepcopy(node_params)
            node_params["table_name"] = resolution.table_name
            select_columns_params = (
                self._select_columns_params_from_resolution(resolution)
            )
            resolved_column_roles = resolution.column_roles
            self.data_info = resolution.eda_result
            self.ori_data_info = resolution.eda_result
            self.input_tokens += resolution.input_tokens
            self.output_tokens += resolution.output_tokens
            self.total_tokens += (
                resolution.input_tokens + resolution.output_tokens
            )

        previous_params = copy.deepcopy(self._node_params[normalized])  # 保存回滚快照。
        previous_properties = copy.deepcopy(node.properties)  # 保存原生属性快照。
        select_columns_mutation: dict[str, Any] | None = None
        try:
            merged_params = copy.deepcopy(previous_params)  # 构造更新后的完整状态。
            merged_params.update(copy.deepcopy(node_params))  # 合并本次增量值。
            native_params = self._to_native_node_params(  # 转换特殊控件属性。
                widget_name,
                merged_params,  # 转换逻辑始终使用完整参数以处理关联选项。
            )
            self._node_params[normalized] = merged_params  # 保存 JSON 友好参数。
            patched = dict(node.properties)  # 复制 Orange 原有属性。
            patched.update(native_params)  # 写入 Orange 能直接消费的参数。
            node.properties = patched  # 把合并结果交回节点。
            if widget_name == "SQL Table":
                select_columns_mutation = self._ensure_sql_select_columns(
                    normalized,
                    node,
                    select_columns_params,
                    resolved_column_roles,
                )
            elif widget_name == "Select Columns":  # 字段角色必须写入数据域上下文。
                resolved_params = self._configure_select_columns_node(
                    normalized,
                    node,
                    merged_params,
                )
                if resolved_params is not None:  # 展示实际生效的三个参数。
                    merged_params.update(resolved_params)
                    self._node_params[normalized] = merged_params
            elif widget_name in {
                "Edit Domain",
                "Unique",
                "Continuize",
                "Predictions",
            }:
                self._configure_data_dependent_node(
                    normalized,
                    node,
                    widget_name,
                    merged_params,
                )
            self._persist()  # 全部转换成功后才原子保存更新。
        except Exception:
            self._rollback_sql_select_columns(select_columns_mutation)
            self._node_params[normalized] = previous_params  # 恢复平台参数。
            node.properties = previous_properties  # 恢复 Orange 原生属性。
            raise  # 让调用方看到准确的字段或输入错误。

    def add_edge(self, args: dict[str, Any]) -> str:
        """在两个节点的最高分兼容通道之间创建边并返回稳定边 ID。"""
        source_node_id = args["source_node_id"]  # 读取源节点 ID。
        target_node_id = args["target_node_id"]  # 读取目标节点 ID。
        source_id, source = self._resolve_node(source_node_id)  # 解析源节点。
        target_id, target = self._resolve_node(target_node_id)  # 解析目标节点。
        if source is target:  # Orange 工作流不允许节点自连接。
            raise WorkflowActionError("不能连接节点自身")

        proposals = self.scheme.propose_links(source, target)  # 获取类型兼容通道。
        if not proposals:  # 没有候选说明两组件不能直接连接。
            raise WorkflowActionError(
                f"节点 {source_id}（{source.description.name}）与节点 "
                f"{target_id}（{target.description.name}）之间没有可用的兼容端口"
            )

        non_selection_proposals = [  # 自动流程优先使用完整输出。
            proposal  # 保留非 selected 候选。
            for proposal in proposals  # 遍历兼容候选。
            if "selected" not in (proposal[0].id or "").lower()
            and "selected" not in proposal[0].name.lower()  # 同时检查显示名。
        ]
        if non_selection_proposals:  # 只有存在替代项时才排除 selected。
            proposals = non_selection_proposals  # 缩小候选集合。

        source_channel, target_channel, _ = max(
            proposals,  # 在剩余候选中选一个。
            key=self._link_semantic_score,  # 使用语义评分排序。
        )
        try:  # Orange 仍可能因通道占用或运行时约束拒绝连线。
            link = self.scheme.new_link(  # 在内存图中创建边。
                source,  # 源节点。
                source_channel,  # 源输出通道。
                target,  # 目标节点。
                target_channel,  # 目标输入通道。
            )
        except (TypeError, ValueError) as exc:
            raise WorkflowActionError(
                f"无法添加从节点 {source_id} 到节点 {target_id} 的边：{exc}"
            ) from exc

        edge_id = self._stable_edge_id(  # 根据最终端点生成公开 ID。
            source_id,  # 源节点 ID。
            source_channel.id,  # 源通道 ID。
            target_id,  # 目标节点 ID。
            target_channel.id,  # 目标通道 ID。
        )
        self._edges[edge_id] = link  # 保存边索引。
        if self._node_widget_names[target_id] == "Select Columns":
            select_params = self._node_params[target_id]  # 读取可能提前设置的角色。
            if select_params["targets"]:  # 参数早于连线设置时创建数据域上下文。
                try:
                    resolved_params = self._configure_select_columns_node(
                        target_id,
                        target,
                        select_params,
                    )
                    if resolved_params is not None:  # 同步实际生效的三个参数。
                        select_params.update(resolved_params)
                except Exception:
                    self.scheme.remove_link(link)  # 恢复添加边前的内存图。
                    del self._edges[edge_id]  # 同步移除公开边索引。
                    raise  # 要求调用方先确保上游能够产生数据。
        elif (
            self._node_widget_names[target_id]
            in {"Edit Domain", "Unique", "Continuize", "Predictions"}
            and (target_channel.id or "").lower() == "data"
        ):
            try:  # 参数早于数据连线设置时补建真实数据域上下文。
                self._configure_data_dependent_node(
                    target_id,
                    target,
                    self._node_widget_names[target_id],
                    self._node_params[target_id],
                )
            except Exception:
                self.scheme.remove_link(link)  # 恢复添加边前的内存图。
                del self._edges[edge_id]  # 同步移除公开边索引。
                raise
        self._persist()  # 持久化新边。
        return edge_id  # 返回给后续删除动作。

    def delete_edge(self, args: dict[str, Any]) -> None:
        """根据 ``args['edge_id']`` 删除一条边并持久化工作流。"""
        edge_id = args["edge_id"]  # 读取外部边 ID。
        normalized = self._as_id(edge_id, "edge_id")  # 规范成字符串键。
        try:  # 从公开边索引解析 Orange 连线。
            link = self._edges[normalized]  # 取得待删除连线。
        except KeyError as exc:  # 将内部索引错误转换成动作错误。
            raise WorkflowActionError(f"边 ID {normalized} 不存在") from exc

        self.scheme.remove_link(link)  # 从 Orange 图删除连线。
        del self._edges[normalized]  # 同步清理公开索引。
        self._persist()  # 保存删除后的图。

    def get_workflow(self) -> dict[str, list[dict[str, Any]]]:
        """返回由 ``nodes`` 和 ``edges`` 组成的 JSON 就绪工作流快照。"""
        reverse_nodes = {node: node_id for node_id, node in self._nodes.items()}
        edges = [  # 序列化全部边。
            {
                "edge_id": edge_id,  # 稳定公开边 ID。
                "source_node_id": reverse_nodes[edge.source_node],  # 源节点 ID。
                "target_node_id": reverse_nodes[edge.sink_node],  # 目标节点 ID。
            }
            for edge_id, edge in self._edges.items()  # 保持内存图顺序。
        ]
        nodes = []  # 按图顺序收集可读节点信息。
        for node_id, node in self._nodes.items():  # 遍历节点索引。
            node_info = {
                "node_id": node_id,
                "widget_name": self._node_widget_names[node_id],
                "node_name": node.title,
                "node_params": copy.deepcopy(self._node_params[node_id])
            }
            nodes.append(node_info)  # 添加本节点的完整快照。
        return {"edges": edges, "nodes": nodes}  # 返回统一外部模式。

    def get_test_and_score_results(
        self,
        node_id: Any | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """执行 Test and Score，并仅在调用期间过滤两个已知第三方警告。

        Args:
            node_id: 可选的 Test and Score 节点 ID；为空时选择工作流中的对应节点。
            timeout: 等待评估完成的最大秒数。

        Returns:
            第一个成功学习器的指标字典。
        """
        with warnings.catch_warnings():  # 确保过滤规则不会泄漏到全局。
            warnings.filterwarnings(  # 忽略 Orange/orangewidget 版本兼容提示。
                "ignore",  # 仅忽略匹配的警告。
                message=r"decorate OWRandomForest\.apply with @gui\.deferred.*",
                category=UserWarning,  # 兼容提示的类别。
                module=r"orangewidget\.gui",  # 严格限制来源模块。
            )
            warnings.filterwarnings(  # 忽略精确率在空预测类别上的已知定义警告。
                "ignore",  # sklearn 已按约定把该类别贡献设为零。
                message=r"Precision is ill-defined.*",
                category=UndefinedMetricWarning,  # 只匹配指标未定义警告。
                module=r"sklearn\.metrics\._classification",  # 限定 sklearn 模块。
            )
            return self._get_test_and_score_results(node_id, timeout)  # 执行核心逻辑。

    def _get_test_and_score_results(
        self,
        node_id: Any | None,
        timeout: float,
    ) -> dict[str, Any]:
        """加载独立运行时工作流，驱动 Qt 事件循环并提取评估指标。

        Raises:
            WorkflowActionError: 参数无效、目标节点错误、执行超时或 Orange 运行失败。
        """
        # 超时必须是正的有限数字，且明确排除 Python 中属于 int 子类的 bool。
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise WorkflowActionError("timeout 必须是大于 0 的有限秒数")

        selected_node_id = None  # None 表示自动选择 Test and Score。
        if node_id is not None:  # 调用方指定节点时执行严格校验。
            selected_node_id, _ = self._resolve_node(node_id)  # 解析公开 ID。
            if (
                self._node_widget_names[selected_node_id]
                != "Test and Score"
            ):
                raise WorkflowActionError(
                    f"节点 {selected_node_id} 不是 Test and Score 组件"
                )

        try:  # 读取 XML ID 以映射运行时节点。
            xml_root = ElementTree.parse(self.workflow_path).getroot()  # 解析 OWS。
            serialized_node_ids = [  # 保存文件节点顺序。
                element.attrib["id"]  # 读取原始节点 ID。
                for element in xml_root.findall("./nodes/node")  # 遍历节点元素。
            ]
        except (OSError, ElementTree.ParseError) as exc:
            raise WorkflowActionError(
                f"无法读取当前工作流：{self.workflow_path}"
            ) from exc

        runtime_scheme = WidgetsScheme()  # 创建会真实实例化控件的独立运行图。
        runtime_scheme.widget_manager.set_creation_policy(
            runtime_scheme.widget_manager.Immediate  # 加载节点时立即创建控件。
        )
        try:  # 确保无论成功失败都释放运行时控件。
            runtime_scheme.load_from(  # 从同一 OWS 加载可执行图。
                str(self.workflow_path),  # 工作流路径。
                registry=self.registry,  # 组件注册表。
            )
            runtime_ids: dict[SchemeNode, str] = {}
            for index, runtime_node in enumerate(runtime_scheme.nodes):
                properties = (
                    runtime_node.properties
                    if isinstance(runtime_node.properties, dict)
                    else {}
                )
                metadata = properties.get(_NODE_METADATA_KEY, {})
                fallback_id = (
                    serialized_node_ids[index]
                    if index < len(serialized_node_ids)
                    else str(index)
                )
                runtime_ids[runtime_node] = str(
                    metadata.get("node_id", fallback_id)
                    if isinstance(metadata, dict)
                    else fallback_id
                )

            target_nodes = [
                runtime_node
                for runtime_node in runtime_scheme.nodes
                if runtime_node.description.name == "Test and Score"
                and (
                    selected_node_id is None
                    or runtime_ids[runtime_node] == selected_node_id
                )
            ]
            if not target_nodes:
                requested = (
                    f"节点 {selected_node_id}"
                    if selected_node_id is not None
                    else "当前工作流"
                )
                raise WorkflowActionError(
                    f"{requested} 中没有可执行的 Test and Score 组件"
                )

            target_widgets = [
                runtime_scheme.widget_for_node(runtime_node)
                for runtime_node in target_nodes
            ]
            deadline = time.monotonic() + float(timeout)  # 使用单调时钟避免系统时间跳变。
            while True:  # 持续驱动 Qt 与 Orange 信号直到完成或超时。
                self._app.processEvents()  # 处理控件事件。
                signal_manager = runtime_scheme.signal_manager  # 获取信号调度器。
                if signal_manager.has_pending():  # 只在有任务时处理队列。
                    signal_manager.process_queued()  # 传播节点输出。
                self._app.processEvents()  # 处理传播触发的新事件。

                complete = all(
                    widget.learners
                    and all(
                        slot.results is not None
                        for slot in widget.learners.values()
                    )
                    for widget in target_widgets
                )
                if complete:  # 所有目标控件的学习器均已给出结果。
                    break  # 退出轮询并序列化分数。
                if time.monotonic() >= deadline:  # 达到调用方截止时间。
                    waiting = [
                        runtime_ids[target_node]
                        for target_node, widget in zip(
                            target_nodes, target_widgets
                        )
                        if not (
                            widget.learners
                            and all(
                                slot.results is not None
                                for slot in widget.learners.values()
                            )
                        )
                    ]
                    raise WorkflowActionError(
                        "等待 Test and Score 运行完成超时；"
                        f"仍未完成的节点：{waiting}"
                    )
                time.sleep(0.01)  # 避免等待期间占满 CPU。

            result_nodes = [
                _serialize_test_and_score_widget(
                    runtime_ids[runtime_node],
                    runtime_node,
                    widget,
                )
                for runtime_node, widget in zip(
                    target_nodes, target_widgets
                )
            ]
            result = {
                "workflow_path": str(self.workflow_path),
                "results": result_nodes,
            }
            return result["results"][0]["learners"][0]["scores"]  # 返回首个学习器分数。
        except WorkflowActionError:
            raise
        except Exception as exc:
            raise WorkflowActionError(
                f"执行工作流并获取 Test and Score 结果失败：{exc}"
            ) from exc
        finally:  # 始终释放临时控件和 Qt 对象。
            runtime_scheme.clear()  # 断开节点与边。
            runtime_scheme.deleteLater()  # 交给 Qt 安全延迟销毁。
            self._app.processEvents()  # 立即处理销毁事件。

    def clear_workflow(self) -> None:
        workflow = self.get_workflow()
        for node in workflow["nodes"]:
            self.delete_node({"node_id": node["node_id"]})
        self._next_position_index = 0

Actions = PlatformAction
# print(json.dumps(action_agent.get_workflow(), indent=2, ensure_ascii=False))

# action_agent.relevant_widgets_names = ["Select Columns"]
# action_agent.add_node({"widget_name": "Select Columns", "node_name": "Test"})
# action_agent.add_edge({"source_node_id": "0", "target_node_id": "5"})
# action_agent.delete_edge({"edge_id":"13457465284687817199"})
# action_agent.delete_node({"node_id": "4"})
# print(json.dumps(action_agent.get_test_and_score_results(), indent=2, ensure_ascii=False))
