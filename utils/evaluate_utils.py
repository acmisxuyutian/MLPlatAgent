from typing import Any, Dict, List, Set, Tuple
import os
import json

def get_project_root():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 向上一级
    current_directory = os.path.dirname(current_file_path)
    # 再次向上一级
    project_root = os.path.dirname(current_directory)

    return project_root

def workflow_graph_f1(wg: Dict[str, Any], rg: Dict[str, Any]) -> float:
    """
    计算预测 Workflow Graph 与标注 Requirements Graph 之间的 Workflow Graph F1。

    计算过程：
    1. 根据 requirements 中的 contrast_level 判断预测节点与标注节点是否匹配；
    2. 在一对一节点映射约束下，寻找最大的结构一致节点集合；
    3. 使用最大结构一致节点数 k 计算：
           precision = k / |V_wg|
           recall    = k / |V_rg|
           F1        = 2k / (|V_wg| + |V_rg|)

    这里的结构比较基于“可达关系”而非直接边：
    - rg 中 A -> B 表示 A 必须先于 B；
    - wg 中只要存在 A 到 B 的路径，就认为该依赖成立；
    - 同时也会惩罚 wg 中多出的反向或额外依赖关系。

    Parameters
    ----------
    wg:
        预测工作流，格式为：
        {
            "nodes": [
                {
                    "node_id": str,
                    "widget_name": str,
                    "node_params": dict,
                    ...
                }
            ],
            "edges": [
                {
                    "source_node_id": str,
                    "target_node_id": str,
                    ...
                }
            ]
        }

    rg:
        标注 requirements graph，格式为：
        {
            "step_requirements": [
                {
                    "node_id": str,
                    "algorithm_step_name": str,
                    "node_type": str,
                    "node_params": dict,
                    "contrast_level": int,
                    ...
                }
            ],
            "dependency_requirements": [
                {
                    "source_node_id": str,
                    "target_node_id": str,
                    ...
                }
            ]
        }

    Returns
    -------
    float
        Workflow Graph F1，范围为 [0, 1]，保留 4 位小数。
    """

    try:
        wg = json.loads(wg)
        rg = json.loads(rg)
    except:
        raise TypeError("wg and rg must both be dictionaries")

    workflow_nodes = wg.get("nodes", [])
    workflow_edges = wg.get("edges", [])
    requirement_nodes = rg.get("step_requirements", [])
    requirement_edges = rg.get("dependency_requirements", [])

    num_predicted_nodes = len(workflow_nodes)
    num_gold_nodes = len(requirement_nodes)

    # 两张图均为空，视为完全一致。
    if num_predicted_nodes == 0 and num_gold_nodes == 0:
        return 1.0

    # 只有一张图为空，不存在公共节点。
    if num_predicted_nodes == 0 or num_gold_nodes == 0:
        return 0.0

    widget_type_map, widget_algorithm_step_map = _load_widget_metadata()

    # 预测图和标注图的传递闭包。
    workflow_reachability = _build_reachability(
        nodes=workflow_nodes,
        edges=workflow_edges,
        node_key="node_id",
    )

    requirement_reachability = _build_reachability(
        nodes=requirement_nodes,
        edges=requirement_edges,
        node_key="node_id",
    )

    # candidates[gold_id] 表示一个标注节点可以匹配哪些预测节点。
    candidates: Dict[str, List[str]] = {}

    for requirement in requirement_nodes:
        requirement_id = requirement["node_id"]

        candidates[requirement_id] = [
            node["node_id"]
            for node in workflow_nodes
            if _workflow_node_matches_requirement(
                workflow_node=node,
                requirement=requirement,
                widget_type_map=widget_type_map,
                widget_algorithm_step_map=widget_algorithm_step_map,
            )
        ]

    # 候选数量少的标注节点优先搜索，可以显著减少回溯空间。
    ordered_requirements = sorted(
        requirement_nodes,
        key=lambda requirement: len(candidates[requirement["node_id"]]),
    )

    max_common_nodes = _maximum_common_induced_mapping_size(
        ordered_requirements=ordered_requirements,
        candidates=candidates,
        workflow_reachability=workflow_reachability,
        requirement_reachability=requirement_reachability,
    )

    f1 = (
        2 * max_common_nodes
        / (num_predicted_nodes + num_gold_nodes)
    )

    return round(f1, 4)

def _load_widget_metadata() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    加载：
    1. widget_name -> widget type
    2. widget_name -> algorithm step name
    """
    from ml_platform.algorithm_step_widget_map import WIDGET_ALGORITHM_STEP_MAP

    widgets_path = os.path.join(
        get_project_root(),
        "data",
        "ml_platform_data",
        "widgets.json",
    )

    with open(widgets_path, "r", encoding="utf-8") as file:
        widgets = json.load(file)

    widget_type_map = {
        widget["widget_name"]: widget["type"]
        for widget in widgets
    }

    return widget_type_map, WIDGET_ALGORITHM_STEP_MAP

def _normalize_widget_name(widget_name: str) -> str:
    """
    保留原 SAT() 中对 Test & Score 的兼容处理，
    但不直接修改输入节点。
    """
    if widget_name == "Test & Score":
        return "Test Score"

    return widget_name

def _workflow_node_matches_requirement(
    workflow_node: Dict[str, Any],
    requirement: Dict[str, Any],
    widget_type_map: Dict[str, str],
    widget_algorithm_step_map: Dict[str, str],
) -> bool:
    """
    判断一个预测 workflow node 是否满足一个 step requirement。

    该逻辑与原 SAT(requirement, 'sr', workflow) 保持一致。
    """
    contrast_level = requirement.get("contrast_level", 0)

    widget_name = _normalize_widget_name(
        workflow_node.get("widget_name", "")
    )

    # contrast_level == 0：
    # 仅要求预测节点和标注节点具有相同的节点类型。
    if contrast_level == 0:
        predicted_node_type = widget_type_map.get(widget_name)
        required_node_type = requirement.get("node_type")

        return (
            predicted_node_type is not None
            and predicted_node_type == required_node_type
        )

    # 将具体组件名称转换为统一的算法步骤名称。
    predicted_algorithm_step = widget_algorithm_step_map.get(
        widget_name,
        widget_name,
    )

    required_algorithm_step = requirement.get("algorithm_step_name")

    if predicted_algorithm_step != required_algorithm_step:
        return False

    # contrast_level == 1：
    # 只要求算法步骤名称一致。
    if contrast_level == 1:
        return True

    # contrast_level >= 2：
    # 标注中要求的参数必须全部出现在预测节点中且值相同。
    required_params = requirement.get("node_params") or {}
    predicted_params = workflow_node.get("node_params") or {}

    for param_name, required_value in required_params.items():
        if param_name not in predicted_params:
            return False

        if predicted_params[param_name] != required_value:
            return False

    return True

def _build_reachability(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    node_key: str,
) -> Dict[Tuple[str, str], bool]:
    """
    计算有向图的可达关系，即传递闭包。

    返回：
        reachability[(source_id, target_id)] = True / False

    与原 floyd_warshall() 不同，这里不将节点到自身视为依赖，
    因为结构匹配时只比较不同节点之间的执行先后关系。
    """
    node_ids = [node[node_key] for node in nodes]
    node_id_set = set(node_ids)

    adjacency: Dict[str, Set[str]] = {
        node_id: set()
        for node_id in node_ids
    }

    for edge in edges:
        source_id = edge.get("source_node_id")
        target_id = edge.get("target_node_id")

        # 忽略引用不存在节点的非法边。
        if source_id not in node_id_set or target_id not in node_id_set:
            continue

        adjacency[source_id].add(target_id)

    reachability: Dict[Tuple[str, str], bool] = {}

    for source_id in node_ids:
        visited: Set[str] = set()
        stack = list(adjacency[source_id])

        while stack:
            current_id = stack.pop()

            if current_id in visited:
                continue

            visited.add(current_id)
            stack.extend(adjacency[current_id] - visited)

        for target_id in node_ids:
            reachability[(source_id, target_id)] = (
                source_id != target_id
                and target_id in visited
            )

    return reachability

def _maximum_common_induced_mapping_size(
    ordered_requirements: List[Dict[str, Any]],
    candidates: Dict[str, List[str]],
    workflow_reachability: Dict[Tuple[str, str], bool],
    requirement_reachability: Dict[Tuple[str, str], bool],
) -> int:
    """
    在节点匹配候选和结构一致约束下，计算最大公共诱导子图节点数。

    mapping 的含义：
        requirement_node_id -> workflow_node_id

    对任何两个已经匹配的节点 r1、r2，均要求：

        rg 中 r1 是否可达 r2
        ==
        wg 中 mapping[r1] 是否可达 mapping[r2]

    并同时检查反方向，因此：
    - 缺少依赖会受到惩罚；
    - 多出依赖也会受到惩罚；
    - 错误的串并行关系会受到惩罚；
    - 反向依赖会受到惩罚。
    """
    maximum_size = 0
    total_requirements = len(ordered_requirements)

    def structurally_consistent(
        requirement_id: str,
        workflow_id: str,
        current_mapping: Dict[str, str],
    ) -> bool:
        for mapped_requirement_id, mapped_workflow_id in current_mapping.items():
            # requirement_id -> mapped_requirement_id
            gold_forward = requirement_reachability.get(
                (requirement_id, mapped_requirement_id),
                False,
            )
            predicted_forward = workflow_reachability.get(
                (workflow_id, mapped_workflow_id),
                False,
            )

            if gold_forward != predicted_forward:
                return False

            # mapped_requirement_id -> requirement_id
            gold_backward = requirement_reachability.get(
                (mapped_requirement_id, requirement_id),
                False,
            )
            predicted_backward = workflow_reachability.get(
                (mapped_workflow_id, workflow_id),
                False,
            )

            if gold_backward != predicted_backward:
                return False

        return True

    def backtrack(
        requirement_index: int,
        current_mapping: Dict[str, str],
        used_workflow_nodes: Set[str],
    ) -> None:
        nonlocal maximum_size

        current_size = len(current_mapping)

        if current_size > maximum_size:
            maximum_size = current_size

        # 分支限界：即使剩余标注节点全部成功匹配，
        # 也不可能超过当前最优结果时，提前终止。
        remaining_requirements = total_requirements - requirement_index

        if current_size + remaining_requirements <= maximum_size:
            return

        if requirement_index >= total_requirements:
            return

        requirement = ordered_requirements[requirement_index]
        requirement_id = requirement["node_id"]

        # 分支1：将当前标注节点映射到某个预测节点。
        for workflow_id in candidates.get(requirement_id, []):
            if workflow_id in used_workflow_nodes:
                continue

            if not structurally_consistent(
                requirement_id=requirement_id,
                workflow_id=workflow_id,
                current_mapping=current_mapping,
            ):
                continue

            current_mapping[requirement_id] = workflow_id
            used_workflow_nodes.add(workflow_id)

            backtrack(
                requirement_index=requirement_index + 1,
                current_mapping=current_mapping,
                used_workflow_nodes=used_workflow_nodes,
            )

            used_workflow_nodes.remove(workflow_id)
            del current_mapping[requirement_id]

        # 分支2：当前标注节点不进入公共子图。
        backtrack(
            requirement_index=requirement_index + 1,
            current_mapping=current_mapping,
            used_workflow_nodes=used_workflow_nodes,
        )

    backtrack(
        requirement_index=0,
        current_mapping={},
        used_workflow_nodes=set(),
    )

    return maximum_size