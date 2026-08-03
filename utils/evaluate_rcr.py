# 构建图的邻接列表
import json
from utils.logs import logger
def build_adj_matrix(workflow):

    num_nodes = len(workflow["nodes"])

    # 初始化邻接矩阵
    matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # 填充邻接矩阵
    for e in workflow["edges"]:
        # 由于节点列表可能是非顺序的或者不连续的，我们需要找到节点在列表中的索引
        source_index = None
        target_index = None
        for i in range(len(workflow["nodes"])):
            if workflow["nodes"][i]["node_id"] == e["source_node_id"]:
                source_index = i
            elif workflow["nodes"][i]["node_id"] == e["target_node_id"]:
                target_index = i

        matrix[source_index][target_index] = 1  # 对于有向图

    return matrix

# 比较两个节点是否相同
def comparing_node(node1, node2):

    # node2为人类标注
    if node2["contrast_level"] == 0:
        # 等级0：较为宽松，要求两个节点所属的任务类型相同
        from utils.utils import get_project_root
        import os
        import json
        path = os.path.join(get_project_root(), r'data/ml_platform_data/widgets.json')
        with open(path, 'r', encoding='utf-8') as f:
            widgets = json.load(f)
        node1_type = None
        node2_type = None
        for w in widgets:
            if w['widget_name'] == node1['widget_name']:
                node1_type = w['package']
            if w['widget_name'] == node2['widget_name']:
                node2_type = w['package']
        return node1_type == node2_type
    elif node2["contrast_level"] == 1:
        # 等级1：一般宽松，要求两个节点的组件类型相同
        return node1["widget_name"] == node2["widget_name"]
    else:
        return (node1["widget_name"] == node2["widget_name"]) and (node1["node_params"] == node2["node_params"])

# 检查Agent构建的工作流是否与人类标注的工作流相同
def comparing_workflow(workflow_A, workflow_H):
    if len(workflow_A['nodes']) != len(workflow_H['nodes']) or len(workflow_A['edges']) != len(workflow_H['edges']):
        return False

    adj_matrixA = build_adj_matrix(workflow_A)
    adj_matrixH = build_adj_matrix(workflow_H)

    # 创一个列表标记
    visited = [False] * len(workflow_H["nodes"])

    for i in range(len(workflow_A["nodes"])):
        for j in range(len(workflow_H["nodes"])):
            if visited[j]:
                continue
            in_degreeA =  sum([adj_matrixA[k][i] for k in range(len(adj_matrixA))])
            in_degreeH = sum([adj_matrixH[k][j] for k in range(len(adj_matrixH))])
            # 两个节点信息相等，对应的连接出度和入度也相等
            if comparing_node(workflow_A["nodes"][i], workflow_H["nodes"][j]) and\
                (sum(adj_matrixA[i]) == sum(adj_matrixH[j])) and\
                    (in_degreeA == in_degreeH):
                visited[j] = True
                break
    if visited == [True] * len(workflow_H["nodes"]):
        return True
    return False

def update_node_id(or_node_id, update_node_id, workflow):
    for node in workflow["nodes"]:
        if node["node_id"] == or_node_id:
            node["node_id"] = update_node_id
            node["matched"] = True
    for edge in workflow["edges"]:
        if edge["source_node_id"] == or_node_id:
            edge["source_node_id"] = update_node_id
        if edge["target_node_id"] == or_node_id:
            edge["target_node_id"] = update_node_id

def floyd_warshall(workflow):
    import numpy as np
    # Extracting node IDs
    node_ids = [node["node_id"] for node in workflow["nodes"]]
    # Creating an adjacency matrix
    n = len(node_ids)
    adjacency_matrix = np.zeros((n, n))

    # Populating the adjacency matrix based on edges
    for edge in workflow["edges"]:
        source_index = node_ids.index(edge["source_node_id"])
        target_index = node_ids.index(edge["target_node_id"])
        adjacency_matrix[source_index, target_index] = 1

    matrix = adjacency_matrix
    """Floyd-Warshall algorithm to find all pairs shortest paths."""
    n = len(matrix)
    # Initialize the solution matrix same as input graph matrix.
    dist = np.copy(matrix)

    # Add all vertices one by one to the set of intermediate vertices.
    for k in range(n):
        # Pick all vertices as source one by one.
        for i in range(n):
            # Pick all vertices as destination for the above picked source.
            for j in range(n):
                # If vertex k is on the shortest path from i to j, then update the value of dist[i][j].
                dist[i][j] = max(dist[i][j], dist[i][k] * dist[k][j])

    # Setting self-loops to 1 (there's always a path from a node to itself)
    np.fill_diagonal(dist, 1)

    return dist.tolist(), node_ids

def SAT(requirement, requirement_type, workflow):
    """
    用于计算需求requirement是否被工作流workflow满足
    :param requirement: 需求可能为三类，
        - sr, 表示要求工作流中存在某个节点
        - dr，表示要求工作流中的某个节点a必须在另一个节点b之前。
    :param workflow: 工作流是一个有向无环图，由节点集合和边集合组成，{"edges": [edge1, edge2, ...], "nodes": [node1, node2, ...]}。
        - edges，边列表中的每一个边会表示为一个JSON：{"edge_id": "边ID", "source_node_id": "边的起始节点ID", "target_node_id": "边的目标节点ID"}；
        - nodes，节点列表中的每一个节点也会表示为一个JSON：{"node_id": "节点ID", "widget_name": "组件名称", "node_name": "节点名称", "node_params": 节点参数}。
    :return:
    """
    if requirement_type == 'sr':
        from experiments.algorithm_step_widget_map import UNIPLORE_ALGORITHM_STEP_MAP
        from utils.utils import get_project_root
        import os
        import json

        path = os.path.join(get_project_root(), r'ml_platform/uniplore/widgets.json')
        with open(path, 'r', encoding='utf-8') as f:
            widgets = json.load(f)
        widget_type_map = {w["widget_name"]: w["type"] for w in widgets}
        for node in workflow["nodes"]:
            if node["matched"]:
                continue
            if requirement["contrast_level"] == 0:
                if node["widget_name"] == "Test & Score":
                    node["widget_name"] = "Test Score"
                if requirement["node_type"] == widget_type_map[node["widget_name"]]:
                    update_node_id(node["node_id"], requirement["node_id"], workflow)
                    return 1
            else:
                if node["widget_name"] in UNIPLORE_ALGORITHM_STEP_MAP.keys():
                    widget_algorithm_step = UNIPLORE_ALGORITHM_STEP_MAP[node["widget_name"]]
                else:
                    widget_algorithm_step = node["widget_name"]
                if requirement["algorithm_step_name"] == widget_algorithm_step:
                    if requirement["contrast_level"] == 1:
                        update_node_id(node["node_id"], requirement["node_id"], workflow)
                        return 1
                    else:
                        is_params_satisfied = True
                        for param in requirement["node_params"]:
                            if param not in node["node_params"]:
                                is_params_satisfied = False
                                break
                            if requirement["node_params"][param] != node["node_params"][param]:
                                is_params_satisfied = False
                                break
                        if is_params_satisfied:
                            update_node_id(node["node_id"], requirement["node_id"], workflow)
                            return 1
        return 0

    elif requirement_type == 'dr':
        path_existence_matrix, node_ids = floyd_warshall(workflow)
        if requirement["source_node_id"] in node_ids and requirement["target_node_id"] in node_ids:
            i = node_ids.index(requirement["source_node_id"])
            j = node_ids.index(requirement["target_node_id"])
            if path_existence_matrix[i][j] == 1:
                return 1
        # logger.info(f"{requirement} is not satisfied")
        return 0
    else:
        raise ValueError("requirement type error, only support 'sr', , 'dr'")

def RCR(requirements, workflow):
    if workflow == {}:
        return 0
    count_requirements = 0
    count_satisfied = 0
    # 标记工作流中的节点是否已被匹配过，一个节点仅能满足一个需求
    for node in workflow["nodes"]:
        node["matched"] = False
    for requirement in requirements["step_requirements"]:
        count_satisfied += SAT(requirement, 'sr', workflow)
        count_requirements += 1
    nodes_id_satisfied = [node["node_id"] for node in workflow["nodes"] if node["matched"]]
    for requirement in requirements["dependency_requirements"]:
        if requirement["source_node_id"] in nodes_id_satisfied and requirement["target_node_id"] in nodes_id_satisfied:
            count_satisfied += SAT(requirement, 'dr', workflow)
            count_requirements += 1
    return round(count_satisfied / count_requirements, 4)

