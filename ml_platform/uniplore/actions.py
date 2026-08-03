import json
import os
import time
from typing import Any, Mapping

import config as project_config
from ml_platform.action import Action, ActionConfigurationError, ActionError
from ml_platform.uniplore.ai_studio import AI_Studio
from utils.utils import get_project_root
from agents.data_loader import resolve_dataset

class PlatformAction(Action):

    def __init__(self, platform_config: Mapping[str, Any]):
        access_token = platform_config.get("access_token")
        workflow_id = platform_config.get("workflow_id")
        api_url = platform_config.get("api_url")
        if not isinstance(access_token, str) or not access_token.strip():
            raise ActionConfigurationError(
                "UNIPLORE_CONFIG['access_token'] must be a non-empty string"
            )
        if (
            workflow_id is None
            or isinstance(workflow_id, bool)
            or not isinstance(workflow_id, (int, str))
            or (
                isinstance(workflow_id, str)
                and not workflow_id.strip()
            )
        ):
            raise ActionConfigurationError(
                "UNIPLORE_CONFIG['workflow_id'] is required"
            )
        if not isinstance(api_url, str) or not api_url.strip():
            raise ActionConfigurationError(
                "UNIPLORE_CONFIG['api_url'] must be a non-empty string"
            )
        self.project_root = get_project_root()
        super().__init__(
            platform_config,
            platform_name="uniplore",
            widgets_path=os.path.join(
                self.project_root,
                "ml_platform",
                "uniplore",
                "widgets.json",
            ),
        )
        self.ai_studio = AI_Studio(
            workflow_id=workflow_id,
            accesstoken=access_token,
            root_url=api_url,
        )
        self.workflow_id = workflow_id
        self.mysql_config = dict(
            getattr(project_config, "MySQL_Config", {})
        )
        self._sql_public_params = {}

    def reset(self):
        super().reset()

    def reset_XY(self):
        self.x = 30
        self.y = 300

    def execute_command(self, command_name, args=None):
        return super().execute_command(command_name, args)

    def add_node(self, args):
        widget_name = args.get("widget_name")
        if widget_name not in self.widgets_by_name:
            raise ActionError(
                f"当前 Uniplore 组件库中不存在组件 {widget_name!r}"
            )
        node_name = args.get("node_name")
        if not isinstance(node_name, str) or not node_name.strip():
            raise ActionError("node_name 必须是非空字符串")
        if widget_name in self.relevant_widgets_names:

            w = self.widgets[self.widgets_name.index(widget_name)]

            widget_params = args.get("widget_params")
            widgetParam = (
                json.dumps(widget_params, ensure_ascii=False)
                if isinstance(widget_params, dict)
                else None
            )

            node = {
                "pos_x": self.x,
                "pos_y": self.y,
                "abstractWidgetId": w["widget_id"],
                "abstractName": w["widget_name"],
                "name": node_name,
                "image": w["image"],
                "package": w["package"],
                "widgetParam": widgetParam,
                "workflow_id": self.ai_studio.workflow_id
            }

            node = json.dumps(node)

            r = self.ai_studio.add_node(node)
            if (
                not isinstance(r, dict)
                or not isinstance(r.get("data"), dict)
                or r["data"].get("id") is None
            ):
                raise ActionError("Uniplore 未返回有效的新节点 ID")
            node_id = r["data"]["id"]

            self.x += 140

            return node_id

        else:
            raise ActionError(
                f"当前任务可用组件中不存在组件 {widget_name!r}，"
                f"添加的组件必须属于 {self.relevant_widgets_names}"
            )

    def delete_node(self, args):
        res = self.ai_studio.delete_node(args)
        if isinstance(res, dict) and res.get("message") == "请求成功":
            self._sql_public_params.pop(str(args["node_id"]), None)
            return f"节点{args['node_id']}删除成功"
        message = res.get("message") if isinstance(res, dict) else res
        raise ActionError(
            f"节点 {args.get('node_id')} 删除失败：{message}"
        )

    def update_node_params(self, args):

        try:
            # 节点参数更新
            if type(args["node_params"]).__name__ == 'dict':
                node_params = args["node_params"]
            else:
                node_params = json.loads(args["node_params"])
            if "widget_name" not in args.keys():
                return "行动输入错误！缺少widget_name参数"
            if args["widget_name"] not in self.widgets_by_name:
                raise ActionError(
                    f"当前 Uniplore 组件库中不存在组件 "
                    f"{args['widget_name']!r}"
                )
            self.validate_node_params(args["widget_name"], node_params)
            index = self.widgets_name.index(args["widget_name"])
            if self.widgets[index]["params"] == []:
                return f"节点{args['node_id']}参数更新成功！"
            if args["widget_name"] == "SQL Table":
                data_description = node_params.get("data_description")
                if (
                    not isinstance(data_description, str)
                    or not data_description.strip()
                ):
                    raise ValueError(
                        "SQL Table 的 data_description 必须是非空字符串"
                    )
                begin_time = time.time()
                resolution = resolve_dataset(
                    data_description,
                    "mysql",
                    self.mysql_config,
                )
                end_time = time.time()
                self.data_retrieval_time += (end_time - begin_time)
                self.data_info = resolution.eda_result
                self.input_tokens += resolution.input_tokens
                self.output_tokens += resolution.output_tokens
                self.total_tokens += (
                    resolution.input_tokens + resolution.output_tokens
                )

                widget_params = {
                    "type": "2",
                    "server": self.mysql_config["server"],
                    "database": self.mysql_config["database"],
                    "schema": "public",
                    "port": self.mysql_config["port"],
                    "username": self.mysql_config["username"],
                    "password": self.mysql_config["password"],
                    "dbType": None,
                    "load_type": "table",
                    "getDatabaseFlag": False,
                    "selectedDatasource": None,
                    "selectedDatasourceUserId": -1,
                    "isNewIDIS": True,
                    "table_name": resolution.table_name,
                    "save2db": False,
                    "sql_code": "",
                    "dump_table": "my_table",
                    "selectedDataset": "",
                    "dataset_name": None,
                    "dataset_sql": None,
                    "field": None,
                    "interact_type": 0,
                    "attr_mapping": resolution.attr_mapping,
                }
                payload = json.dumps(
                    {
                        "workflow_id": self.ai_studio.workflow_id,
                        "param_info": {
                            str(args["node_id"]): widget_params
                        },
                    },
                    ensure_ascii=False,
                )
                update_result = self.ai_studio.update_node_params(payload)
                if (
                    not isinstance(update_result, dict)
                    or update_result.get("message") != "请求成功"
                ):
                    message = (
                        update_result.get("message")
                        if isinstance(update_result, dict)
                        else update_result
                    )
                    raise Exception(
                        "Uniplore SQL Table 参数更新失败："
                        f"{message}"
                    )

                self._sql_public_params[str(args["node_id"])] = {
                    "data_description": data_description,
                    "table_name": resolution.table_name,
                }
                self.run_workflow()
                self.ori_data_info = self.get_dataset_info(args["node_id"])
                return f"节点{args['node_id']}参数更新成功！"
            else:
                index = self.widgets_name.index(args["widget_name"])
                new_node_params = {}
                # {
                #     "key": 18,
                #     "name": "FriedPotato_Consumption",
                #     "data_type": "numeric",
                #     "role": "feature"
                # }
                if args["widget_name"] == "Select Columns":
                    if "targets" not in node_params:
                        node_params["targets"] = []
                    else:
                        node_params["targets"] = [node_params["targets"]]
                    if "ignores" not in node_params:
                        node_params["ignores"] = []
                    if "features" not in node_params:
                        node_params["features"] = []
                    data = self.ori_data_info["columns_info"]
                    column_id_map = {c["name"]: c["key"] for c in data}
                    visited = {c["name"]: False for c in data}

                    params = ["targets", "ignores", "features"]
                    for param in params:
                        new_node_params[param] = []
                        for column in node_params[param]:
                            if column not in column_id_map:
                                continue
                            visited[column] = True
                            new_node_params[param].append(column_id_map[column])

                    for column in data:
                        if visited[column["name"]]:
                            continue
                        if column["role"] == "feature":
                            new_node_params["features"].append(column["key"])
                        elif column["role"] == "target":
                            new_node_params["targets"].append(column["key"])
                        else:
                            new_node_params["ignores"].append(column["key"])
                elif args["widget_name"] == "Change Domain":
                    # {'categoricalAttrs': ['BMI_Category'], 'numericAttrs': ['BMI']} textAttrs,datetimeAttrs
                    data = self.ori_data_info["columns_info"]
                    column_map = {c["name"]: c for c in data}
                    if "textAttrs" in node_params:
                        for attr in node_params["textAttrs"]:
                            if attr not in column_map:
                                continue
                            column_map[attr]["data_type"] = "text"
                    if "numericAttrs" in node_params:
                        for attr in node_params["numericAttrs"]:
                            if attr not in column_map:
                                continue
                            column_map[attr]["data_type"] = "numeric"
                    if "datetimeAttrs" in node_params:
                        for attr in node_params["datetimeAttrs"]:
                            if attr not in column_map:
                                continue
                            column_map[attr]["data_type"] = "datetime"
                    if "categoricalAttrs" in node_params:
                        for attr in node_params["categoricalAttrs"]:
                            if attr not in column_map:
                                continue
                            column_map[attr]["data_type"] = "categorical"

                    new_node_params = {
                        "textAttrs": [],
                        "numericAttrs": [],
                        "datetimeAttrs": [],
                        "categoricalAttrs": [],
                    }
                    for c in column_map:
                        if column_map[c]["data_type"] == "text":
                            new_node_params["textAttrs"].append(column_map[c]["key"])
                        elif column_map[c]["data_type"] == "numeric":
                            new_node_params["numericAttrs"].append(column_map[c]["key"])
                        elif column_map[c]["data_type"] == "datetime":
                            new_node_params["datetimeAttrs"].append(column_map[c]["key"])
                        elif column_map[c]["data_type"] == "categorical":
                            new_node_params["categoricalAttrs"].append(column_map[c]["key"])
                        else:
                            pass
                elif args["widget_name"] == "Edit Domain":
                    new_node_params = {
                        "json_obj": {
                            node_params["column_name"]: {
                                "nameMapping": {
                                    "originName": node_params["column_name"],
                                    "currentName": node_params["column_name"]
                                },
                                "valueMappings": node_params["valueMappings"]
                            }
                        }
                    }
                else:
                    for p in self.widgets[index]["params"]:
                        if p["name"] in node_params:
                            new_node_params[p["name"]] = node_params[p["name"]]
                        else:
                            new_node_params[p["name"]] = p["default"]

                    if new_node_params == {}:
                        new_node_params = node_params
                if args["widget_name"] == "File":
                    if new_node_params["filename"] == "iris.csv":
                        new_node_params["mapping"] = {
                            "4": {
                                "type": 1,
                                "role": 1
                            }
                        }
                    elif new_node_params["filename"] == "wine.csv":
                        new_node_params["mapping"] = {
                            "13": {
                                "type": 1,
                                "role": 1
                            }
                        }
                    elif new_node_params["filename"] == "breast_cancer.csv":
                        new_node_params["mapping"] = {
                            "30": {
                                "type": 1,
                                "role": 1
                            }
                        }
                    else:
                        pass
                data = {
                    "workflow_id": self.ai_studio.workflow_id,
                    "param_info": {str(args["node_id"]): new_node_params},
                }

                data = json.dumps(data, ensure_ascii=False)
                result = self.ai_studio.update_node_params(data)
                if args["widget_name"] == "File":
                    self.run_workflow()
                    self.data_info = self.get_dataset_info(args["node_id"])
                    self.ori_data_info = self.data_info
                    from utils.utils import update_data_info
                    update_data_info(dataset_info=json.dumps(self.data_info,ensure_ascii=False))

                if result["message"] == '更新控件参数失败：list index out of range':
                    raise Exception(f"节点{args['node_id']}参数更新失败！因为该节点不存在，请查看工作流信息检查节点ID是否正确！")
                if result["message"] == '请求成功':
                    return f"节点{args['node_id']}参数更新成功！"
                else:
                    raise Exception(f"节点{args['node_id']}参数更新失败，错误原因为：{result['message']}")
        except Exception as e:
            workflow = self.get_workflow()
            workflow_ids = {
                str(node["node_id"]) for node in workflow["nodes"]
            }
            if str(args["node_id"]) not in workflow_ids:
                raise Exception(f"节点ID{args['node_id']}不存在！请仔细核对工作流中是否存在该节点。")
            raise Exception(f"节点{args['node_id']}参数更新失败，错误信息如下：\n{e}")


    def add_edge(self, args):
        # # 添加边 Test
        # data = {
        #     "source_node_id": 993,
        #     "source_endpoint_id": 1234,
        #     "target_node_id": 999,
        #     "target_endpoint_id": 1257
        # }
        #
        # print(ai_studio.add_edge(data))

        source_node_endpoints = []
        target_node_endpoints = []
        source_endpoint_id = None
        target_endpoint_id = None
        source_node = ""
        target_node = ""

        workflow_info = self.ai_studio.get_workflow()
        if (
            not isinstance(workflow_info, dict)
            or not isinstance(workflow_info.get("nodes"), list)
            or not isinstance(workflow_info.get("edges"), list)
        ):
            raise ActionError("无法从 Uniplore 获取有效的当前工作流")
        for n in workflow_info["nodes"]:
            if n["node_id"] == str(args["source_node_id"]):
                source_node = n["widget_name"]
                for e in n["endpoints"]:
                    if e["type"] == "source":
                        source_node_endpoints.append(e)

            elif n["node_id"] == str(args["target_node_id"]):
                target_node = n["widget_name"]
                for e in n["endpoints"]:
                    if e["type"] == "target":
                        target_node_endpoints.append(e)
        if not source_node:
            raise ActionError(
                f"源节点 {args.get('source_node_id')} 不存在"
            )
        if not target_node:
            raise ActionError(
                f"目标节点 {args.get('target_node_id')} 不存在"
            )
        if not source_node_endpoints or not target_node_endpoints:
            raise ActionError(
                f"节点 {args.get('source_node_id')} 与 "
                f"{args.get('target_node_id')} 没有可用连接端点"
            )
        if source_node == "Logistic Regression" or source_node == "Linear Regression":
            new_source_node_endpoints = []
            for e in source_node_endpoints:
                if e["short_name"] != "data":
                    new_source_node_endpoints.append(e)

            source_node_endpoints = new_source_node_endpoints

        if source_node == "Data Sampler" and target_node == "Test & Score":
            if (
                len(source_node_endpoints) < 2
                or len(target_node_endpoints) < 2
            ):
                raise ActionError(
                    "Data Sampler 到 Test & Score 缺少所需连接端点"
                )
            data = {
                "source_node_id": args["source_node_id"],
                "source_endpoint_id": source_node_endpoints[0]["id"],
                "target_node_id": args["target_node_id"],
                "target_endpoint_id": target_node_endpoints[1]["id"]
            }
            first_result = self.ai_studio.add_edge(data)
            if (
                not isinstance(first_result, dict)
                or first_result.get("message") != "添加连接！"
            ):
                message = (
                    first_result.get("message")
                    if isinstance(first_result, dict)
                    else first_result
                )
                raise ActionError(f"添加第一条数据划分连接失败：{message}")

            data = {
                "source_node_id": args["source_node_id"],
                "source_endpoint_id": source_node_endpoints[1]["id"],
                "target_node_id": args["target_node_id"],
                "target_endpoint_id": target_node_endpoints[0]["id"]
            }
            res = self.ai_studio.add_edge(data)

            if isinstance(res, dict) and res.get("message") == "添加连接！":
                return f"添加连接成功！"
            message = res.get("message") if isinstance(res, dict) else res
            raise ActionError(
                f"无法添加从 {args.get('source_node_id')} 到 "
                f"{args.get('target_node_id')} 的第二条连接：{message}"
            )

        elif source_node == "Data Sampler":
            if target_node == "Predictions":
                source_endpoint_id = source_node_endpoints[0]["id"]
            else:
                workflow = workflow_info
                source_ids = [e["source_endpoint_id"] for e in workflow["edges"]]
                if source_node_endpoints[1]["id"] not in source_ids:
                    source_endpoint_id = source_node_endpoints[1]["id"]
                elif source_node_endpoints[0]["id"] not in source_ids:
                    source_endpoint_id = source_node_endpoints[0]["id"]
                else:
                    source_endpoint_id = source_node_endpoints[1]["id"]

            for te in target_node_endpoints:
                if te["short_name"] in ["data", "tsd"]:
                    target_endpoint_id = te["id"]
                    break
        elif target_node == "Test & Score":
            workflow = workflow_info
            target_ids = [e["target_endpoint_id"] for e in workflow["edges"]]
            tag = False
            for se in source_node_endpoints:
                for te in target_node_endpoints:
                    if se["short_name"] == te["short_name"] or \
                            se["short_name"] == "data" and te["short_name"] == "trndt" or \
                            se["short_name"] == "data" and te["short_name"] == "tstdt":
                        if te["id"] not in target_ids:
                            tag = True
                            source_endpoint_id = se["id"]
                            target_endpoint_id = te["id"]
                            break
                if tag:
                    break
            if not tag:
                for se in source_node_endpoints:
                    for te in target_node_endpoints:
                        if se["short_name"] == te["short_name"] or \
                                se["short_name"] == "data" and te["short_name"] == "trndt" or \
                                se["short_name"] == "data" and te["short_name"] == "tstdt":
                            source_endpoint_id = se["id"]
                            target_endpoint_id = te["id"]
        elif source_node == "Predictions":
            if target_node_endpoints[0]["short_name"]== "evr":
                source_endpoint_id = source_node_endpoints[0]["id"]
            else:
                source_endpoint_id = source_node_endpoints[1]["id"]
            target_endpoint_id = target_node_endpoints[0]["id"]

        elif source_node == "Predictions":
            if target_node_endpoints[0]["short_name"]== "evr":
                source_endpoint_id = source_node_endpoints[0]["id"]
            else:
                source_endpoint_id = source_node_endpoints[1]["id"]
            target_endpoint_id = target_node_endpoints[0]["id"]

        else:
            for se in source_node_endpoints:
                for te in target_node_endpoints:
                    if se["short_name"] == te["short_name"] or \
                            se["short_name"] == "data" and te["short_name"] == "tsd" or \
                            se["short_name"] == "fore" and te["short_name"] == "data":
                        source_endpoint_id = se["id"]
                        target_endpoint_id = te["id"]

        if source_endpoint_id is None or target_endpoint_id is None:
            raise ActionError(
                f"无法为 {source_node!r} 到 {target_node!r} "
                "选择兼容的连接端点"
            )

        data = {
            "source_node_id": args["source_node_id"],
            "source_endpoint_id": source_endpoint_id,
            "target_node_id": args["target_node_id"],
            "target_endpoint_id": target_endpoint_id
        }
        res = self.ai_studio.add_edge(data)

        if isinstance(res, dict) and res.get("message") == "添加连接！":
            return f"添加连接成功！"
        message = res.get("message") if isinstance(res, dict) else res
        raise ActionError(
            f"无法添加从 {args.get('source_node_id')} 到 "
            f"{args.get('target_node_id')} 的边：{message}"
        )

    def delete_edge(self, args):
        res = self.ai_studio.delete_edge(args)
        if isinstance(res, dict) and res.get("message") == "请求成功":
            return f"边{args['edge_id']}删除成功"
        message = res.get("message") if isinstance(res, dict) else res
        raise ActionError(
            f"边 {args.get('edge_id')} 删除失败：{message}"
        )

    def task_finish(self, args):

        return args["final_response"]

    def get_workflow(self):
        workflow_info = self.ai_studio.get_workflow()
        if (
            not isinstance(workflow_info, dict)
            or not isinstance(workflow_info.get("nodes"), list)
            or not isinstance(workflow_info.get("edges"), list)
        ):
            raise ActionError("无法从 Uniplore 获取有效的当前工作流")
        edges = []
        nodes = []
        for e in workflow_info["edges"]:
            is_exist = False
            for edge in edges:
                if edge["source_node_id"] == e["source_node_id"] and edge["target_node_id"] == e["target_node_id"]:
                    is_exist = True
                    break
            if not is_exist:
                edges.append({
                    "edge_id": e["edge_id"],
                    "source_node_id": e["source_node_id"],
                    "target_node_id": e["target_node_id"]
                })

        for n in  workflow_info["nodes"]:
            node_params = n.get("node_params", {})
            if n["widget_name"] == "SQL Table":
                if isinstance(node_params, str):
                    try:
                        node_params = json.loads(node_params)
                    except json.JSONDecodeError:
                        node_params = {}
                if not isinstance(node_params, dict):
                    node_params = {}
                public_params = dict(
                    self._sql_public_params.get(str(n["node_id"]), {})
                )
                table_name = node_params.get("table_name")
                if (
                    "table_name" not in public_params
                    and isinstance(table_name, str)
                    and table_name.strip()
                ):
                    public_params["table_name"] = table_name.strip()
                public_params.setdefault("data_description", "")
                node_params = public_params
            nodes.append({
                "node_id": n["node_id"],
                "widget_name": n["widget_name"],
                "node_name": n["node_name"],
                "node_params": node_params
            })
        return {"edges": edges, "nodes": nodes}

    def get_node_results(self, args):
        """
    运行状态说明
    FAILED = -1 运行出错
    NOSTATE = 0 无状态
    PENDING = 1 等待运行
    RUNNING = 2 运行中
    SUCCEED = 3 运行成功
        """
        import pandas as pd
        workflow_info = self.ai_studio.get_workflow()
        for node in workflow_info["nodes"]:
            if node["node_id"] == str(args["node_id"]):
                if node["widget_name"] == "Test & Score":
                    # 使用10折交叉验证
                    data = {
                        "widget_id": str(args["node_id"]),
                        "interact_type": 1,
                        "resampling_type": 0,
                        "n_folds": 10,
                        "cv_stratified": True,
                        "fold_feature": None,
                        "n_repeats": 3,
                        "sample_size": 5,
                        "shuffle_stratified": True,
                        "target_idx": 0,
                        "workflow_id": self.ai_studio.workflow_id
                    }
                    self.ai_studio.run_widget(json.dumps(data, ensure_ascii=False))
                    data = {
                        "widget_id": str(args["node_id"]),
                        "workflow_id": self.ai_studio.workflow_id
                    }

                    while True:
                        run_info = self.ai_studio.get_run_widget_status(json.dumps(data, ensure_ascii=False))
                        if run_info["status"] == 3:
                            print(f"节点{args['node_id']}运行成功")
                            if "confusion_matrixes" in run_info["data"]:
                                d = {
                                    "confusion_matrixes": run_info["data"]["confusion_matrixes"],
                                    "colssum": run_info["data"]["colssum"],
                                    "rowssum": run_info["data"]["rowssum"],
                                    "headers": run_info["data"]["headers"],
                                    "total": run_info["data"]["total"],
                                    "model_params": run_info["data"]["model_params"],
                                }
                                return d
                            elif "score_table" in run_info["data"]:
                                return run_info["data"]["score_table"]
                            else:
                                return run_info["data"]
                        elif run_info["status"] == -1:
                            print(f"节点{args['node_id']}运行失败")
                            return {}
                        else:
                            print(f"节点{args['node_id']}运行中...")
                            print(json.dumps(run_info, ensure_ascii=False, indent=4))
                            time.sleep(3)
                if node["widget_name"] == "Data Table":
                    page = 1
                    page_size = 100
                    result = []
                    while True:
                        data = {
                            "widget_id": str(args["node_id"]),
                            "page": page,
                            "page_size": page_size,
                            "workflow_id": self.ai_studio.workflow_id
                        }
                        page += 1
                        print(page)
                        self.ai_studio.run_widget(json.dumps(data, ensure_ascii=False))

                        data = {
                            "widget_id": str(args["node_id"]),
                            "workflow_id": self.ai_studio.workflow_id
                        }

                        while True:
                            run_info = self.ai_studio.get_run_widget_status(json.dumps(data, ensure_ascii=False))
                            if run_info["status"] == 3:
                                result.extend(run_info["data"]["table_data"])
                                print(f"成功获取{len(result)}条数据")
                                if len(result) >= run_info["data"]["metadata"]["n_sample"]:
                                    return pd.DataFrame(result)
                                else:
                                    break
                            elif run_info["status"] == -1:
                                print(f"节点{args['node_id']}运行失败")
                                return {}
                            else:
                                print(f"节点{args['node_id']}运行中...")
                                print(json.dumps(run_info, ensure_ascii=False, indent=4))
                                time.sleep(3)
                if node["widget_name"] == "Save Data":
                    data = {
                      "filename": "mydata",
                      "ext": ".csv",
                      "widget_id": str(args["node_id"]),
                      "workflow_id": self.ai_studio.workflow_id
                    }
                    self.ai_studio.run_widget(json.dumps(data, ensure_ascii=False))
                    data = {
                        "widget_id": str(args["node_id"]),
                        "workflow_id": self.ai_studio.workflow_id
                    }

                    while True:
                        run_info = self.ai_studio.get_run_widget_status(json.dumps(data, ensure_ascii=False))
                        if run_info["status"] == 3:
                            print(f"节点{args['node_id']}运行成功")
                            return run_info["data"]["url"]

                        elif run_info["status"] == -1:
                            print(f"节点{args['node_id']}运行失败")
                            return {}
                        else:
                            print(f"节点{args['node_id']}运行中...")
                            print(json.dumps(run_info, ensure_ascii=False, indent=4))
                            time.sleep(3)

        return f"工作流中不存在节点{args['node_id']}"

    def get_dataset_info(self, node_id):
        return self.ai_studio.get_data_info(int(node_id))
        # 获取指定节点的输入数据集信息
        # data = self.ai_studio.get_data_info(1203)
        # self.ai_studio.run_workflow()
        # while (True):
        #     time.sleep(3)
        #     res = self.ai_studio.get_run_workflow_info()
        #     data = res["data"]
        #     if data[0]["status"] == 3:
        #         time.sleep(8)
        #         return self.ai_studio.get_data_info(args["node_id"])
        #
        #     elif data[0]["status"] == -1:
        #         print("工作流运行出错")
        #         return {}
        #     else:
        #         print("工作流运行中")

    def clear_workflow(self):
        workflow_info = self.get_workflow()
        for n in workflow_info["nodes"]:
            self.delete_node({"node_id": n["node_id"]})

        remaining = self.get_workflow()["nodes"]
        if remaining:
            remaining_ids = [str(node["node_id"]) for node in remaining]
            raise ActionError(
                "Uniplore 工作流清空后仍存在节点："
                + ", ".join(remaining_ids)
            )

    def run_workflow(self):
        self.ai_studio.run_workflow()
        print("工作流运行中...")
        time.sleep(10)

        # max_try = 3
        # try_times = 0
        # count = 0
        # while True:
        #     res = self.ai_studio.get_run_workflow_info()
        #     data = res["data"]
        #     if data[0]["status"] == 3:
        #         print("工作流运行成功！")
        #         return "", True
        #     elif data[0]["status"] == -1:
        #         print("工作流运行出错")
        #         print(res)
        #         return res, False
        #     elif data[0]["status"] == 0:
        #         print("无状态！")
        #         if count==0:
        #             self.ai_studio.stop_workflow()
        #             time.sleep(1)
        #         else:
        #             count += 1
        #     else:
        #         print("工作流运行中...")
        #         time.sleep(3)
        #     try_times += 1
        #     if try_times > max_try:
        #         print("尝试次数过多，退出")
        #         return res, False

    def get_widget_output(self, args):
        dataset_id = None
        import pandas as pd
        max_try_times = 2
        for i in range(max_try_times):
            if i != 0:
                time.sleep(5)
            res = self.ai_studio.get_widget_output(str(args["node_id"]))
            dataset_id = args["node_id"]
            datas = []
            data_number = len(res["data"])
            for i in range(data_number):
                datas.append(res["data"][i]["data"])
            if datas[0] in ["暂无输出结果", "None"]:
                self.run_workflow()
                # workflow = self.get_workflow()
                # pending_nodes = [str(args["node_id"])]
                # run_index = 0
                # while run_index >= 0:
                #     is_run_success = False
                #     data = {
                #         "widget_id": pending_nodes[run_index],
                #         "workflow_id": self.ai_studio.workflow_id
                #     }
                #     data_str = json.dumps(data, ensure_ascii=False)
                #     self.ai_studio.run_widget(data_str)
                #     print(pending_nodes)
                #     print(run_index)
                #     while True:
                #         run_info = self.ai_studio.get_run_widget_status(data_str)
                #         if run_info["status"] == 3:
                #             print(f"节点{args['node_id']}运行成功")
                #             is_run_success = True
                #             break
                #
                #         elif run_info["status"] == -1:
                #             print(f"节点{args['node_id']}运行失败")
                #             break
                #         else:
                #             print(f"节点{args['node_id']}运行中...")
                #             time.sleep(1)
                #     if not is_run_success:
                #         for e in workflow["edges"]:
                #             if e["target_node_id"] == pending_nodes[run_index]:
                #                 pending_nodes.append(e["source_node_id"])
                #                 run_index += 1
                #     else:
                #         pending_nodes.pop(run_index)
                #         run_index -= 1
            else:
                break
        if datas[0] in ["暂无输出结果", "None"]:
            workflow = self.get_workflow()
            pending_nodes = [str(args["node_id"])]
            run_index = 0
            max_try_times = 5
            for i in range(max_try_times):
                res = self.ai_studio.get_widget_output(pending_nodes[run_index])
                dataset_id = pending_nodes[run_index]

                datas = []
                data_number = len(res["data"])
                for i in range(data_number):
                    datas.append(res["data"][i]["data"])
                if datas[0] in ["暂无输出结果", "None"]:
                    for e in workflow["edges"]:
                        if e["target_node_id"] == pending_nodes[run_index]:
                            pending_nodes.append(e["source_node_id"])
                            run_index += 1
                else:
                    break
        if datas[0] in ["暂无输出结果", "None"]:
            return None, None
        else:
            try:
                columns = datas[0]["columns"]
                values = []
                for d in datas:
                    if d is not None and d != "None":
                        values.extend(d["data"])
                data_df = pd.DataFrame(values, columns=columns)
                return data_df, dataset_id
            except:
                return None, None

    def nni_hpo(self, args):
        res = self.ai_studio.hpo(args)
        if res is None:
            return None
        else:
            return res["data"]

    def get_model_score(self, args):
        res = self.ai_studio.get_model_score(args)
        if res is None:
            return None
        else:
            return res["data"]["default_score"]

Actions = PlatformAction
# action_agent.modif_workflow_status()

# print(action_agent.run_workflow())
# print(action_agent.get_workflow())
# print(action_agent.get_widget_output({"node_id": 2240}))
# print(json.dumps(action_agent.nni_hpo({"widget_name": "XGBoost", "metric": "auc_roc"}),indent=4, ensure_ascii=False))
# print(action_agent.get_model_score({"widget_name": "XGBoost", "metric": "auc_roc"}))
# print(action_agent.get_model_score({"widget_name": "Linear Regression", "metric": "auc_roc"}))
# widget_model_map = {
#     "AdaBoost": "adaboost",
#     "Gradient Boosting Decision Tree": "gbdt",
#     "KNN": "knn",
#     "LightGBM": "lightgbm",
#     "Logistic Regression": "logistic_regression",
#     "Linear Regression": "linear_regression",
#     "Neural Network": "neural_network",
#     "Random Forest": "random_forest",
#     "SVM": "svm",
#     "Tree": "decision_tree",
#     "XGBoost": "xgboost",
#     "Stochastic Gradient Descent": "sgd",
#     "Naive Bayes": "bayes"}
# for m in widget_model_map:
#     print(action_agent.get_model_score({"widget_name": m, "metric": "auc_roc"}))
# action_agent.run_workflow()
# action_agent.run_workflow()
# print(json.dumps(action_agent.get_dataset_info(1183), ensure_ascii=False))
# print(action_agent.get_workflow())
# print(action_agent.get_widget_output({"node_id": 1189}))
# print(action_agent.get_widget_output({"node_id": 1189}))
# action_agent.update_node_params({"node_id": "1189", "widget_name": "Feature Constructor", "node_params": {}})
# # 设置测试数据特征构建参数
# test_feature_params = {
#     "expressions": [
#         {
#             "key": 0,
#             "name": "BMI_Category",
#             "expression": "np.where(df['BMI'] < 18.5, 'Underweight', np.where(df['BMI'] <= 24.9, 'Normal weight', np.where(df['BMI'] <= 29.9, 'Overweight', 'Obesity')))",
#             "type": "1",
#             "image": "C",
#             "categories": "Underweight, Normal weight, Overweight, Obesity"
#         }
#     ]
# }
# args = {"node_id": '730', "widget_name": 'Feature Constructor', "node_params": test_feature_params}
# print(action_agent.update_node_params(args))
# 将数据划分节点的训练集输出连接到训练数据预处理节点
# print(action_agent.add_edge({"source_node_id": '392', "target_node_id": '396'}))
# # 将数据划分节点的测试集输出连接到测试数据预处理节点
# action_agent.add_edge({"source_node_id": '290', "target_node_id": '292'})
# print(json.dumps(action_agent.get_workflow(), indent=4, ensure_ascii=False))
# print(json.dumps(action_agent.get_dataset_info('3149'), indent=4, ensure_ascii=False))
# args = {"source_node_id": 2818, "target_node_id": 2819}
# print(action_agent.add_edge(args))
# action_agent.run_workflow()
# print(action_agent.get_workflow())
# action_agent.add_edge({"source_node_id": '2129', "target_node_id": '2130'})
# action_agent.run_workflow()
# res = action_agent.get_node_results({"node_id": 2185})
# print(res)
# print(action_agent.get_workflow())
# # action_agent.run_workflow()
# print(action_agent.get_dataset_info({"node_id": 2076}))
# node_results = action_agent.get_node_results({"node_id": 2075})
# print(node_results)
# print(json.dumps(node_results, ensure_ascii=False, indent=4))
# data = node_results["table_data"]
# import pandas as pd
# data_frame = pd.DataFrame(data)
# print(data_frame.to_string())

# action_agent.relevant_widgets_names = ["SQL Table"]
# workflow = action_agent.get_workflow()
# print("#################### 边信息 ####################")
# for e in workflow["edges"]:
#     print(e)
# print("#################### 节点信息 ####################")
# for n in workflow["nodes"]:
#     print(n["node_id"])
#     print(n["node_name"])
#     print(n["node_params"])
#
# action = "update_node_params"
#
# action_input = {"node_id": 1879, "node_name": "Logistic Regression",
#                 "node_params": {'name': 'Logistic Regression', 'penalty': 'l2', 'C': 1.0}}
# action_result = action_agent.execute_command(action, action_input)
# result = []
# for i in range(10):
#     action_result = action_agent.get_node_results({"node_id": 1868})
#     result.append(action_result)
#
# for r in result:
#     print(r)
#
#
# # action_agent.add_node({"widget_name":"SQL Table"})
#
# attr_mapping = {
#     "PassengerId": {
#         "key": 0,
#         "name": "PassengerId",
#         "type": 2,
#         "role": 2
#     },
#     "Survived": {
#         "key": 1,
#         "name": "Survived",
#         "type": 1,
#         "role": 1
#     },
#     "Pclass": {
#         "key": 2,
#         "name": "Pclass",
#         "type": 1,
#         "role": 0
#     },
#     "Name": {
#         "key": 3,
#         "name": "Name",
#         "type": 3,
#         "role": 0
#     },
#     "Sex": {
#         "key": 4,
#         "name": "Sex",
#         "type": 1,
#         "role": 0
#     },
#     "Age": {
#         "key": 5,
#         "name": "Age",
#         "type": 2,
#         "role": 0
#     },
#     "SibSp": {
#         "key": 6,
#         "name": "SibSp",
#         "type": 2,
#         "role": 0
#     },
#     "Parch": {
#         "key": 7,
#         "name": "Parch",
#         "type": 2,
#         "role": 0
#     },
#     "Ticket": {
#         "key": 8,
#         "name": "Ticket",
#         "type": 3,
#         "role": 2
#     },
#     "Fare": {
#         "key": 9,
#         "name": "Fare",
#         "type": 2,
#         "role": 0
#     },
#     "Cabin": {
#         "key": 10,
#         "name": "Cabin",
#         "type": 3,
#         "role": 0
#     },
#     "Embarked": {
#         "key": 11,
#         "name": "Embarked",
#         "type": 1,
#         "role": 0
#     }
# }
#
# widget_params = {
#     "type": "2",
#     "server": MySQL_Config["server"],
#     "database": MySQL_Config["database"],
#     "schema": "public",
#     "port": MySQL_Config["port"],
#     "username": MySQL_Config["username"],
#     "password": MySQL_Config["password"],
#     "dbType": None,
#     "load_type": "table",
#     "getDatabaseFlag": False,
#     "selectedDatasource": None,
#     "selectedDatasourceUserId": -1,
#     "isNewIDIS": True,
#     "table_name": "titanic",
#     "save2db": False,
#     "sql_code": "",
#     "dump_table": "my_table",
#     "selectedDataset": "",
#     "dataset_name": None,
#     "dataset_sql": None,
#     "field": None,
#     "interact_type": 0,
#     "attr_mapping": attr_mapping
# }
#
# print(action_agent.update_node_params({"node_id": "10564", "widget_name": "SQL Table", "node_params": {"data_description":"bank customer training dataset"}}))

