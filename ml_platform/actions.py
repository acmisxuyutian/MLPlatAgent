import json
import os
import time
from ml_platform.ai_studio import AI_Studio
from utils.utils import get_project_root
from agents.data_loader import load_data

class Actions():

    def __init__(self):
        self.ai_studio = AI_Studio()
        self.project_root = get_project_root()
        self.widgets_path = os.path.join(self.project_root, r"data/ml_platform_data_example/widgets.json")
        with open(self.widgets_path, encoding='utf-8') as  f:
            self.widgets = json.load(f)
        self.widgets_name = [widget['widget_name'] for widget in self.widgets]
        self.relevant_widgets_names = []
        self.x = 30
        self.y = 300
        self.data_info = None
        self.total_tokens = 0
        self.ori_data_info = None

    def reset_XY(self):
        self.x = 30
        self.y = 300

    def execute_command(self, command_name, args):
        try:
            # Check if the method exists in the Workflow class
            if hasattr(self, command_name):
                method = getattr(self, command_name)
                result = method(args)
                return result
            else:
                return f'名为"{command_name}"的Action不存在！'

        except Exception as e:
            return f"Error: {e}"

    def add_node(self, args):
        if args["widget_name"] in self.relevant_widgets_names:

            w = self.widgets[self.widgets_name.index(args["widget_name"])]

            try:
                widgetParam = json.dumps(args["widget_params"])
            except:
                widgetParam = None

            node = {
                "pos_x": self.x,
                "pos_y": self.y,
                "abstractWidgetId": w["widget_id"],
                "abstractName": w["widget_name"],
                "name": args["node_name"],
                "image": w["image"],
                "package": w["package"],
                "widgetParam": widgetParam,
                "workflow_id": self.ai_studio.workflow_id
            }

            node = json.dumps(node)

            r = self.ai_studio.add_node(node)
            node_id = r["data"]["id"]

            self.x += 140

            return node_id

        else:
            raise Exception(f"当前任务可用组件中不存在组件{args['widget_name']}，添加的组件应该是{self.relevant_widgets_names}中的一个。")

    def delete_node(self, args):
        res = self.ai_studio.delete_node(args)
        if res["message"] == "请求成功":
            return f"节点{args['node_id']}删除成功"
        else:
            return f"节点{args['node_id']}删除失败，错误原因为：{res['message']}"

    def update_node_params(self, args):

        try:
            # 节点参数更新
            if type(args["node_params"]).__name__ == 'dict':
                node_params = args["node_params"]
            else:
                node_params = json.loads(args["node_params"])
            if "widget_name" not in args.keys():
                return "行动输入错误！缺少widget_name参数"
            index = self.widgets_name.index(args["widget_name"])
            if self.widgets[index]["params"] == []:
                return f"节点{args['node_id']}参数更新成功！"
            if args["widget_name"] == "SQL Table":
                eda_result, price, tokens = load_data(node_params["data_description"], args["node_id"])
                self.data_info = eda_result
                self.total_tokens += tokens
                self.run_workflow()
                self.ori_data_info = self.get_dataset_info(args["node_id"])
                return f"节点{args['node_id']}参数更新成功！"
            else:
                index = self.widgets_name.index(args["widget_name"])
                new_node_params = {}
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
        except:
            workflow = self.get_workflow()
            workflow_ids = [node["node_id"] for node in workflow["nodes"]]
            if args["node_id"] not in workflow_ids:
                raise Exception(f"节点ID{args['node_id']}不存在！请仔细核对工作流中是否存在该节点。")
            raise Exception(f"节点{args['node_id']}参数更新失败，可能是节点参数配置存在问题，注意查看当前任务可用组件中对该组件参数的介绍！")

    def add_edge(self, args):
        source_node_endpoints = []
        target_node_endpoints = []
        source_endpoint_id = 0
        target_endpoint_id = 0
        source_node = ""
        target_node = ""

        workflow_info = self.ai_studio.get_workflow()
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
        if source_node == "Logistic Regression" or source_node == "Linear Regression":
            new_source_node_endpoints = []
            for e in source_node_endpoints:
                if e["short_name"] != "data":
                    new_source_node_endpoints.append(e)

            source_node_endpoints = new_source_node_endpoints

        if source_node == "Data Sampler" and target_node == "Test & Score":
            data = {
                "source_node_id": args["source_node_id"],
                "source_endpoint_id": source_node_endpoints[0]["id"],
                "target_node_id": args["target_node_id"],
                "target_endpoint_id": target_node_endpoints[1]["id"]
            }
            res = self.ai_studio.add_edge(data)

            data = {
                "source_node_id": args["source_node_id"],
                "source_endpoint_id": source_node_endpoints[1]["id"],
                "target_node_id": args["target_node_id"],
                "target_endpoint_id": target_node_endpoints[0]["id"]
            }
            res = self.ai_studio.add_edge(data)

            if res["message"] == "添加连接！":
                return f"添加连接成功！"
            elif res["message"] == "控件间无法添加连接！":
                return f"无法添加从{data['begin_node_id']}到{data['end_node_id']}的边。"
            return res["message"]

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

        data = {
            "source_node_id": args["source_node_id"],
            "source_endpoint_id": source_endpoint_id,
            "target_node_id": args["target_node_id"],
            "target_endpoint_id": target_endpoint_id
        }
        res = self.ai_studio.add_edge(data)

        if res["message"] == "添加连接！":
            return f"添加连接成功！"
        elif res["message"] == "控件间无法添加连接！":
            return f"无法添加从{data['begin_node_id']}到{data['end_node_id']}的边。"
        return res["message"]

    def delete_edge(self, args):
        res = self.ai_studio.delete_edge(args)
        if res["message"]== "请求成功":
            return f"边{args['edge_id']}删除成功"
        return f"边{args['edge_id']}删除失败"

    def task_finish(self, args):

        return args["final_response"]

    def get_workflow(self):
        workflow_info = self.ai_studio.get_workflow()
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
            nodes.append({
                "node_id": n["node_id"],
                "widget_name": n["widget_name"],
                "node_name": n["node_name"],
                "node_params": n["node_params"]
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
                        "workflow_id": Workflow_id
                    }
                    self.ai_studio.run_widget(json.dumps(data, ensure_ascii=False))
                    data = {
                        "widget_id": str(args["node_id"]),
                        "workflow_id": Workflow_id
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
                            "workflow_id": Workflow_id
                        }
                        page += 1
                        print(page)
                        self.ai_studio.run_widget(json.dumps(data, ensure_ascii=False))

                        data = {
                            "widget_id": str(args["node_id"]),
                            "workflow_id": Workflow_id
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
                      "workflow_id": Workflow_id
                    }
                    self.ai_studio.run_widget(json.dumps(data, ensure_ascii=False))
                    data = {
                        "widget_id": str(args["node_id"]),
                        "workflow_id": Workflow_id
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

        print("工作流已清空！")

    def run_workflow(self):
        self.ai_studio.run_workflow()
        count_0 = 0
        while True:
            res = self.ai_studio.get_run_workflow_info()
            data = res["data"]
            if data[0]["status"] == 3:
                print("工作流运行成功！")
                return "", True
            elif data[0]["status"] == -1:
                print("工作流运行出错")
                print(res)
                return res, False
            elif data[0]["status"] == 0:
                print("无状态！")
                if count_0 == 0:
                    self.ai_studio.stop_workflow()
                    time.sleep(3)
                if count_0 == 1:
                    self.ai_studio.run_workflow()
                    time.sleep(3)
                count_0 += 1
            else:
                print("工作流运行中...")
                time.sleep(5)

    def get_widget_output(self, args):
        import pandas as pd
        max_try_times = 2
        for i in range(max_try_times):
            if i != 0:
                time.sleep(5)
            res = self.ai_studio.get_widget_output(str(args["node_id"]))
            datas = []
            data_number = len(res["data"])
            for i in range(data_number):
                datas.append(res["data"][i]["data"])
            if datas[0] == "暂无输出结果":
                self.run_workflow()
            else:
                break
        if datas[0] == "暂无输出结果":
            workflow = self.get_workflow()
            pending_nodes = [str(args["node_id"])]
            run_index = 0
            max_try_times = 5
            for i in range(max_try_times):
                res = self.ai_studio.get_widget_output(pending_nodes[run_index])

                datas = []
                data_number = len(res["data"])
                for i in range(data_number):
                    datas.append(res["data"][i]["data"])
                if datas[0] == "暂无输出结果":
                    for e in workflow["edges"]:
                        if e["target_node_id"] == pending_nodes[run_index]:
                            pending_nodes.append(e["source_node_id"])
                            run_index += 1
                else:
                    break
        if datas[0] == "暂无输出结果":
            return None
        else:
            try:
                columns = datas[0]["columns"]
                values = []
                for d in datas:
                    values.extend(d["data"])
                data_df = pd.DataFrame(values, columns=columns)
                return data_df
            except:
                return None

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

action_agent = Actions()
if __name__ == '__main__':
    print(action_agent.get_workflow())

