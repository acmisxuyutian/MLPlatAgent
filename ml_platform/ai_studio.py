# -*- coding: utf-8 -*-
import json
import requests
import os
from utils.utils import get_project_root
new_widgets_path = os.path.join(get_project_root(), r"data/ml_platform_data_example/widgets.json")
from config import Workflow_id, Accesstoken, AI_STUDIO_URL

class AI_Studio():

    def __init__(self):
        self.Accesstoken = Accesstoken
        self.workflow_id = Workflow_id
        self.root_url = AI_STUDIO_URL

    def add_node(self, data):

        url = self.root_url + 'api-db/widget/'

        headers = {
            'Content-Type': 'application/json',  # Specify the content type as JSON
            'Accesstoken': self.Accesstoken
        }

        response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def delete_node(self, args):

        url = self.root_url + f'api-db/widget/{args["node_id"]}?workflow_id={self.workflow_id}'

        headers = {
            'Accesstoken': self.Accesstoken
        }

        response = requests.delete(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def update_node_params(self, data):

        url = self.root_url + 'api-db/widget/batch-config/'

        headers = {
            'Content-Type': 'application/json',  # Specify the content type as JSON
            'Accesstoken': self.Accesstoken
        }

        response = requests.post(url, data=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def add_edge(self, args):
        url = self.root_url + 'api-db/connection/'

        # data = {
        #   "src_node_id": args["input_widget_id"],
        #   "dst_node_id": args["output_widget_id"],
        #   "workflow_id": self.workflow_id
        # }

        data = {
            "workflow_id": self.workflow_id,
            "src_node_id": args["source_node_id"],
            "src_output_id": args["source_endpoint_id"],
            "dst_node_id": args["target_node_id"],
            "dst_input_id": args["target_endpoint_id"]
        }

        data = json.dumps(data)

        headers = {
            'Content-Type': 'application/json',  # Specify the content type as JSON
            'Accesstoken': self.Accesstoken
        }

        response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def delete_edge(self, args):
        url = self.root_url + f'api-db/connection/{args["edge_id"]}?workflow_id={self.workflow_id}'

        headers = {
            'Accesstoken': self.Accesstoken
        }

        response = requests.delete(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def update_pos_xy(self, args):
        url = self.root_url + f'api-db/widget/{args["node_id"]}?x={args["x"]}&y={args["y"]}&workflow_id={self.workflow_id}'

        headers = {
            'Accesstoken': self.Accesstoken
        }

        response = requests.put(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def get_workflow(self):
        url = self.root_url + f"api-db/workflow/{self.workflow_id}"
        headers = {
            'Accesstoken': self.Accesstoken
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            res = response.json()
            edges = []
            for edge in res["data"]["edges"]:
                e = {
                    "edge_id": edge["id"],
                    "source_node_id": edge["sourceNode"],
                    "source_endpoint_id": edge["source"],
                    "target_node_id": edge["targetNode"],
                    "target_endpoint_id": edge["target"]
                }
                edges.append(e)

            nodes = []
            for node in res["data"]["nodes"]:
                # n = {
                #     "node_id": node["id"],
                #     "node_name": node["name"],
                #     "widget_id": node["abstractWidgetId"],
                #     "widget_name": node["abstractName"],
                #     "node_params": node["widgetParam"]
                # }
                widget_id = node["abstractWidgetId"]
                widget_param = {}
                with open(new_widgets_path, 'r', encoding='utf-8') as  f:
                    new_widgets = json.load(f)
                for widget in new_widgets:
                    if widget["widget_id"] == widget_id:
                        if widget["widget_name"] == "Data Table":
                            widget_param = node["widgetParam"]
                        elif widget["widget_name"] == "SQL Table":
                            try:
                                target_column = None
                                attr_mapping = node["widgetParam"]["attr_mapping"]
                                for key in attr_mapping:
                                    if attr_mapping[key]["role"] == 1:
                                        target_column = key
                                widget_param = {
                                    "table_name": node["widgetParam"]["table_name"],
                                    "target_column": target_column
                                }
                            except:
                                widget_param = {}
                        elif widget["widget_name"] == "Edit Domain":
                            nodes_params = node["widgetParam"]
                            try:
                                for key, column in nodes_params["json_obj"].items():
                                    if column["valueMappings"] != []:
                                        widget_param = {
                                            "column_name": column["nameMapping"]["currentName"],
                                            "valueMappings": column["valueMappings"]
                                        }
                                        break

                            except:
                                widget_param = {}
                        else:
                            for param in widget["params"]:
                                if param["name"] in node["widgetParam"]:
                                    widget_param[param["name"]] = node["widgetParam"][param["name"]]
                                else:
                                    widget_param[param["name"]] = param["default"]
                new_endpoints = []
                for ep in node["endpoints"]:
                    new_endpoints.append({
                        "id": ep["id"],
                        "type": ep["type"],
                        "variable": ep["variable"],
                        "name": ep["name"],
                        "short_name": ep["short_name"],

                    })
                n = {
                    "node_id": node["id"],
                    "widget_name": node["abstractName"],
                    "node_name": node["name"],
                    "pos_x": node["left"],
                    "pos_y": node["top"],
                    "node_params": widget_param,
                    "endpoints": new_endpoints
                }
                try:
                    del n["node_params"]["widget_id"]
                except:
                    pass
                nodes.append(n)

            workflow_info = {
                "edges": edges,
                "nodes": nodes
            }

            return workflow_info
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def get_data_info(self, widget_id):
        # 可以获取组件 widget_id 的输入数据信息
        url = self.root_url + f'api-io/dataset/metadata?widget_id={widget_id}'

        headers = {
            'Accesstoken': self.Accesstoken
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            res = response.json()
            if res["data"] != None:
                columns_info = []
                for ad in res["data"]["attr_descs"]:
                    columns_info.append({
                        "key": ad["key"],
                        "name": ad["name"],
                        "data_type": res["data"]["idx2type"][str(ad["type"])],
                        "role": res["data"]["idx2role"][str(ad["role"])]
                    })
                data = {
                    "columns_info": columns_info,
                    "n_sample": res["data"]["n_sample"],
                    "n_feature": res["data"]["n_feature"],
                    "missing_values": res["data"]["missing_in_attr"]
                }
                return data
            return {}
        else:
            print('请求失败，状态码：', response.status_code)
            return {}

    def get_widget_output(self, widget_id):
        # 可以获取组件 widget_id 的输入数据信息
        url = self.root_url + f'api-db/widget/result/{widget_id}?workflow_id={self.workflow_id}'

        headers = {
            'Accesstoken': self.Accesstoken
        }
        requests.options(url, headers=headers)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            res = response.json()
            return res
        else:
            print('请求失败，状态码：', response.status_code)
            return None

    def run_widget(self, data):
        url = self.root_url + 'api-engine/engine/run-widget'

        headers = {
            'Content-Type': 'application/json',  # Specify the content type as JSON
            'Accesstoken': self.Accesstoken
        }
        requests.options(url, headers=headers)
        response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            res = response.json()
            return res
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def get_run_widget_status(self, data):
        url = self.root_url + 'api-engine/engine/widget-run-status'

        headers = {
            'Content-Type': 'application/json',  # Specify the content type as JSON
            'Accesstoken': self.Accesstoken
        }
        requests.options(url, headers=headers)
        response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            res = response.json()
            return res["data"]
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def run_workflow(self):
        url = self.root_url + 'api-engine/engine/run-workflow'
        headers = {
            'Accesstoken': self.Accesstoken
        }

        data = {
            "workflow_id": self.workflow_id,
        }
        requests.options(url, headers=headers)
        response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def stop_workflow(self):
        url = self.root_url + 'api-engine/engine/stop-running'
        headers = {
            'Accesstoken': self.Accesstoken
        }

        data = {
            "task_id": self.workflow_id,
            "task_type": "workflow"
        }

        requests.options(url, headers=headers)
        response = requests.post(url, data=data, headers=headers)
        print(f"工作流停止:{response}")

        if response.status_code == 200:
            res = response.json()
            return res
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def update_workflow(self):
        url = self.root_url + f"api-db/workflow/{self.workflow_id}"
        headers = {
            'Accesstoken': self.Accesstoken
        }

        response = requests.put(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def get_run_workflow_info(self):

        url = self.root_url + f"api-db/workflow/status?workflow_id={self.workflow_id}"
        headers = {
            'Accesstoken': self.Accesstoken
        }

        data = {
            "workflow_id": self.workflow_id,
        }
        requests.options(url, headers=headers)
        response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def hpo(self, args):
        url = self.root_url + f"api-io/hpo/nni"
        response = requests.post(url, data=args)
        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response)
            return None

    def get_model_score(self, args):
        url = self.root_url + f"api-io/hpo/default"
        response = requests.post(url, data=args)
        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response)
            return None

if __name__ == '__main__':
    # Test whether the platform can be accessed normally
    ai_studio = AI_Studio()
    workflow = ai_studio.get_workflow()
    print(workflow)