# -*- coding: utf-8 -*-
import json

import requests
import os
from utils.utils import get_project_root
new_widgets_path = os.path.join(get_project_root(), r"ml_platform/uniplore/widgets.json")

class AI_Studio():

    def __init__(self, workflow_id, accesstoken, root_url):
        self.Accesstoken = accesstoken
        self.workflow_id = workflow_id
        self.root_url = root_url.rstrip("/")
        # self.root_url = "http://idis.uniplore.cn:30350/"
        # self.root_url = "http://192.168.2.170"

    def add_node(self, data):

        # url = self.root_url + 'api-db/widget/'
        url = self.root_url + ':5000/widget/'

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

        # url = self.root_url + f'api-db/widget/{args["node_id"]}?workflow_id={self.workflow_id}'
        url = self.root_url + f':5000/widget/{args["node_id"]}?workflow_id={self.workflow_id}'

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

        # url = self.root_url + 'api-db/widget/batch-config/'
        url = self.root_url + ':5000/widget/batch-config/'

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
        # url = self.root_url + 'api-db/connection/'
        url = self.root_url + ':5000/connection/'

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
        # url = self.root_url + f'api-db/connection/{args["edge_id"]}?workflow_id={self.workflow_id}'
        url = self.root_url + f':5000/connection/{args["edge_id"]}?workflow_id={self.workflow_id}'

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
        url = self.root_url + f':5000/widget/{args["node_id"]}?x={args["x"]}&y={args["y"]}&workflow_id={self.workflow_id}'

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
        url = self.root_url + f":5000/workflow/{self.workflow_id}"
        headers = {
            'Accesstoken': self.Accesstoken
        }
        requests.options(url, headers=headers)
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
        url = self.root_url + f':5002/dataset/metadata?widget_id={widget_id}'

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
        url = self.root_url + f':5000/widget/result/{widget_id}?workflow_id={self.workflow_id}'

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
        url = self.root_url + ':5001/engine/run-widget'

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

    def get_run_widget_status(self,data):
        url = self.root_url + ':5001/engine/widget-run-status'

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
        url = self.root_url + ':5001/engine/run-workflow'
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
        url = self.root_url + ':5001/engine/stop-running'
        headers = {
            'Accesstoken': self.Accesstoken
        }

        data = {
            "task_id": self.workflow_id,
            "task_type": "workflow"
        }

        requests.options(url, headers=headers)
        response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            res = response.json()
            return res
        else:
            print('请求失败，状态码：', response.status_code)
            return response.status_code

    def update_workflow(self):
        url = self.root_url + f":5000//workflow/{self.workflow_id}"
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

        url = self.root_url + f":5000/workflow/status?workflow_id={self.workflow_id}"
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
        url = self.root_url + f":5007/hpo/nni"
        response = requests.post(url, data=args)
        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response)
            return None
    def get_model_score(self, args):
        url = self.root_url + f":5007/hpo/default"
        response = requests.post(url, data=args)
        if response.status_code == 200:
            return response.json()
        else:
            print('请求失败，状态码：', response)
            return None




if __name__ == '__main__':

    from config import UNIPLORE_CONFIG
    # data_info_path = os.path.join(get_project_root(), "data/data_info.json")
    # with open(data_info_path, 'r', encoding='utf-8') as f:
    #     data_info = json.load(f)
    ai_studio = AI_Studio(
        UNIPLORE_CONFIG["workflow_id"],
        UNIPLORE_CONFIG["access_token"],
        UNIPLORE_CONFIG["api_url"],
    )
    # # ai_studio.get_model_score({"widget_name": "Random Forest", "metric": "auc_roc"})
    # res = ai_studio.hpo({"widget_name": "Random Forest", "metric": "auc_roc"})
    # print(json.dumps(res, indent=4, ensure_ascii=False))
    # print(res)

    # data = {
    #     "widget_id": "1124",
    #     "workflow_id": ai_studio.workflow_id
    # }
    # data_str = json.dumps(data, ensure_ascii=False)
    # print(ai_studio.get_run_widget_status(data_str))
    # 获取工作流信息 Test
    workflow = ai_studio.get_workflow()

    # mapp = {
    #     "PassengerId": {
    #         "key": 0,
    #         "name": "PassengerId",
    #         "type": 2,
    #         "role": -1
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
    #         "type": 1,
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
    #         "type": 1,
    #         "role": 0
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
    #         "type": 1,
    #         "role": 0
    #     },
    #     "Embarked": {
    #         "key": 11,
    #         "name": "Embarked",
    #         "type": 1,
    #         "role": 0
    #     }
    # }
    # from config import MySQL_Config
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
    #     "table_name": "titanic_test",
    #     "save2db": False,
    #     "sql_code": "",
    #     "dump_table": "my_table",
    #     "selectedDataset": "",
    #     "dataset_name": None,
    #     "dataset_sql": None,
    #     "field": None,
    #     "interact_type": 0,
    #     "attr_mapping": mapp
    # }
    #
    # data = {
    #     "workflow_id": ai_studio.workflow_id,
    #     "param_info": {"2110": widget_params}
    # }
    #
    # data = json.dumps(data, ensure_ascii=False)
    #
    # result = ai_studio.update_node_params(data)
    # print(result)

    # # ai_studio.run_workflow()
    # tag = 0
    # while True:
    #     time.sleep(3)
    #     res = ai_studio.get_run_workflow_info()
    #     data = res["data"]
    #     if data[0]["status"] == 3:
    #         print("工作流运行成功！")
    #         break
    #     elif data[0]["status"] == -1:
    #         print("工作流运行出错")
    #         print(res)
    #     elif data[0]["status"] == 0:
    #         print(data[0]["status"])
    #         if tag == 0:
    #             tag += 1
    #             print(ai_studio.stop_workflow())
    #         elif tag == 1:
    #             tag += 1
    #             print(ai_studio.run_workflow())
    #     else:
    #         print("工作流运行中...")
    # 更改节点位置
    # data = {
    #     "node_id": 1507,
    #     "x": 100,
    #     "y": 100
    # }

    # print(ai_studio.update_pos_xy(data))

    # # 获取指定节点的数据集信息
    # data = ai_studio.get_data_info(1339)
    # print(json.dumps(data, ensure_ascii=False, indent=4))

    # 运行节点
    # data = {
    #     "widget_id": "1253",
    #     "workflow_id": 112,
    #     "interact_type": 1
    # }
    # ai_studio.run_widget(json.dumps(data,ensure_ascii=False))
    # data = {
    #     "widget_id": "1253",
    #     "workflow_id": 112
    # }
    # while True:
    #     run_info = ai_studio.get_run_widget_status(json.dumps(data,ensure_ascii=False))
    #     if run_info["status"] == 3:
    #         print("运行成功")
    #         print(json.dumps(run_info["data"], ensure_ascii=False, indent=4))
    #         break
    #     elif run_info["status"] == -1:
    #         print("运行失败")
    #         break
    #     else:
    #         print("运行中...")
    #         print(json.dumps(run_info, ensure_ascii=False, indent=4))
    #         time.sleep(3)
    #
    # 节点参数更新

    # node_id = "1256"
    #
    # node_param = {'expressions': [{'key': 0, 'name': 'FamilySize', 'expression': 'SibSp + Parch + 1', 'type': '2', 'image': 'N', 'categories': 'A, B'}]}
    #
    # param_info = {
    #     node_id: node_param
    # }
    #
    # data = {
    #     "workflow_id": ai_studio.workflow_id,
    #     "param_info": param_info,
    # }
    #
    # data = json.dumps(data, ensure_ascii=False)
    #
    # print(ai_studio.update_node_params(data))
    #
    # print(json.dumps(ai_studio.get_workflow(), ensure_ascii=False))

    # # 添加边 Test
    # data = {
    #     "source_node_id": 993,
    #     "source_endpoint_id": 1234,
    #     "target_node_id": 999,
    #     "target_endpoint_id": 1257
    # }
    #
    # print(ai_studio.add_edge(data))

    # # 添加控件
    # widgetParam = {
    #     "type": "sample",
    #     "filename": "car.csv",
    #     "mapping": {
    #         "0": {  # 属性编号
    #             "role": 1,  # 属性类别编号,0:特征属性,1:目标属性,2:描述属性,-1:忽略该属性
    #             "type": 1  # 属性类型编号,1:离散属性,2:数值属性,3:文本属性,4:日期属性
    #         }
    #     }
    # }
    # widgetParam =  json.dumps(widgetParam)
    #
    # data = {"pos_x":378,"pos_y":183,"abstractWidgetId":13,"abstractName":"File","name":"File","image":"File.svg","package":"io","widgetParam":widgetParam,"workflow_id":workflow_id}
    # print(json.dumps(data,ensure_ascii=False))
    # print(ai_studio.add_node(json.dumps(data,ensure_ascii=False)))



#     """
# 运行状态说明
# FAILED = -1 运行出错
# NOSTATE = 0 无状态
# PENDING = 1 等待运行
# RUNNING = 2 运行中
# SUCCEED = 3 运行成功
#     """
#     print(ai_studio.run_workflow())
#
#     import  time
#     while (True):
#         time.sleep(5)
#         res = ai_studio.get_run_workflow_info()
#         print(res)
#         data = res["data"]
#         if data[0]["status"] == 3:
#             break
#         elif data[0]["status"] == -1:
#             print("运行出错")
#             break
#         else:
#             print("运行中")
#
#     for  d in data:
#         print("组件名称：" + d["name"])
#         print("工作流运行状态：" + str(d["workflow"]["status"]))
#         print("组件运行状态：" + str(d["status"]))
#         if d["status"] == -1:
#             print("组件运行出错信息：" + json.dumps(d,indent=4,ensure_ascii=False))
