# -*- coding: utf-8 -*-
# If you wish to use your own dataset, please remember to configure the database;
# otherwise, you will only be able to use the sample dataset.
import json
from agents.mlagent import MLAgent


def main():
    mlagent = MLAgent()
    requirement = "你的任务是根据银行客户的各种信息，如账户活动、服务使用情况等，构建一个能够预测客户是否会流失的机器学习模型。"
    results = mlagent.run(requirement)
    print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
