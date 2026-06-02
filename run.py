# -*- coding: utf-8 -*-
# If you wish to use your own dataset, please remember to configure the database;
# otherwise, you will only be able to use the sample dataset.
import json
from agents.mlagent import MLAgent
mlagent = MLAgent()
requirement = "Load the Iris example dataset and build a model."
results = mlagent.run(requirement)
print(json.dumps(results, indent=4, ensure_ascii=False))