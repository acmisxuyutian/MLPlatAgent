from pathlib import Path
############### orange3 / uniplore ###############
PLATFORM = "orange3"

Accesstoken = ""
Workflow_id = 26
AI_STUDIO_URL = "http://172.19.92.160"

UNIPLORE_CONFIG = {
    "access_token": Accesstoken,
    "workflow_id": Workflow_id,
    "api_url": AI_STUDIO_URL,
}

ORANGE3_CONFIG = {
    "workflow_path": str(
        Path(__file__).resolve().parent
        / "ml_platform"
        / "orange3"
        / "Orange3-3.40.0"
        / "workspace"
        / "current.ows"
    ),
}

############### llm ###############
# Configure the accessible model URL and model name
Model_PATH = "http://127.0.0.1:8000/v1"
# qwen2_5-14b-coder / qwen2.5-72b
MODEL_NAME = "qwen2.5-72b"
API_KEY = "sk-"
RANDOM_SEED = 42

# Section 4.5 robustness experiment.  The online runner only manages the
# explicitly dedicated Uniplore workflow below.  Set ``rescore_only`` to True
# to rejudge saved Agent outputs without rerunning MLPlatAgent or contacting
# Uniplore; ``result_model`` may then select a previously generated result.
ROBUSTNESS_CONFIG = {
    "dedicated_workflow_id": 29,
    "rescore_only": False,
    "result_model": "",
}

# Provide a MySQL database that can be accessed via the cloud, enabling Uniplore to access your personal data.
MySQL_Config = {
    "server": "127.0.0.1",
    "port": "3306",
    "username": "root",
    "password": "",
    "database": ""
}

# PostgreSQL data source used by the Orange3 SQL Table adapter.
PG_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "",
    "schema": "public",
    "user": "postgres",
    "password": ""
}
