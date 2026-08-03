# Orange3 组件名称到 Uniplore 组件名称的对应关系。只收录 ml_platform/orange3/widgets.json 中已经开放的组件。名称相同的组件
# 直接对应；名称不同但功能等价的组件使用 Uniplore 中的真实组件名称。
ORANGE3_UNIPLORE_WIDGET_MAP = {
    "File": "File",
    "SQL Table": "SQL Table",
    "Rank": "Rank",
    "Edit Domain": "Edit Domain",
    "Save Data": "Save Data",
    "Data Sampler": "Data Sampler",
    "Select Columns": "Select Columns",
    "Unique": "Unique",
    "Impute": "Impute",
    "Continuize": "Continuize",
    "Formula": "Feature Constructor",
    "Scatter Plot": "Scatter Plot",
    "Line Plot": "Line Chart",
    "kNN": "KNN",
    "Tree": "Tree",
    "Random Forest": "Random Forest",
    "Gradient Boosting": "Gradient Boosting Decision Tree",
    "Linear Regression": "Linear Regression",
    "Logistic Regression": "Logistic Regression",
    "Naive Bayes": "Naive Bayes",
    "Neural Network": "Neural Network",
    "Test and Score": "Test Score",
    "Predictions": "Predictions",
    "Confusion Matrix": "Confusion Matrix",
    "Correlations": "Correlogram",
}

# 反向映射用于根据 Uniplore 组件名称查找对应的 Orange3 组件。
UNIPLORE_ORANGE3_WIDGET_MAP = {
    uniplore_widget: orange3_widget
    for orange3_widget, uniplore_widget in ORANGE3_UNIPLORE_WIDGET_MAP.items()
}

# 平台组件到 benchmark 算法步骤的对应关系

ALGORITHM_STEP_ORANGE3_MAP = {
    "Confusion Matrix": ["Confusion Matrix"],
    "Correlogram": ["Correlations"],
    "Dataset Division": ["Data Sampler"],
    "Decision Tree Model": ["Tree"],
    "Edit Domain": ["Edit Domain"],
    "Feature Construction": ["Formula"],
    "Feature Selection": ["Rank"],
    "Gradient Boosting Decision Tree": ["Gradient Boosting"],
    "Image Dataset Loading": [],
    "KNN": ["kNN"],
    "LightGBM": [],
    "Line Chart": ["Line Plot"],
    "Linear Regression": ["Linear Regression"],
    "Logistic Regression": ["Logistic Regression"],
    "Missing Value Handling": ["Impute"],
    "Model Evaluate": ["Test and Score"],
    "Naive Bayes": ["Naive Bayes"],
    "Neural Network": ["Neural Network"],
    "Object Detection Model": [],
    "One Hot Encoding": ["Continuize"],
    "Random Forest": ["Random Forest"],
    "Rank": ["Rank"],
    "Save Data": ["Save Data"],
    "Scatter Plot": ["Scatter Plot"],
    "Select Columns": ["Select Columns"],
    "Tabular Dataset Loading": ["File", "SQL Table"],
    "Test Dataset Predictions": ["Predictions"],
    "Train Log": [],
    "Unique": ["Unique"],
    "XGBoost": [],
}

ORANGE3_ALGORITHM_STEP_MAP = {
    widget: algorithm_step
    for algorithm_step, widgets in ALGORITHM_STEP_ORANGE3_MAP.items()
    for widget in widgets
}

ALGORITHM_STEP_UNIPLORE_MAP = {
    'Tabular Dataset Loading': ['File', 'SQL Table'],
    'Text Dataset Loading': ['Text'],
    'Image Dataset Loading': ['Image'],
    'Missing Value Handling': ['Impute'],
    'Dataset Division': ['Data Sampler'],
    'One Hot Encoding': ['One Hot Encoder'],
    'Feature Construction': ['Feature Constructor'],
    'Feature Selection': ['Select Best N Attributes'],
    'Image Segmentation Model': ['Segmentation'],
    'Image Classification Model': ['Image Classification'],
    'Object Detection Model': ['Object Detection'],
    'Text Classification Model': ['Text Classification'],
    'Text Translation Model': ['Translation'],
    'Test Dataset Predictions': ['Predictions', 'Infer'],
    'Model Evaluate': ['Test Score', 'Test & Score'],
    'Decision Tree Model': ['Tree']
}

UNIPLORE_ALGORITHM_STEP_MAP = {widget: step for step, widgets in ALGORITHM_STEP_UNIPLORE_MAP.items() for widget in widgets}
