"""
Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.
"""

# EVOLVE_PROMPT = """你是一个基于遗传算法的automl，你的目标是优化模型{model_name}的超参数，你需要严格遵循模型超参数的介绍：
# ```
# {params_info}
# ```
# 你需要按照以下两个步骤开启遗传算法：
# 1.交叉以下两个超参数配置：
# hyper params 1: {parent1}, {metric}: {score1}
# hyper params 2: {parent2}, {metric}: {score2}
# 根据两个配置的模型评估结果，一步一步思考两个配置各自的优势，通过交叉操作结合两者的优势，生成一个能使模型评估结果更好的超参数配置。
# 2.变异步骤1生成的超参数配置：考虑步骤1中生成的超参数配置有什么缺陷，通过变异操作对其进行修改，生成最终的超参数配置，格式为：<hyperparam>变异后的超参数配置，确保内容能够被json.loads()解析，不要有多余的解释</hyperparam>。
# 现在，开始进行交叉和变异！
# """

EVOLVE_PROMPT = """你是一个基于遗传算法的automl，你的目标是优化模型{model_name}的超参数，你需要严格遵循模型超参数的介绍：
```
{params_info}
```
你需要按照以下两个步骤开启遗传算法：
1.交叉以下两个超参数配置：
params 1: {parent1}, {metric}: {score1}
params 2: {parent2}, {metric}: {score2}
根据两个配置的模型评估结果，一步一步思考两个配置各自的优势，通过交叉操作结合两者的优势，生成一个能使模型评估结果更好的超参数配置。
2.变异步骤1生成的超参数配置：考虑步骤1中生成的超参数配置有什么缺陷，通过变异操作对其进行修改，生成变异后的超参数配置。
你的输出应该严格遵循以下格式：
Cross Thought:交叉操作的思考
Cross Params:交叉后的超参数配置
Mutation Thought:变异操作的思考
Mutation Params:变异后的超参数配置应该是一个严格的JSON，且不要有多余的输出，记住不要废话，不要解释，给出配置即可。
注意：所有的思考都应该足够简短精炼，尽量用一句中文概括即可
现在，开始进行交叉和变异！
"""

INI_PROMPT = """你是一个automl，你的目标是优化模型{model_name}的超参数，你需要严格遵循模型超参数的介绍：
```
{params_info}
```
你的输出应该是一个严格的JSON，且不要有多余的输出。例如：{params}
现在，给出一个新的超参数配置确保模型性能得到改善，记住不要废话，不要解释，给出配置即可。
"""