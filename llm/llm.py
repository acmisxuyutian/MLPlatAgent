# -*- coding: utf-8 -*-
import json
from utils.logs import logger
import openai
from config import Model_PATH, API_KEY, MODEL_NAME
openai.api_key = API_KEY
openai.api_base = Model_PATH

class Qwen_Model:

    def predict(self, messages, temperature=0, top_p=1, max_tokens=4096, tools=None):

        logger.info("llm input:\n" + json.dumps(messages, ensure_ascii=False, indent=4))

        retry_time = 1
        max_time = 5

        for i in range(max_time):
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    max_tokens=max_tokens
                )
                message = response.choices[0].message
                # 处理为和GPT格式统一
                new_message = {
                    "role": message.role,
                    "content": message.content
                }
                if "tool_calls" in message:
                    if message.tool_calls != None and message.tool_calls != []:
                        new_message = message
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                price = self.costs(input_tokens, output_tokens)
                if new_message["content"] is not None and new_message["content"] != "":
                    logger.info("llm output:\n" + new_message["content"])
                logger.info(f"input_tokens: {input_tokens}, output_tokens: {output_tokens}, price: {price}")
                return new_message["content"], new_message, input_tokens, output_tokens, price
            except Exception as e:
                logger.info(f"llm调用失败，重试第{retry_time}次，错误信息：{e}")
                retry_time += 1
        raise Exception("llm调用失败")

    def costs(self, input_tokens, output_tokens):
        return 0

if __name__ == '__main__':
    import json
    messages = [
        {"role": "system",
         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
        {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
    ]
    model = Qwen_Model()
    content, message, input_tokens, output_tokens, price = model.predict(messages=messages)
