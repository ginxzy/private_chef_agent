import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def get_model():

    return init_chat_model(
        model="qwen3.8-flash",
        model_provider="openai",
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )