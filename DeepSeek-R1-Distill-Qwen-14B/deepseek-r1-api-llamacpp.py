#!/usr/bin/python3
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import uvicorn, json, datetime
import gc
import os, re

here_dir = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# 让 app 可以跨域
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Llama 模型
def init_model():
    llm = Llama.from_pretrained(
        repo_id="lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        filename="DeepSeek-Coder-V2-Lite-Instruct-Q6_K.gguf",
        n_ctx=40960,
        cache_dir=here_dir,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repeat_penalty=1.0,
        seed=42,
        logits_all=True,
        verbose=False        
    )
    return llm

model = init_model()

# 绑定路由
@app.post("/")
async def create_item(request: Request):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    try:
        json_post_raw = await request.json()
        prompt = json_post_raw.get('prompt')
        max_length = json_post_raw.get('max_input', 81920)
        max_tokens = json_post_raw.get('max_tokens', 512)
        top_p = json_post_raw.get('top_p', 1.0)
        top_k = json_post_raw.get('top_k', 1)
        temperature = json_post_raw.get('temperature', 0.0)
        

        # 直接使用当前提示进行推理，不包含历史信息
        output = model.create_chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature
        )
        response = output['choices'][0]['message']['content']
        answer = {
            "response": response,
            "status": 200,
            "time": time,
        }
        # 记录审计日志
        log = f"[{time}] \"prompt\":\"{prompt}\", \"response\":\"{repr(response)}\""
        print(log)
        return answer
    except Exception as e:
        answer = {
            "response": f"推理过程出现错误: {str(e)}",
            "status": 500,
            "time": time
        }
        # 记录详细错误日志
        with open("error_log.txt", "a") as f:
            f.write(f"[{time}] {str(e)}\n")
        return answer
    finally:
        # 清理内存
        gc.collect()

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=10086, workers=1)