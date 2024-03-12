#!/usr/bin/python3
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

# 清理显存缓存
def torch_gc():
  if torch.cuda.is_available():
    with torch.cuda.device(CUDA_DEVICE):
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()


app = FastAPI()

# 让app可以跨域
origins = ["*"]
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# 绑定路由
@app.post("/")
async def create_item(request: Request):
  global model, tokenizer
  json_post_raw = await request.json()
  json_post = json.dumps(json_post_raw)
  json_post_list = json.loads(json_post)
  print(json_post_list)
  prompt = json_post_list.get('prompt')
  max_length = json_post_list.get('max_length')
  top_p = json_post_list.get('top_p')
  top_k = json_post_list.get('top_k')
  temperature = json_post_list.get('temperature')
  messages=[
      { 'role': 'user', 'content': F"""
  我会给你提供一份git commit的描述信息，请将信息归类为如下单词之一：“Bug 修复”，“性能优化”，“CVE修复”，“新增功能或删除功能”，“文档改进”，“测试代码改进”，或“其他”，只回答分类单词，不需要赘述：
  {prompt}"""}
  ]
  inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
  response = model.generate(inputs,
                            max_length=max_length if max_length else 2048,
                            top_k=top_k if top_k else 50,
                            top_p=top_p if top_p else 0.7,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=temperature if temperature else 0.9,
                            eos_token_id=tokenizer.eos_token_id)
  response = tokenizer.decode(response[0][len(inputs[0]):], skip_special_tokens=True)
  response.replace("<|EOT|>", "")
  now = datetime.datetime.now()
  time = now.strftime("%Y-%m-%d %H:%M:%S")
  answer = {
    "response": response,
    "status": 200,
    "time": time
  }
  log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
  print(log)
  torch_gc()
  return answer


if __name__ == '__main__':
  model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
  model.eval()
  uvicorn.run(app, host='0.0.0.0', port=10086, workers=1)