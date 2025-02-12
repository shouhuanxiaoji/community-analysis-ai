#!/usr/bin/python3
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import uvicorn, json, datetime
import torch
import gc

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
      { 'role': 'user', 'content': f"{prompt}"}
  ]

  response = tokenizer.decode(response[0][len(inputs[0]):], skip_special_tokens=True)
  now = datetime.datetime.now()
  time = now.strftime("%Y-%m-%d %H:%M:%S")
  try:
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    response = model.generate(inputs,
                              max_length=max_length if max_length else 8192,
                              top_k=top_k if top_k else 1,
                              top_p=top_p if top_p else 0.01,
                              num_return_sequences=1,
                              do_sample=True,
                              temperature=temperature if temperature else 0.1,
                              eos_token_id=tokenizer.eos_token_id)

    answer = {
      "response": response,
      "status": 200,
      "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return json.dumps(answer, indent=2)
  except Exception as e:
    answer = {
      "response": f"推理过程出现错误: {e}",
      "status": 500,
      "time": time
    }
    return answer
  finally:
    # 清理输入的tokens
    if 'inputs' in locals():
        del inputs
    if 'response' in locals():
        del response
    # 清理显存
    torch_gc()
    # 清理内存
    gc.collect()
    # 重置模型的缓存（对于一些支持缓存的模型）
    if hasattr(model, 'reset_cache'):
        model.reset_cache()

if __name__ == '__main__':
  torch.set_num_threads(1)
  model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
  # 虽然有量化参数，但默认关闭
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )
  model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,quantization_config=None, device_map="auto")
  model.eval()
  uvicorn.run(app, host='0.0.0.0', port=10086, workers=1)