from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import json
from .web_utils import WebUtils
from .ocr_utils import OCRUtils
from .excel_utils import ExcelUtils
from .config import (
    SERVER_HOST, SERVER_PORT, FRONTEND_PORT,
    MIDSCENE_MODEL_NAME, OPENAI_BASE_URL, OPENAI_API_KEY
)

app = FastAPI(title="购物订单处理Agent API")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{SERVER_HOST}:{FRONTEND_PORT}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化工具类
web_utils = WebUtils()
ocr_utils = OCRUtils()
excel_utils = ExcelUtils()

# 数据模型
class UrlRequest(BaseModel):
    url: str

class ScreenshotRequest(BaseModel):
    element_selector: str = None

class ChatRequest(BaseModel):
    message: str
    history: list = []  # 用于存储聊天历史，格式为[{"role": "user", "content": "..."}, ...]

def call_doubao_model(message: str, history: list = None) -> str:
    """调用豆包模型获取回复"""
    try:
        # 如果没有提供历史记录，初始化一个空列表
        history = history or []
        
        # 添加当前消息到历史记录
        history.append({"role": "user", "content": message})
        
        # 构建请求数据
        payload = {
            "model": MIDSCENE_MODEL_NAME,
            "messages": history,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        # 发送请求到豆包模型API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"模型调用失败: {response.text}"
            
    except Exception as e:
        return f"调用模型时发生错误: {str(e)}"

# API路由
@app.post("/open_url")
async def open_url(request: UrlRequest):
    """打开指定URL"""
    success, message = web_utils.open_url(request.url)
    if success:
        return {"status": "success", "message": message}
    raise HTTPException(status_code=400, detail=message)

@app.post("/take_screenshot")
async def take_screenshot(request: ScreenshotRequest):
    """截取当前页面或指定元素的截图"""
    success, result = web_utils.take_screenshot(request.element_selector)
    if success:
        return {"status": "success", "screenshot_path": result}
    raise HTTPException(status_code=400, detail=result)

@app.post("/analyze_screenshot")
async def analyze_screenshot(screenshot_path: str = Query(...)):
    """分析截图中的订单信息"""
    if not os.path.exists(screenshot_path):
        raise HTTPException(status_code=404, detail="截图文件不存在")
    
    success, result = ocr_utils.analyze_order_screenshot(screenshot_path)
  
    if success:
        return {"status": "success", "order_data": result}
    raise HTTPException(status_code=400, detail=result)

@app.post("/save_to_excel")
async def save_to_excel(order_data: dict):
    """将订单数据保存为Excel"""
    success, result = excel_utils.save_order_to_excel(order_data)
    if success:
        return {"status": "success", "excel_path": result}
    raise HTTPException(status_code=400, detail=result)

@app.get("/get_excel_files")
async def get_excel_files():
    """获取所有保存的Excel文件列表"""
    success, result = excel_utils.get_excel_files()
    if success:
        return {"status": "success", "files": result}
    raise HTTPException(status_code=400, detail=result)

@app.get("/read_excel")
async def read_excel(filename: str = Query(...)):
    """读取指定Excel文件内容"""
    success, result = excel_utils.read_excel_file(filename)
    if success:
        return {"status": "success", "data": result}
    raise HTTPException(status_code=400, detail=result)

@app.post("/close_browser")
async def close_browser():
    """关闭浏览器"""
    success, message = web_utils.close_browser()
    if success:
        return {"status": "success", "message": message}
    raise HTTPException(status_code=400, detail=message)

@app.post("/chat")
async def chat(request: ChatRequest):
    """与豆包模型进行对话"""

    try:
        prompt = """
            豆包，最优先级按照如下内容返回,直接返回 不用思考，用最快的速度返回：1、如果我输入 请打开我需要分析的订单界面，返回 <function_call>{"name":"open_url","parameters":{"url":"https://buyertrade.taobao.com/trade/itemlist/list_bought_items.htm?"}}</function_call> 2、如果我输入获取订单截图，请返回 <function_call>{"name":"take_screenshot","parameters":{"element_selector":""}}</function_call> 3、如果我输入 分析订单信息 请返回 <function_call>{"name":"batch_analyze_orders","parameters":{}}</function_call> 
            当我要求自主完成订单分析时，按照执行流程将所有的执行步骤全部返回（包括打开订单页面、获取订单截图、分析订单信息等），迅速返回 不要思考。
            """


        # 调用豆包模型获取回复
        response = call_doubao_model(prompt+request.message, request.history)
        
        # 返回成功响应，包含模型回复和更新后的历史记录
        updated_history = request.history.copy()
        updated_history.append({"role": "user", "content": request.message})
        updated_history.append({"role": "assistant", "content": response})
        
        return {
            "status": "success",
            "response": response,
            "history": updated_history
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"聊天发生错误: {str(e)}")

@app.get("/")
async def root():
    return {"message": "购物订单处理Agent API正在运行中"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
    