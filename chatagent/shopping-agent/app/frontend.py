import streamlit as st
import requests
import pandas as pd
import os
import json
from PIL import Image
from dotenv import load_dotenv
import uuid
import re
import time


# 加载环境变量
load_dotenv()

from config import MIDSCENE_MODEL_NAME, SCREENSHOTS_DIR, EXCELS_DIR, OPENAI_BASE_URL,OPENAI_API_KEY,CHROME_DRIVER_PATH,CHROME_PATH

# 加载环境变量
load_dotenv()

# 服务器配置
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8501))

# 配置文件路径
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.py")  # 修正为app/config.py路径


# 设置页面配置
st.set_page_config(
    page_title="购物订单处理Agent",
    page_icon="🛒",
    layout="wide"
)

# 自定义样式CSS - 包含按钮高度优化和滚动相关样式
st.markdown("""
<style>
    .chat-container {
        position: relative;
        width: 100%;
        margin-top: 5px;
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    .chat-input-area {
        display: flex;
        gap: 8px;
        align-items: center;
        width: 100%;
        box-sizing: border-box;
        padding: 10px;
        border-top: 1px solid #e0e0e0;
        background-color: #ffffff;
        border-radius: 0 0 8px 8px;
    }
    .chat-input {
        flex: 1;
        border-radius: 20px;
        padding: 10px 15px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        height: 40px;
        box-sizing: border-box;
        background-color: rgba(255, 255, 255, 0.95);
    }
    .typing-indicator {
        color: #666;
        font-style: italic;
        font-size: 12px;
        margin-top: 5px;
        margin-left: 10px;
    }
    .dataframe-container {
        max-height: 500px;
        overflow-y: auto;
    }
    .left-panel {
        display: flex;
        flex-direction: column;
        height: 100%;
        flex: 1;
        padding-right: 10px;
    }
    .stApp {
        overflow: hidden;
    }
    /* 修改列容器样式 - 关键修改点 */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
        /* 移除所有列的固定高度和默认滚动 */
        padding-top: 0 !important;  /* 优化按钮区域高度 */
        padding-bottom: 0 !important; /* 优化按钮区域高度 */
    }
    /* 仅保留左侧聊天区域的滚动效果 */
    [data-testid="column"]:first-child {
        height: calc(100vh - 120px);
        overflow-y: auto;
    }
    /* 明确禁用中间和右侧列的滚动 */
    [data-testid="column"]:nth-child(2),
    [data-testid="column"]:nth-child(3) {
        overflow-y: visible;
        height: auto;
    }
    .stSubheader {
        margin-bottom: 0.5rem !important;
    }
    .batch-results {
        margin-top: 20px;
        padding: 15px;
        background-color: #f9f9f9;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .order-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #fff;
    }
    .function-call {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9em;
        margin: 10px 0;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .function-result {
        background-color: #f0fdf4;
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
        word-wrap: break-word;
    }
    .function-call-sequence {
        margin-left: 15px;
        padding-left: 10px;
        border-left: 2px solid #93c5fd;
        margin-bottom: 15px;
    }
    .selection-container {
        position: relative;
        width: 100%;
        margin: 10px 0;
    }
    .selection-image {
        width: 100%;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .selection-box {
        position: absolute;
        border: 2px solid #4CAF50;
        background-color: rgba(76, 175, 80, 0.2);
        display: none;
    }
    .selection-buttons {
        margin-top: 10px;
        display: flex;
        gap: 10px;
    }
    .clear-chat-btn {
        margin-bottom: 10px;
        padding: 4px 8px;
        font-size: 0.8em;
        height: auto;
    }
    .chat-message {
        word-wrap: break-word;
        white-space: normal;
    }
    .step-indicator {
        font-size: 0.9em;
        color: #666;
        margin: 5px 0;
    }
    .config-form {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .config-input {
        margin-bottom: 15px;
    }
    /* 按钮区域高度优化 */
    [data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important; /* 减少列之间的间距 */
        margin-bottom: 0.5rem !important; /* 减少整个块的底部间距 */
    }
    
    /* 调整按钮容器的边距 */
    .element-container:has(.stButton) {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* 调整按钮本身的尺寸 */
    .stButton > button {
        padding: 0.25rem 0.5rem !important; /* 减少按钮内边距 */
        height: 2.25rem !important; /* 降低按钮高度 */
        min-height: auto !important; /* 移除最小高度限制 */
    }
    
    /* 调整按钮内文字的大小和边距 */
    .stButton > button [data-testid="stMarkdownContainer"] p {
        margin: 0 !important; /* 移除段落默认边距 */
        font-size: 0.875rem !important; /* 适当减小字体大小 */
    }
</style>
""", unsafe_allow_html=True)

# 引入Font Awesome图标库
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)

# 添加自定义JavaScript处理区域选择和输入框清空
st.markdown("""
<script>
// 全局变量存储选择区域
let selection = {
    startX: 0,
    startY: 0,
    endX: 0,
    endY: 0,
    isSelecting: false
};

// 初始化区域选择功能
function initSelection() {
    const container = document.getElementById('selection-container');
    const image = document.getElementById('selection-image');
    const selectionBox = document.getElementById('selection-box');
    
    if (!container || !image || !selectionBox) return;
    
    // 点击开始选择
    image.addEventListener('mousedown', function(e) {
        const rect = image.getBoundingClientRect();
        selection.startX = e.clientX - rect.left;
        selection.startY = e.clientY - rect.top;
        selection.isSelecting = true;
        
        // 显示选择框
        selectionBox.style.display = 'block';
        selectionBox.style.left = selection.startX + 'px';
        selectionBox.style.top = selection.startY + 'px';
        selectionBox.style.width = '0px';
        selectionBox.style.height = '0px';
    });
    
    // 鼠标移动调整选择区域
    document.addEventListener('mousemove', function(e) {
        if (!selection.isSelecting) return;
        
        const rect = image.getBoundingClientRect();
        selection.endX = e.clientX - rect.left;
        selection.endY = e.clientY - rect.top;
        
        // 计算选择框位置和大小
        const left = Math.min(selection.startX, selection.endX);
        const top = Math.min(selection.startY, selection.endY);
        const width = Math.abs(selection.endX - selection.startX);
        const height = Math.abs(selection.endY - selection.startY);
        
        // 应用到选择框
        selectionBox.style.left = left + 'px';
        selectionBox.style.top = top + 'px';
        selectionBox.style.width = width + 'px';
        selectionBox.style.height = height + 'px';
    });
    
    // 结束选择
    document.addEventListener('mouseup', function() {
        if (selection.isSelecting) {
            selection.isSelecting = false;
            
            // 计算相对图片的比例
            const imageWidth = image.offsetWidth;
            const imageHeight = image.offsetHeight;
            
            // 确保选择区域有效
            if (Math.abs(selection.endX - selection.startX) > 10 && 
                Math.abs(selection.endY - selection.startY) > 10) {
                
                // 计算相对原图的百分比
                const left = Math.min(selection.startX, selection.endX) / imageWidth * 100;
                const top = Math.min(selection.startY, selection.endY) / imageHeight * 100;
                const width = Math.abs(selection.endX - selection.startX) / imageWidth * 100;
                const height = Math.abs(selection.endY - selection.startY) / imageHeight * 100;
                
                // 将选择区域保存到隐藏输入框
                document.getElementById('selection-coords').value = 
                    `${left},${top},${width},${height}`;
                
                // 显示选择信息
                document.getElementById('selection-info').textContent = 
                    `已选择区域: 左上角(${Math.round(left)}%, ${Math.round(top)}%), 大小(${Math.round(width)}%, ${Math.round(height)}%)`;
            } else {
                // 选择区域太小，重置
                selectionBox.style.display = 'none';
                document.getElementById('selection-coords').value = '';
                document.getElementById('selection-info').textContent = '未选择区域';
            }
        }
    });
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 处理回车键发送消息
    setTimeout(function() {
        const chatInput = document.querySelector('input[placeholder="输入消息..."]');
        
        if (chatInput) {
            chatInput.setAttribute('autocomplete', 'off');
            chatInput.setAttribute('autocapitalize', 'off');
            chatInput.setAttribute('spellcheck', 'false');
            
            let isSubmitting = false;
            
            chatInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && this.value.trim() !== '' && !isSubmitting) {
                    e.preventDefault();
                    isSubmitting = true;
                    
                    const event = new Event('input', { bubbles: true });
                    this.dispatchEvent(event);
                    
                    const form = this.closest('form');
                    if (form) {
                        form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                    }
                    
                    // 清空输入框
                    this.value = '';
                    
                    setTimeout(() => {
                        isSubmitting = false;
                    }, 1000);
                }
            });
        }
    }, 100);
    
    // 初始化区域选择
    initSelection();
});
</script>
""", unsafe_allow_html=True)

# API基础URL
API_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# 页面标题
st.title("🛒 订单智能分析Agent")

# 侧边栏导航
st.sidebar.title("功能导航")
selected_tab = st.sidebar.radio(
    "选择功能",
    ["智能识别订单", "系统设置"]
)

# 初始化会话状态变量
if "is_thinking" not in st.session_state:
    st.session_state["is_thinking"] = False
if "screenshots" not in st.session_state:
    st.session_state["screenshots"] = []  # 存储多次截图信息，格式: [{id, path, timestamp}, ...]
if "batch_analysis_results" not in st.session_state:
    st.session_state["batch_analysis_results"] = []  # 存储所有识别的订单信息
if "order_data" not in st.session_state:
    st.session_state["order_data"] = None
if "current_order_data" not in st.session_state:
    st.session_state["current_order_data"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "submit_triggered" not in st.session_state:
    st.session_state["submit_triggered"] = False
if "last_submitted_message" not in st.session_state:
    st.session_state["last_submitted_message"] = ""
if "full_page_screenshot" not in st.session_state:
    st.session_state["full_page_screenshot"] = None  # 存储全页截图用于区域选择
if "selection_coords" not in st.session_state:
    st.session_state["selection_coords"] = ""  # 存储选择的区域坐标
if "function_results" not in st.session_state:
    st.session_state["function_results"] = []  # 存储函数调用结果
if "show_analysis_results" not in st.session_state:
    st.session_state["show_analysis_results"] = False  # 控制是否显示分析结果
if "function_call_sequence" not in st.session_state:
    st.session_state["function_call_sequence"] = []  # 存储待执行的函数调用序列
if "current_function_index" not in st.session_state:
    st.session_state["current_function_index"] = 0  # 当前执行的函数索引
# 新增：用于UI联动的状态变量
if "url_input" not in st.session_state:
    st.session_state["url_input"] = ""  # URL输入框内容
if "element_selector_input" not in st.session_state:
    st.session_state["element_selector_input"] = ""  # 元素选择器输入框内容
if "trigger_open_url" not in st.session_state:
    st.session_state["trigger_open_url"] = False  # 触发打开URL
if "trigger_take_screenshot" not in st.session_state:
    st.session_state["trigger_take_screenshot"] = False  # 触发截图
if "trigger_batch_analysis" not in st.session_state:
    st.session_state["trigger_batch_analysis"] = False  # 触发批量分析
# 配置修改相关状态
if "show_config_success" not in st.session_state:
    st.session_state["show_config_success"] = False  # 显示配置保存成功提示

# 辅助函数: 生成唯一ID
def generate_unique_id():
    return str(uuid.uuid4())[:8]

# 辅助函数: 移除截图
def remove_screenshot(screenshot_id):
    st.session_state["screenshots"] = [
        s for s in st.session_state["screenshots"] 
        if s["id"] != screenshot_id
    ]
    # 如果删除的截图已分析，同时从结果中移除
    st.session_state["batch_analysis_results"] = [
        r for r in st.session_state["batch_analysis_results"] 
        if r.get("screenshot_id") != screenshot_id
    ]

# 辅助函数: 清除聊天记录
def clear_chat_history():
    st.session_state["chat_history"] = []
    st.session_state["function_results"] = []
    st.session_state["function_call_sequence"] = []
    st.session_state["current_function_index"] = 0
    st.success("聊天记录已清除")

# 辅助函数: 保存配置到config.py文件
def save_config_to_py(configs):
    """保存配置到config.py文件"""
    try:
        # 确保config.py文件存在
        if not os.path.exists(CONFIG_FILE_PATH):
            st.error(f"配置文件 {CONFIG_FILE_PATH} 不存在")
            return False, f"配置文件 {CONFIG_FILE_PATH} 不存在"
        
        # 读取当前配置文件内容
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 更新配置值
        for i, line in enumerate(lines):
            # 跳过空行和注释行
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('#'):
                continue
                
            # 查找变量定义行
            for key, value in configs.items():
                if stripped_line.startswith(f"{key} ="):
                    # 处理字符串类型的值，添加引号
                    if isinstance(value, str) and not (value.startswith('"') and value.endswith('"')) and not (value.startswith("'") and value.endswith("'")):
                        value = f"'{value}'"
                    
                    # 保留注释
                    comment_index = line.find('#')
                    if comment_index != -1:
                        comment = line[comment_index:]
                        lines[i] = f"{key} = {value} {comment}\n"
                    else:
                        lines[i] = f"{key} = {value}\n"
        
        # 写回配置文件
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return True, "配置保存成功"
    except Exception as e:
        return False, f"保存失败: {str(e)}"

# 辅助函数: 显示订单分析结果
def display_analysis_results():
    """展示所有订单分析结果"""
    if not st.session_state["batch_analysis_results"]:
        return "没有可展示的订单分析结果"
    
    result_html = "<div class='batch-results'>"
    result_html += "<h3>所有订单分析结果</h3>"
    
    # 显示每个订单的完整信息
    for i, order in enumerate(st.session_state["batch_analysis_results"], 1):
        # 确保order是字典
        if not isinstance(order, dict):
            result_html += f"<p>订单 #{i} 数据格式不正确，跳过显示</p>"
            continue
            
        result_html += "<div class='order-card'>"
        result_html += f"<h4>订单 #{i}</h4>"
        
        # 订单基本信息
        result_html += "<div style='display: flex; gap: 10px; margin-bottom: 10px;'>"
        result_html += "<div style='flex: 1;'>"
        result_html += f"<p><strong>订单号:</strong> {order.get('订单号', 'N/A')}</p>"
        result_html += f"<p><strong>订单日期:</strong> {order.get('订单日期', 'N/A')}</p>"
        result_html += "</div>"
        result_html += "<div style='flex: 1;'>"
        result_html += f"<p><strong>截图时间:</strong> {order.get('screenshot_timestamp', 'N/A')}</p>"
        result_html += f"<p><strong>总金额:</strong> {order.get('总金额', 'N/A')}</p>"
        result_html += "</div>"
        result_html += "</div>"
        
        # 商品列表
        result_html += "<p><strong>商品列表:</strong></p>"
        products = order.get('商品名称列表', [])
        
        if not isinstance(products, list) or not products:
            result_html += "<p>未识别到有效的商品列表</p>"
        else:
            result_html += "<ul>"
            for product in products:
                if isinstance(product, dict):
                    product_name = product.get("名称", "未知名称")
                    product_quantity = product.get("数量", "N/A")
                    product_price = product.get("单价", "N/A")
                    result_html += f"<li>{product_name} - 数量: {product_quantity}, 单价: {product_price}</li>"
                else:
                    result_html += f"<li>{str(product)}</li>"
            result_html += "</ul>"
        
        result_html += "</div>"  # 关闭订单卡片
    
    result_html += "</div>"  # 关闭batch-results
    return result_html

# 函数调用处理相关函数
def extract_function_calls(response_text):
    """从响应文本中提取多个函数调用"""
    # 查找函数调用标记
    start_tag = "<function_call>"
    end_tag = "</function_call>"
    
    function_calls = []
    start_idx = 0
    
    # 循环提取所有函数调用
    while True:
        start_pos = response_text.find(start_tag, start_idx)
        if start_pos == -1:
            break
            
        end_pos = response_text.find(end_tag, start_pos)
        if end_pos == -1:
            break
            
        # 提取并解析函数调用
        function_json = response_text[start_pos + len(start_tag):end_pos].strip()
        try:
            function_call = json.loads(function_json)
            function_calls.append(function_call)
        except json.JSONDecodeError:
            st.error(f"解析函数调用失败: {function_json}")
            
        start_idx = end_pos + len(end_tag)
    
    return function_calls if function_calls else None

def execute_function_call(function_call):
    """执行单个函数调用并返回结果，同时更新UI状态"""
    if not function_call or "name" not in function_call:
        return {"status": "error", "message": "无效的函数调用"}
    
    function_name = function_call["name"]
    parameters = function_call.get("parameters", {})
    result = {"status": "error", "message": f"未找到函数: {function_name}"}
    
    # 打开网页 - 与UI联动
    if function_name == "open_url":
        url = parameters.get("url", "")
        if not url:
            return {"status": "error", "message": "缺少URL参数"}
        
        # 更新UI状态：将URL填入输入框并触发点击
        st.session_state["url_input"] = url
        st.session_state["trigger_open_url"] = True
        
        # 短暂等待UI更新
        time.sleep(1)
            
        try:
            response = requests.post(
                f"{API_BASE_URL}/open_url",
                json={"url": url}
            )
            if response.status_code == 200:
                result = {"status": "success", "message": "网页已打开", "data": {"url": url}}
            else:
                result = {"status": "error", "message": f"失败: {response.json().get('detail')}"}
        except Exception as e:
            result = {"status": "error", "message": f"错误: {str(e)}"}
        
        # 重置触发状态
        st.session_state["trigger_open_url"] = False
    
    # 关闭浏览器
    elif function_name == "close_browser":
        try:
            response = requests.post(f"{API_BASE_URL}/close_browser")
            if response.status_code == 200:
                result = {"status": "success", "message": "浏览器已关闭"}
            else:
                result = {"status": "error", "message": f"失败: {response.json().get('detail')}"}
        except Exception as e:
            result = {"status": "error", "message": f"错误: {str(e)}"}
    
    # 单次截图 - 与UI联动
    elif function_name == "take_screenshot":
        element_selector = parameters.get("element_selector", "")
        
        # 更新UI状态：将选择器填入输入框并触发截图
        st.session_state["element_selector_input"] = element_selector
        st.session_state["trigger_take_screenshot"] = True
        
        # 短暂等待UI更新
        time.sleep(1)
            
        try:
            response = requests.post(
                f"{API_BASE_URL}/take_screenshot",
                json={"element_selector": element_selector}
            )
            if response.status_code == 200:
                screenshot_path = response.json().get("screenshot_path")
                
                # 生成唯一ID和时间戳
                screenshot_id = generate_unique_id()
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 添加到截图列表
                st.session_state["screenshots"].append({
                    "id": screenshot_id,
                    "path": screenshot_path,
                    "timestamp": timestamp
                })
                
                result = {
                    "status": "success", 
                    "message": f"截图成功！已添加 {len(st.session_state['screenshots'])} 张截图",
                    "data": {
                        "screenshot_id": screenshot_id,
                        "path": screenshot_path
                    }
                }
            else:
                result = {"status": "error", "message": f"失败: {response.json().get('detail')}"}
        except Exception as e:
            result = {"status": "error", "message": f"错误: {str(e)}"}
        
        # 重置触发状态
        st.session_state["trigger_take_screenshot"] = False
    
    # 获取全页截图
    elif function_name == "take_fullpage_screenshot":
        try:
            response = requests.post(
                f"{API_BASE_URL}/take_fullpage_screenshot"
            )
            if response.status_code == 200:
                screenshot_path = response.json().get("screenshot_path")
                st.session_state["full_page_screenshot"] = screenshot_path
                st.session_state["selection_coords"] = ""
                
                result = {
                    "status": "success", 
                    "message": "已获取全页截图",
                    "data": {"path": screenshot_path}
                }
            else:
                result = {"status": "error", "message": f"失败: {response.json().get('detail')}"}
        except Exception as e:
            result = {"status": "error", "message": f"错误: {str(e)}"}
    
    # 截取所选区域
    elif function_name == "take_selected_screenshot":
        coords = parameters.get("coordinates")
        if not coords or len(coords) != 4:
            return {"status": "error", "message": "缺少或无效的坐标参数"}
            
        if not st.session_state["full_page_screenshot"]:
            return {"status": "error", "message": "请先获取全页截图"}
            
        try:
            left, top, width, height = coords
            
            response = requests.post(
                f"{API_BASE_URL}/take_selected_screenshot",
                json={
                    "fullpage_screenshot_path": st.session_state["full_page_screenshot"],
                    "region": {
                        "left": float(left),
                        "top": float(top),
                        "width": float(width),
                        "height": float(height)
                    }
                }
            )
            
            if response.status_code == 200:
                screenshot_path = response.json().get("screenshot_path")
                
                # 生成唯一ID和时间戳
                screenshot_id = generate_unique_id()
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 添加到截图列表
                st.session_state["screenshots"].append({
                    "id": screenshot_id,
                    "path": screenshot_path,
                    "timestamp": timestamp
                })
                
                result = {
                    "status": "success", 
                    "message": f"截图成功！已添加 {len(st.session_state['screenshots'])} 张截图",
                    "data": {
                        "screenshot_id": screenshot_id,
                        "path": screenshot_path
                    }
                }
            else:
                result = {"status": "error", "message": f"失败: {response.json().get('detail')}"}
        except Exception as e:
            result = {"status": "error", "message": f"错误: {str(e)}"}
    
    # 批量分析订单 - 与UI联动
    elif function_name == "batch_analyze_orders":
        if not st.session_state["screenshots"]:
            return {"status": "error", "message": "请先截取至少一张订单截图"}
        
        # 更新UI状态：触发批量分析
        st.session_state["trigger_batch_analysis"] = True
        
        # 短暂等待UI更新
        time.sleep(1)
            
        try:
            # 清空之前的分析结果
            st.session_state["batch_analysis_results"] = []
            
            # 逐个分析截图，保存所有订单信息
            success_count = 0
            for idx, screenshot in enumerate(st.session_state["screenshots"], 1):
                response = requests.post(
                    f"{API_BASE_URL}/analyze_screenshot",
                    params={"screenshot_path": screenshot["path"]}
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    # 确保order_data是字典类型
                    if isinstance(result_data.get("order_data"), dict):
                        order_data = result_data.get("order_data")
                        # 添加截图ID用于关联
                        order_data["screenshot_id"] = screenshot["id"]
                        order_data["screenshot_timestamp"] = screenshot["timestamp"]
                        st.session_state["batch_analysis_results"].append(order_data)
                        success_count += 1
                    elif isinstance(result_data.get("order_data"), list):
                        # 如果返回的是订单数组，逐个添加
                        for item in result_data.get("order_data", []):
                            if isinstance(item, dict):
                                item["screenshot_id"] = screenshot["id"]
                                item["screenshot_timestamp"] = screenshot["timestamp"]
                                st.session_state["batch_analysis_results"].append(item)
                                success_count += 1
                    else:
                        st.warning(f"第 {idx} 张截图返回了不支持的数据格式")
                else:
                    error_msg = f"分析失败: {response.json().get('detail')}"
                    st.warning(f"第 {idx} 张截图分析失败: {error_msg}")
            
            # 生成结果展示HTML
            results_html = display_analysis_results()
            
            result = {
                "status": "success", 
                "message": f"批量分析完成！成功分析 {success_count} 张截图",
                "data": {
                    "total_analyzed": len(st.session_state["screenshots"]),
                    "success_count": success_count,
                    "results_html": results_html
                }
            }
            
            # 触发结果显示
            st.session_state["show_analysis_results"] = True
            
        except Exception as e:
            result = {"status": "error", "message": f"分析过程出错: {str(e)}"}
        
        # 重置触发状态
        st.session_state["trigger_batch_analysis"] = False
    
    # 清除所有数据
    elif function_name == "clear_all_data":
        st.session_state["screenshots"] = []
        st.session_state["batch_analysis_results"] = []
        st.session_state["full_page_screenshot"] = None
        st.session_state["selection_coords"] = ""
        st.session_state["show_analysis_results"] = False
        st.session_state["url_input"] = ""
        st.session_state["element_selector_input"] = ""
        result = {"status": "success", "message": "已清除所有截图和分析结果"}
    
    # 记录函数调用结果
    func_result_entry = {
        "function": function_name,
        "parameters": parameters,
        "result": result,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state["function_results"].append(func_result_entry)
    
    return result, func_result_entry

def process_function_sequence():
    """处理函数调用序列，连续执行所有操作直到完成"""
    # 如果没有函数需要执行，直接返回
    if not st.session_state["function_call_sequence"]:
        st.session_state["is_thinking"] = False
        st.session_state["current_function_index"] = 0
        return True
        
    # 获取当前要执行的函数
    current_index = st.session_state["current_function_index"]
    if current_index >= len(st.session_state["function_call_sequence"]):
        # 所有函数执行完毕，获取最终AI反馈
        get_final_ai_response()
        
        # 重置状态
        st.session_state["is_thinking"] = False
        st.session_state["current_function_index"] = 0
        st.session_state["function_call_sequence"] = []
        return True
    
    # 显示当前步骤信息
    st.markdown(f'''
    <div class="step-indicator">
        <i class="fas fa-cog fa-spin"></i> 正在执行步骤 {current_index + 1}/{len(st.session_state["function_call_sequence"])}: 
        {st.session_state["function_call_sequence"][current_index]["name"]}
    </div>
    ''', unsafe_allow_html=True)
    
    # 执行当前函数
    current_function = st.session_state["function_call_sequence"][current_index]
    function_result, func_result_entry = execute_function_call(current_function)
    
    # 显示当前步骤结果
    status = "成功" if function_result["status"] == "success" else "失败"
    status_color = "#10b981" if function_result["status"] == "success" else "#ef4444"
    st.markdown(f'''
    <div style="
        text-align: left;
        margin-bottom: 8px;
        padding-left: 10px;
    ">
        <div class="function-result">
            <strong style="color: {status_color};">步骤 {current_index + 1}/{len(st.session_state["function_call_sequence"])}: {current_function["name"]} {status}</strong>
            <br>{function_result["message"]}
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 在步骤之间增加等待
    st.markdown(f'''
    <div class="step-indicator">
        <i class="fas fa-clock"></i> 等待3秒后执行下一步...
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(3)  # 等待3秒
    
    # 更新索引
    st.session_state["current_function_index"] += 1
    
    # 自动触发下一个函数
    st.rerun()

def get_final_ai_response():
    """所有函数执行完成后获取最终AI反馈"""
    try:
        # 构建请求，包含所有函数执行结果
        functions = [
            {
                "name": "open_url",
                "description": "打开指定的URL网页",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "要打开的网页URL地址"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "close_browser",
                "description": "关闭浏览器",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "take_screenshot",
                "description": "对当前页面或指定元素进行截图",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_selector": {
                            "type": "string",
                            "description": "CSS选择器，用于指定要截图的元素，为空则截取全屏"
                        }
                    }
                }
            },
            {
                "name": "take_fullpage_screenshot",
                "description": "获取当前页面的全页截图，用于后续手动选择区域",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "take_selected_screenshot",
                "description": "截取之前选择的区域",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinates": {
                            "type": "array",
                            "description": "区域坐标，格式为[left, top, width, height]，百分比值",
                            "items": {"type": "number"}
                        }
                    },
                    "required": ["coordinates"]
                }
            },
            {
                "name": "batch_analyze_orders",
                "description": "批量分析所有已截取的订单截图，并展示详细结果",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "clear_all_data",
                "description": "清除所有截图和分析结果",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        # 准备所有函数执行结果摘要
        execution_summary = {
            "total_steps": len(st.session_state["function_results"]),
            "success_steps": sum(1 for res in st.session_state["function_results"] if res["result"]["status"] == "success"),
            "last_step": st.session_state["function_results"][-1] if st.session_state["function_results"] else None
        }
        
        chat_request = {
            "message": f"所有操作已完成，执行结果: {json.dumps(execution_summary, ensure_ascii=False)}",
            "history": st.session_state["chat_history"],
            "functions": functions,
            "function_call": "auto",
            "execution_summary": execution_summary
        }
        
        # 调用API获取最终响应
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=chat_request,
            timeout=60  # 设置超时时间
        )
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            if result["status"] == "success":
                assistant_response = result["response"]
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": assistant_response
                })
            else:
                error_msg = f"处理失败: {result.get('detail', '未知错误')}"
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": error_msg
                })
        else:
            error_detail = response.json().get("detail", "未知错误")
            error_msg = f"API调用失败: {error_detail}"
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": error_msg
            })
        
    except Exception as e:
        error_msg = f"通信错误: {str(e)}"
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": error_msg
        })

# 调整三栏布局比例
left_col, middle_col, right_col = st.columns([0.8, 1.2, 1.2])

# 左侧区域：带完整消息发送功能的聊天区域，支持函数调用
with left_col:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    
    if selected_tab == "智能识别订单":
        st.subheader("💬 订单助手")
        st.caption("可以通过自然语言指令控制订单处理流程")
        
        # 添加清除聊天记录按钮
        if st.button("清除聊天记录", key="clear_chat", use_container_width=True, type="secondary"):
            clear_chat_history()
        
        # 展示聊天记录
        if st.session_state["chat_history"]:
            for msg in st.session_state["chat_history"]:
                if msg["role"] == "user":
                    # 用户消息样式 - 添加自动换行类
                    st.markdown(f'''
                    <div style="
                        text-align: right;
                        margin-bottom: 8px;
                        padding-right: 10px;
                    ">
                        <div style="
                            display: inline-block;
                            background-color: #4285f4;
                            color: white;
                            padding: 8px 12px;
                            border-radius: 18px 18px 4px 18px;
                            max-width: 80%;
                            text-align: left;
                        " class="chat-message">
                            {msg["content"]}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    # 助手消息样式 - 添加自动换行类
                    content = msg["content"]
                    
                    # 检查是否包含函数调用
                    function_calls = extract_function_calls(content)
                    
                    # 如果有函数调用，格式化显示
                    if function_calls:
                        # 提取纯文本部分
                        text_part = content
                        for call in function_calls:
                            text_part = text_part.replace(
                                f"<function_call>{json.dumps(call)}</function_call>", 
                                ""
                            ).strip()
                        
                        st.markdown(f'''
                        <div style="
                            text-align: left;
                            margin-bottom: 8px;
                            padding-left: 10px;
                        ">
                            <div style="
                                display: inline-block;
                                background-color: #e0e0e0;
                                color: #333;
                                padding: 8px 12px;
                                border-radius: 18px 18px 18px 4px;
                                max-width: 80%;
                            " class="chat-message">
                                {text_part}
                            </div>
                            <div>
                                <strong>执行操作序列:</strong>
                                <div class="function-call-sequence">
                        ''', unsafe_allow_html=True)
                        
                        # 显示每个函数调用
                        for i, call in enumerate(function_calls):
                            st.markdown(f'''
                                <div class="function-call">
                                    <strong>步骤 {i+1}:</strong> {call["name"]}
                                    <br>
                                    <strong>参数:</strong> {json.dumps(call.get("parameters", {}), ensure_ascii=False)}
                                </div>
                            ''', unsafe_allow_html=True)
                        
                        st.markdown(f'''
                                </div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        # 普通回复 - 添加自动换行类
                        st.markdown(f'''
                        <div style="
                            text-align: left;
                            margin-bottom: 8px;
                            padding-left: 10px;
                        ">
                            <div style="
                                display: inline-block;
                                background-color: #e0e0e0;
                                color: #333;
                                padding: 8px 12px;
                                border-radius: 18px 18px 18px 4px;
                                max-width: 80%;
                            " class="chat-message">
                                {content}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
        else:
            # 空状态提示
            st.markdown('''
            <div style="text-align: center; color: #999; padding: 20px 0;">
                暂无聊天记录，开始对话吧！<br>
                示例: "帮我打开京东并截取订单页面"
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # 关闭聊天记录容器
        
        # 消息输入和发送区域
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        
        # 输入框与发送按钮并排布局
        input_col, button_col = st.columns([8, 2])
        
        with input_col:
            user_input = st.text_input(
                "消息输入",
                value="",
                placeholder="输入消息...",
                key="chat_input",
                autocomplete="off",
                label_visibility="collapsed"
            )
        
        with button_col:
            send_button = st.button(
                "发送", 
                use_container_width=True,
                type="primary",
                disabled=st.session_state["is_thinking"]  # 思考时禁用按钮
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 加载状态显示
        if st.session_state["is_thinking"]:
            if st.session_state["function_call_sequence"] and st.session_state["current_function_index"] > 0:
                # 显示函数执行进度
                total = len(st.session_state["function_call_sequence"])
                current = st.session_state["current_function_index"]
                st.markdown(f'''
                <div class="typing-indicator">
                    <i class="fas fa-circle-notch fa-spin"></i> 正在执行步骤 {current}/{total}...
                </div>
                ''', unsafe_allow_html=True)
            else:
                # 显示思考状态
                st.markdown(f'''
                <div class="typing-indicator">
                    <i class="fas fa-circle-notch fa-spin"></i> 正在处理...
                </div>
                ''', unsafe_allow_html=True)
        
        # 消息发送逻辑（按钮点击或回车键）
        if (
            (user_input.strip() and send_button) or  # 按钮发送
            (user_input.strip() and not st.session_state["is_thinking"] and 
             user_input.strip() != st.session_state.get("last_submitted_message", ""))  # 回车发送
        ):
            # 保存并清空输入
            st.session_state["last_submitted_message"] = user_input.strip()
            st.session_state["chat_history"].append({
                "role": "user",
                "content": user_input.strip()
            })
            st.session_state["is_thinking"] = True
            st.session_state["function_call_sequence"] = []
            st.session_state["current_function_index"] = 0
            
            # 刷新页面以显示新消息
            st.rerun()
        
        # 处理AI响应和函数调用
        if st.session_state["is_thinking"]:
            # 检查是否有函数序列需要处理
            if st.session_state["function_call_sequence"]:
                # 处理函数序列（自动触发下一个函数）
                process_function_sequence()
            else:
                # 首次获取AI响应
                with st.spinner(""):  # 隐藏默认spinner，使用自定义样式
                    try:
                        # 构建请求，包含函数调用能力说明
                        functions = [
                            {
                                "name": "open_url",
                                "description": "打开指定的URL网页",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "url": {
                                            "type": "string",
                                            "description": "要打开的网页URL地址"
                                        }
                                    },
                                    "required": ["url"]
                                }
                            },
                            {
                                "name": "close_browser",
                                "description": "关闭浏览器",
                                "parameters": {
                                    "type": "object",
                                    "properties": {}
                                }
                            },
                            {
                                "name": "take_screenshot",
                                "description": "对当前页面或指定元素进行截图",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "element_selector": {
                                            "type": "string",
                                            "description": "CSS选择器，用于指定要截图的元素，为空则截取全屏"
                                        }
                                    }
                                }
                            },
                            {
                                "name": "take_fullpage_screenshot",
                                "description": "获取当前页面的全页截图，用于后续手动选择区域",
                                "parameters": {
                                    "type": "object",
                                    "properties": {}
                                }
                            },
                            {
                                "name": "take_selected_screenshot",
                                "description": "截取之前选择的区域",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "coordinates": {
                                            "type": "array",
                                            "description": "区域坐标，格式为[left, top, width, height]，百分比值",
                                            "items": {"type": "number"}
                                        }
                                    },
                                    "required": ["coordinates"]
                                }
                            },
                            {
                                "name": "batch_analyze_orders",
                                "description": "批量分析所有已截取的订单截图，并展示详细结果",
                                "parameters": {
                                    "type": "object",
                                    "properties": {}
                                }
                            },
                            {
                                "name": "clear_all_data",
                                "description": "清除所有截图和分析结果",
                                "parameters": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        ]
                        
                        chat_request = {
                            "message": st.session_state["last_submitted_message"],
                            "history": st.session_state["chat_history"],
                            "functions": functions,
                            "function_call": "auto"
                        }
                        
                        # 调用API获取响应
                        response = requests.post(
                            f"{API_BASE_URL}/chat",
                            json=chat_request,
                            timeout=60  # 设置超时时间
                        )
                        
                        # 处理响应
                        if response.status_code == 200:
                            result = response.json()
                            if result["status"] == "success":
                                assistant_response = result["response"]
                                st.session_state["chat_history"].append({
                                    "role": "assistant",
                                    "content": assistant_response
                                })
                                
                                # 检查是否包含函数调用并执行
                                function_calls = extract_function_calls(assistant_response)
                                if function_calls:
                                    st.session_state["function_call_sequence"] = function_calls
                                    st.session_state["current_function_index"] = 0
                                    st.rerun()
                                else:
                                    st.session_state["is_thinking"] = False
                            else:
                                error_msg = f"处理失败: {result.get('detail', '未知错误')}"
                                st.session_state["chat_history"].append({
                                    "role": "assistant",
                                    "content": error_msg
                                    })
                                st.session_state["is_thinking"] = False
                        else:
                            error_detail = response.json().get("detail", "未知错误")
                            error_msg = f"API调用失败: {error_detail}"
                            st.session_state["chat_history"].append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            st.session_state["is_thinking"] = False
                        
                    except Exception as e:
                        error_msg = f"通信错误: {str(e)}"
                        st.session_state["chat_history"].append({
                            "role": "assistant",
                            "content": error_msg
                        })
                        st.session_state["is_thinking"] = False
                    finally:
                        if not st.session_state["function_call_sequence"]:
                            st.rerun()
    else:
        pass
    
    st.markdown('</div>', unsafe_allow_html=True)

# 中间：操作区域，修复布局紧凑性+label空值问题
with middle_col:
    if selected_tab == "智能识别订单":
        st.subheader("智能识别订单")
        
        with st.expander("使用指南", expanded=False):
            st.write("""
            1. 输入订单页面URL并打开网页
            2. 选择截图方式：
               - 方式一：使用元素选择器自动截图
               - 方式二：手动选择区域截图（推荐）
            3. 多次截取不同订单区域
            4. 点击"批量分析订单"处理所有截图
            5. 在下方查看所有订单结果并保存
            
            也可以通过左侧聊天窗口发送指令自动完成以上操作，例如：
            - "帮我打开https://example.com/orders"
            - "截取当前页面的订单区域"
            - "分析所有订单截图"
            """)
        
        # URL输入框与UI联动
        url = st.text_input(
            "输入订单URL", 
            value=st.session_state["url_input"],
            autocomplete="off",
            key="url_input_field"
        )
        # 同步到session_state
        st.session_state["url_input"] = url
        
        col1, col2 = st.columns(2)
        with col1:
            # 打开网页按钮，支持自动触发
            open_url_clicked = st.button("打开网页", use_container_width=True) or st.session_state["trigger_open_url"]
            if open_url_clicked and url:
                with st.spinner("打开中..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/open_url",
                            json={"url": url}
                        )
                        if response.status_code == 200:
                            st.success("网页已打开")
                        else:
                            st.error(f"失败: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"错误: {str(e)}")
        
        with col2:
            if st.button("关闭浏览器", type="secondary", use_container_width=True):
                with st.spinner("关闭中..."):
                    try:
                        response = requests.post(f"{API_BASE_URL}/close_browser")
                        if response.status_code == 200:
                            st.success("浏览器已关闭")
                        else:
                            st.error(f"失败: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"错误: {str(e)}")
        
        # 截图方式选择
        screenshot_method = st.radio(
            "订单截图范围",
            ["默认全屏", "手动选择区域"],
            horizontal=True
        )
        
        # 元素选择器方式
        if screenshot_method == "默认全屏":
            # 元素选择器输入框与UI联动
            element_selector = st.text_input(
                "CSS元素选择器",
                value=st.session_state["element_selector_input"],
                autocomplete="off",
                placeholder="可选：输入CSS选择器（如#order-container）",
                key="element_selector_field"
            )
            # 同步到session_state
            st.session_state["element_selector_input"] = element_selector
            
            # 单次截图按钮，支持自动触发
            take_screenshot_clicked = st.button("单次截图", use_container_width=True) or st.session_state["trigger_take_screenshot"]
            if take_screenshot_clicked:
                with st.spinner("截图中..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/take_screenshot",
                            json={"element_selector": element_selector}
                        )
                        if response.status_code == 200:
                            screenshot_path = response.json().get("screenshot_path")
                            
                            # 生成唯一ID和时间戳
                            screenshot_id = generate_unique_id()
                            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # 添加到截图列表
                            st.session_state["screenshots"].append({
                                "id": screenshot_id,
                                "path": screenshot_path,
                                "timestamp": timestamp
                            })
                            
                            st.success(f"截图成功！已添加 {len(st.session_state['screenshots'])} 张截图")
                            
                            # 显示最新截图
                            if os.path.exists(screenshot_path):
                                image = Image.open(screenshot_path)
                                st.image(image, caption="截图结果", use_column_width=True)
                        else:
                            st.error(f"失败: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"错误: {str(e)}")
        
        # 手动选择区域方式
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("获取全页截图", use_container_width=True):
                    with st.spinner("获取全页截图中..."):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/take_fullpage_screenshot"
                            )
                            if response.status_code == 200:
                                screenshot_path = response.json().get("screenshot_path")
                                st.session_state["full_page_screenshot"] = screenshot_path
                                st.session_state["selection_coords"] = ""
                                st.success("已获取全页截图，请在下方选择区域")
                                st.rerun()
                            else:
                                st.error(f"失败: {response.json().get('detail')}")
                        except Exception as e:
                            st.error(f"错误: {str(e)}")
            
            with col_b:
                if st.button("重置选择区域", use_container_width=True, type="secondary"):
                    st.session_state["selection_coords"] = ""
                    st.success("已重置选择区域")
        
        # 显示全页截图并允许选择区域
        if st.session_state["full_page_screenshot"] and os.path.exists(st.session_state["full_page_screenshot"]):
            st.markdown("### 选择截图区域")
            st.write("在下方图片上按住鼠标拖动选择需要截取的区域")
            
            # 隐藏的输入框用于存储选择的坐标
            st.markdown(f'''
            <input type="hidden" id="selection-coords" value="{st.session_state['selection_coords']}">
            <div id="selection-container" class="selection-container">
                <img id="selection-image" class="selection-image" src="data:image/png;base64,{st.session_state['full_page_screenshot']}" alt="全页截图">
                <div id="selection-box" class="selection-box"></div>
            </div>
            <p id="selection-info" class="text-sm text-gray-600">
                { "已选择区域" if st.session_state['selection_coords'] else "未选择区域" }
            </p>
            ''', unsafe_allow_html=True)
            
            # 确认选择并截图
            if st.button("截取所选区域", use_container_width=True, type="primary"):
                if not st.session_state["selection_coords"]:
                    st.warning("请先选择截图区域")
                else:
                    with st.spinner("截取所选区域中..."):
                        try:
                            # 解析坐标
                            left, top, width, height = st.session_state["selection_coords"].split(',')
                            
                            response = requests.post(
                                f"{API_BASE_URL}/take_selected_screenshot",
                                json={
                                    "fullpage_screenshot_path": st.session_state["full_page_screenshot"],
                                    "region": {
                                        "left": float(left),
                                        "top": float(top),
                                        "width": float(width),
                                        "height": float(height)
                                    }
                                }
                            )
                            
                            if response.status_code == 200:
                                screenshot_path = response.json().get("screenshot_path")
                                
                                # 生成唯一ID和时间戳
                                screenshot_id = generate_unique_id()
                                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                # 添加到截图列表
                                st.session_state["screenshots"].append({
                                    "id": screenshot_id,
                                    "path": screenshot_path,
                                    "timestamp": timestamp
                                })
                                
                                st.success(f"截图成功！已添加 {len(st.session_state['screenshots'])} 张截图")
                                
                                # 显示截取的区域
                                if os.path.exists(screenshot_path):
                                    image = Image.open(screenshot_path)
                                    st.image(image, caption="截取的区域", use_column_width=True)
                            else:
                                st.error(f"失败: {response.json().get('detail')}")
                        except Exception as e:
                            st.error(f"错误: {str(e)}")
        
        # 批量分析订单按钮，支持自动触发
        batch_analyze_clicked = st.button("批量分析订单", use_container_width=True, type="primary") or st.session_state["trigger_batch_analysis"]
        if batch_analyze_clicked:
            if not st.session_state["screenshots"]:
                st.warning("请先截取至少一张订单截图")
            else:
                with st.spinner(f"正在批量分析 {len(st.session_state['screenshots'])} 张截图..."):
                    try:
                        # 清空之前的分析结果
                        st.session_state["batch_analysis_results"] = []
                        
                        # 逐个分析截图，保存所有订单信息
                        success_count = 0
                        for idx, screenshot in enumerate(st.session_state["screenshots"], 1):
                            with st.spinner(f"分析第 {idx}/{len(st.session_state['screenshots'])} 张截图..."):
                                response = requests.post(
                                    f"{API_BASE_URL}/analyze_screenshot",
                                    params={"screenshot_path": screenshot["path"]}
                                )
                                
                                if response.status_code == 200:
                                    result_data = response.json()
                                    # 确保order_data是字典类型
                                    if isinstance(result_data.get("order_data"), dict):
                                        order_data = result_data.get("order_data")
                                        # 添加截图ID用于关联
                                        order_data["screenshot_id"] = screenshot["id"]
                                        order_data["screenshot_timestamp"] = screenshot["timestamp"]
                                        st.session_state["batch_analysis_results"].append(order_data)
                                        success_count += 1
                                    elif isinstance(result_data.get("order_data"), list):
                                        # 如果返回的是订单数组，逐个添加
                                        for item in result_data.get("order_data", []):
                                            if isinstance(item, dict):
                                                item["screenshot_id"] = screenshot["id"]
                                                item["screenshot_timestamp"] = screenshot["timestamp"]
                                                st.session_state["batch_analysis_results"].append(item)
                                                success_count += 1
                                    else:
                                        st.warning(f"第 {idx} 张截图返回了不支持的数据格式")
                                else:
                                    error_msg = f"分析失败: {response.json().get('detail')}"
                                    st.warning(f"第 {idx} 张截图分析失败: {error_msg}")
                        
                        st.success(f"批量分析完成！成功分析 {success_count} 张截图")
                        st.session_state["show_analysis_results"] = True
                    except Exception as e:
                        st.error(f"分析过程出错: {str(e)}")
        
        # 显示分析结果（如果有）
        if st.session_state["show_analysis_results"] and st.session_state["batch_analysis_results"]:
            st.markdown('<div class="batch-results">', unsafe_allow_html=True)
            st.subheader("所有订单分析结果")
            
            # 显示每个订单的完整信息
            for i, order in enumerate(st.session_state["batch_analysis_results"], 1):
                # 确保order是字典
                if not isinstance(order, dict):
                    st.warning(f"订单 #{i} 数据格式不正确，跳过显示")
                    continue
                    
                st.markdown(f'<div class="order-card">', unsafe_allow_html=True)
                st.subheader(f"订单 #{i}")
                
                # 订单基本信息
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**订单号:** {order.get('订单号', 'N/A')}")
                    st.write(f"**订单日期:** {order.get('订单日期', 'N/A')}")
                with col2:
                    st.write(f"**截图时间:** {order.get('screenshot_timestamp', 'N/A')}")
                    st.write(f"**总金额:** {order.get('总金额', 'N/A')}")
                
                # 商品列表
                st.write("**商品列表:**")
                products = order.get('商品名称列表', [])
                # 确保products是列表
                if not isinstance(products, list):
                    st.info("未识别到有效的商品列表")
                elif products:
                    product_data = []
                    for product in products:
                        # 处理商品数据，确保是字典
                        if isinstance(product, dict):
                            product_info = {
                                "商品名称": product.get("名称", "未知名称"),
                                "数量": product.get("数量", "N/A"),
                                "单价": product.get("单价", "N/A"),
                                "总价": product.get("小计", "N/A")
                            }
                        elif isinstance(product, list):
                            # 如果是列表，尝试转换为字符串描述
                            product_info = {
                                "商品名称": f"商品信息: {', '.join(map(str, product))}",
                                "数量": "N/A",
                                "单价": "N/A",
                                "总价": "N/A"
                            }
                        else:
                            # 其他类型直接转换为字符串
                            product_info = {
                                "商品名称": str(product),
                                "数量": "N/A",
                                "单价": "N/A",
                                "总价": "N/A"
                            }
                        product_data.append(product_info)
                    
                    df = pd.DataFrame(product_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("未识别到商品信息")
                
                st.markdown('</div>', unsafe_allow_html=True)  # 关闭订单卡片
            
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "系统设置":
        st.subheader("系统设置")
        
        # 配置表单
        st.markdown('<div class="config-form">', unsafe_allow_html=True)
        st.subheader("模型配置")
        
        # 输入框，预填当前值
        new_server_host = st.text_input(
            "服务器地址 (SERVER_HOST)", 
            value=SERVER_HOST,
            key="server_host_input"
        )
        
        new_server_port = st.text_input(
            "服务器端口 (SERVER_PORT)", 
            value=str(SERVER_PORT),
            key="server_port_input"
        )
        
        new_frontend_port = st.text_input(
            "前端端口 (FRONTEND_PORT)", 
            value=str(FRONTEND_PORT),
            key="frontend_port_input"
        )
        
        new_model_name = st.text_input(
            "模型名称 (MIDSCENE_MODEL_NAME)", 
            value=MIDSCENE_MODEL_NAME,
            key="model_name_input"
        )
        
        new_openai_base_url = st.text_input(
            "API地址 (OPENAI_BASE_URL)", 
            value=OPENAI_BASE_URL,
            key="openai_base_url_input"
        )
        
        # 密码框显示API密钥，默认显示掩码
        new_openai_api_key = st.text_input(
            "API密钥 (OPENAI_API_KEY)", 
            value=OPENAI_API_KEY,
            type="password",
            key="openai_api_key_input"
        )
        
        # 浏览器配置
        new_chrome_path = st.text_input(
            "Chrome路径 (CHROME_PATH)", 
            value=CHROME_PATH,
            key="chrome_path_input"
        )
        
        new_chrome_driver_path = st.text_input(
            "Chrome驱动路径 (CHROME_DRIVER_PATH)", 
            value=CHROME_DRIVER_PATH,
            key="chrome_driver_path_input"
        )
        
        # 保存配置按钮
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("保存配置", use_container_width=True, type="primary"):
                # 验证端口是否为数字
                try:
                    if new_server_port:
                        int(new_server_port)
                    if new_frontend_port:
                        int(new_frontend_port)
                except ValueError:
                    st.error("端口号必须是数字")
                else:
                    # 准备要保存的配置
                    configs = {
                        "SERVER_HOST": new_server_host,
                        "SERVER_PORT": new_server_port,
                        "FRONTEND_PORT": new_frontend_port,
                        "MIDSCENE_MODEL_NAME": new_model_name,
                        "OPENAI_BASE_URL": new_openai_base_url,
                        "OPENAI_API_KEY": new_openai_api_key,
                        "CHROME_PATH": new_chrome_path,
                        "CHROME_DRIVER_PATH": new_chrome_driver_path
                    }
                    
                    # 保存配置
                    success, message = save_config_to_py(configs)
                    if success:
                        st.session_state["show_config_success"] = True
                        st.success(message + "，请重启应用使配置生效")
                    else:
                        st.error(message)
        
        with col_cancel:
            if st.button("取消", use_container_width=True, type="secondary"):
                st.session_state["show_config_success"] = False
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.write("**当前生效配置**")
        st.write("**API服务器:**", API_BASE_URL)
        st.write("**截图路径:**", SCREENSHOTS_DIR.split(os.sep)[-1])
        st.write("**Excel路径:**", EXCELS_DIR.split(os.sep)[-1])
        
        st.write("\n**模型配置**")
        st.write("**当前模型:**", MIDSCENE_MODEL_NAME)
        st.write("**API地址:**", OPENAI_BASE_URL)
        
        st.markdown("---")
        st.subheader("环境变量信息")
        env_config = {
            "SERVER_HOST": str(SERVER_HOST),
            "SERVER_PORT": str(SERVER_PORT),
            "FRONTEND_PORT": str(FRONTEND_PORT),
            "模型名称": MIDSCENE_MODEL_NAME.split("-")[0] + "...",
            "API地址": OPENAI_BASE_URL
        }
        st.dataframe(pd.DataFrame(list(env_config.items()), columns=["配置项", "当前值"]))
        
        st.info("配置保存在config.py文件中，修改后需要重启应用才能生效")

# 右侧：所有订单数据汇总展示
with right_col:
    if selected_tab == "智能识别订单":
        st.subheader("订单分析汇总")
        
        if st.session_state["batch_analysis_results"]:
            # 显示所有订单的汇总表格
            all_products = []
            for order in st.session_state["batch_analysis_results"]:
                # 确保order是字典
                if not isinstance(order, dict):
                    continue
                    
                products = order.get('商品名称列表', [])
                # 确保products是列表
                if not isinstance(products, list):
                    continue
                    
                order_time = order.get('订单日期', 'N/A')
                order_number = order.get('订单号', f"未知-{order.get('screenshot_id', 'id')}")
                
                for product in products:
                    # 安全处理商品数据
                    if isinstance(product, dict):
                        formatted_product = {
                            "订单号": order_number,
                            "订单时间": order_time,
                            "商品名称": product.get("名称", "未知名称"),
                            "数量": product.get("数量", "N/A"),
                            "单价": product.get("单价", "N/A"),
                            "总价": product.get("小计", "N/A")
                        }
                    else:
                        formatted_product = {
                            "订单号": order_number,
                            "订单时间": order_time,
                            "商品名称": str(product),
                            "数量": "N/A",
                            "单价": "N/A",
                            "总价": "N/A"
                        }
                    all_products.append(formatted_product)
            
            if all_products:
                df = pd.DataFrame(all_products)[["订单号", "订单时间", "商品名称", "数量", "单价", "总价"]]
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, height=500)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 添加导出功能
                if st.button("导出为Excel", use_container_width=True):
                    try:
                        excel_path = os.path.join(EXCELS_DIR, f"订单汇总_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                        df.to_excel(excel_path, index=False)
                        st.success(f"已导出至: {excel_path}")
                        
                        # 提供下载链接
                        with open(excel_path, "rb") as file:
                            st.download_button(
                                label="下载Excel文件",
                                data=file,
                                file_name=os.path.basename(excel_path),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
            else:
                st.info("未从截图中识别到商品信息")
        else:
            st.info("请先进行订单分析")
        
        # 清除所有数据按钮
        if st.button("清除所有数据", type="secondary", use_container_width=True):
            st.session_state["screenshots"] = []
            st.session_state["batch_analysis_results"] = []
            st.session_state["full_page_screenshot"] = None
            st.session_state["selection_coords"] = ""
            st.session_state["show_analysis_results"] = False
            st.session_state["url_input"] = ""
            st.session_state["element_selector_input"] = ""
            st.success("已清除所有截图和分析结果")
            st.rerun()
    
    elif selected_tab == "系统设置":
        st.subheader("配置说明")
        st.write("""
        - **服务器地址**：后端服务的IP和端口
        - **截图路径**：订单截图保存目录
        - **Excel路径**：订单数据保存目录
        - **模型配置**：豆包模型参数
        
        配置修改格式：
        ```
        SERVER_HOST=localhost
        SERVER_PORT=8000
        OPENAI_API_KEY=your_key
        ```
        """)
        
        st.subheader("常见问题")
        with st.expander("无法连接服务器"):
            st.write("1. 检查后端是否启动\n2. 确认服务器地址配置正确\n3. 检查网络连接")
        
        with st.expander("截图失败"):
            st.write("1. 确认浏览器配置正确\n2. 检查网页是否可访问\n3. 尝试不同选择器或手动选择区域")
        
        with st.expander("AI分析失败"):
            st.write("1. 检查API密钥\n2. 确认网络正常\n3. 尝试更清晰的截图")

if __name__ == "__main__":
    pass
