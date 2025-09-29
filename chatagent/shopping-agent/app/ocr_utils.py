import base64
import requests
import json
from .config import MIDSCENE_MODEL_NAME, OPENAI_BASE_URL, OPENAI_API_KEY

class OCRUtils:
    def __init__(self):
        self.model_name = MIDSCENE_MODEL_NAME
        self.api_base = OPENAI_BASE_URL
        self.api_key = OPENAI_API_KEY
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def encode_image(self, image_path):
        """将图片编码为base64格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_order_screenshot(self, image_path):
        """
        分析订单截图，提取所有订单信息
        返回解析后的所有订单数据列表
        """
        try:
            # 编码图片
            base64_image = self.encode_image(image_path)
            
            # 构建提示信息 - 修改为提取所有订单信息
            prompt = """
            请仔细分析这张购物订单截图，提取其中所有订单的信息并以JSON格式返回。
            如果截图中包含多个订单，请返回一个订单对象的数组。每个订单应包含以下信息：
            - 订单号
            - 订单日期
            - 商品名称列表（每个商品包括名称、数量、单价、小计）
            - 总金额
            - 支付方式
            - 收货地址
            - 收货人
            - 联系电话
            
            如果某些信息不存在，请对应字段值设为null。
            确保返回的是纯JSON数组，不要有其他文本。即使只有一个订单，也请返回包含一个元素的数组。
            """
        
            # 构建请求数据
            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                "temperature": 0.0
            }
            
            # 发送请求
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                data=json.dumps(data)
            )
          
            if response.status_code != 200:
                return False, f"API请求失败: {response.text}"
            
            # 解析响应
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 尝试解析JSON
            try:
                # 现在返回的是订单数组
                order_data = json.loads(content)
                return True, order_data
            except json.JSONDecodeError:
                return False, f"解析模型输出失败: {content}"
                
        except Exception as e:
            return False, f"分析图片时出错: {str(e)}"
