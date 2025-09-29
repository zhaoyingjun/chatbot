import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 服务器配置
SERVER_HOST = 'localhost'
SERVER_PORT = '8000'
FRONTEND_PORT = '8501'

# 豆包模型配置
MIDSCENE_MODEL_NAME = 'doubao-seed-1-6-250615'
OPENAI_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
OPENAI_API_KEY = ''

# 路径配置
SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")
EXCELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "excels")

# 浏览器配置
CHROME_PATH = ''
CHROME_DRIVER_PATH = './dist/resources/chromedriver.exe'

# 确保目录存在
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(EXCELS_DIR, exist_ok=True)