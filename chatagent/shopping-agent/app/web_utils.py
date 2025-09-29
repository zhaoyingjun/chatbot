from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
    ElementNotVisibleException,
    ElementNotInteractableException
)
import time
import os
import subprocess
import socket
import sys
import base64
from datetime import datetime
from PIL import Image  # 需要安装Pillow: pip install pillow
# 服务器配置
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8501))

# 豆包模型配置
MIDSCENE_MODEL_NAME = os.getenv("MIDSCENE_MODEL_NAME", "doubao-seed-1-6-vision-250815")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "c87d16de-85a4-4ea1-8cf1-a47a3dd64634")

# 路径配置
SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")
EXCELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "excels")

# 浏览器配置
CHROME_PATH = os.getenv("CHROME_PATH", "")
CHROME_DRIVER_PATH = os.getenv("CHROME_DRIVER_PATH", "")


class WebUtils:
    def __init__(self):
        self.driver = None
        self.current_url = None
        self.chrome_driver_path = self.find_chrome_driver()
        self.debug_port = self.find_available_port()
        self.initialize_driver()
    
    def find_available_port(self):
        """查找本地可用的端口，避免端口冲突"""
        for _ in range(5):  # 尝试5次获取端口
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', 0))  # 绑定到0会自动分配可用端口
                    port = s.getsockname()[1]
                    # 验证端口确实可用
                    if self.is_port_available(port):
                        return port
                except:
                    continue
        raise Exception("无法找到可用的端口，请检查系统端口占用情况")
    
    def is_port_available(self, port):
        """检查端口是否真的可用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    def find_chrome_driver(self):
        """查找本地Chrome驱动程序路径"""
        if CHROME_DRIVER_PATH and os.path.exists(CHROME_DRIVER_PATH):
            return CHROME_DRIVER_PATH
            
        common_paths = [
            "C:/chromedriver/chromedriver.exe",
            "C:/Program Files/Google/Chrome/Driver/chromedriver.exe",
            "C:/Program Files (x86)/Google/Chrome/Driver/chromedriver.exe",
            os.path.join(os.path.expanduser("~"), "chromedriver/chromedriver.exe")
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.path.isfile(path):
                return path
        
        raise Exception(f"""
        未找到Chrome驱动程序，请按以下步骤解决：
        1. 查看Chrome版本：在Chrome地址栏输入 chrome://version/
        2. 下载对应版本驱动：https://sites.google.com/chromium.org/driver/
           或国内镜像：https://npm.taobao.org/mirrors/chromedriver/
        3. 解压后将chromedriver.exe放在以下任一位置：
           - C:/chromedriver/
           - 或在.env文件中设置：CHROME_DRIVER_PATH=你的驱动完整路径
        """)
    
    def initialize_driver(self):
        """初始化浏览器驱动"""
        # 清理之前可能残留的Chrome进程
        self.cleanup_chrome_processes()
        
        # 查找Chrome浏览器路径
        chrome_path = self.find_chrome_path()
        
        # 禁用Selenium自动驱动管理
        os.environ["SE_MANAGER_DOWNLOAD"] = "0"
        
        try:
            # 构建Chrome启动命令
            chrome_profile_dir = os.path.abspath("./chrome_profile")
            os.makedirs(chrome_profile_dir, exist_ok=True)  # 确保目录存在
            
            command = [
                chrome_path,
                f"--remote-debugging-port={self.debug_port}",
                f"--user-data-dir={chrome_profile_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                "--disable-gpu",
                "--no-sandbox"
            ]
            
            # 启动Chrome
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True  # Windows下需要shell=True处理路径中的空格
            )
            
            # 等待Chrome启动并监听端口
            max_attempts = 20
            attempts = 0
            while attempts < max_attempts:
                if not self.is_port_available(self.debug_port):
                    print(f"Chrome已在端口 {self.debug_port} 启动")
                    break
                attempts += 1
                time.sleep(1)
            
            if attempts >= max_attempts:
                raise Exception(f"Chrome启动失败，无法连接到端口 {self.debug_port}")
            
            # 连接到Chrome
            chrome_options = Options()
            chrome_options.add_experimental_option("debuggerAddress", f"localhost:{self.debug_port}")
            self.driver = webdriver.Chrome(
                service=Service(self.chrome_driver_path),
                options=chrome_options
            )
        
        except Exception as e:
            raise Exception(f"无法启动或连接到Chrome浏览器: {str(e)}")
        
        self.driver.implicitly_wait(10)
        self.driver.set_window_size(1920, 1080)
    
    def find_chrome_path(self):
        """查找Chrome浏览器安装路径"""
        # 优先使用配置文件中的Chrome路径
        chrome_paths = []
        if CHROME_PATH and os.path.exists(CHROME_PATH):
            chrome_paths.append(CHROME_PATH)
        
        # 添加默认Chrome路径
        chrome_paths.extend([
            "C:/Program Files/Google/Chrome/Application/chrome.exe",
            "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
            f"C:/Users/{os.getlogin()}/AppData/Local/Google/Chrome/Application/chrome.exe",
            "/usr/bin/google-chrome",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        ])
        
        # 查找有效Chrome路径
        for path in chrome_paths:
            if os.path.exists(path) and os.path.isfile(path):
                return path
        
        raise Exception("""
        未找到Chrome浏览器，请按以下步骤解决：
        1. 确认已安装Chrome浏览器
        2. 在.env文件中手动指定路径：CHROME_PATH=你的Chrome安装路径
           示例：CHROME_PATH=C:/Program Files/Google/Chrome/Application/chrome.exe
        """)
    
    def cleanup_chrome_processes(self):
        """清理可能残留的Chrome进程"""
        try:
            if sys.platform.startswith('win'):
                # Windows系统
                subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elif sys.platform.startswith('linux'):
                # Linux系统
                subprocess.run(["pkill", "chrome"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elif sys.platform.startswith('darwin'):
                # macOS系统
                subprocess.run(["pkill", "Google Chrome"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)
        except Exception as e:
            print(f"清理Chrome进程时发生错误: {str(e)}")
    
    def open_url(self, url):
        """打开指定URL"""
        try:
            if not self.driver:
                self.initialize_driver()
                
            self.driver.get(url)
            self.current_url = url
            # 等待页面加载完成
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return True, f"成功打开网页: {url}"
        except Exception as e:
            return False, f"打开网页失败: {str(e)}"
    
    def take_screenshot(self, element_selector=None, max_attempts=3):
        """
        截取当前页面或指定元素的截图，增加重试机制
        element_selector: CSS选择器，如 "#order-container"
        max_attempts: 最大重试次数
        """
        # 确保截图目录存在
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        
        # 生成唯一的截图文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(SCREENSHOTS_DIR, f"screenshot_{timestamp}.png")
        
        for attempt in range(max_attempts):
            try:
                # 检查浏览器是否仍然可用
                if not self.driver:
                    return False, "浏览器未初始化或已关闭"
                
                # 尝试刷新页面（如果是第一次失败）
                if attempt > 0:
                    self.driver.refresh()
                    time.sleep(2)  # 等待页面刷新
                
                if element_selector:
                    # 截取指定元素
                    try:
                        # 先滚动到元素位置
                        element = WebDriverWait(self.driver, 15).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, element_selector))
                        )
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                        time.sleep(1)  # 等待滚动完成
                        
                        # 尝试元素截图
                        if element.screenshot(screenshot_path):
                            return True, screenshot_path
                            
                        # 元素截图失败，尝试全屏截图后裁剪
                        full_screenshot_path = screenshot_path.replace(".png", "_full.png")
                        self.driver.save_screenshot(full_screenshot_path)
                        
                        # 获取元素位置和大小
                        location = element.location
                        size = element.size
                        
                        # 处理可能的缩放问题
                        dpr = self.driver.execute_script("return window.devicePixelRatio;")
                        left = location['x'] * dpr
                        top = location['y'] * dpr
                        right = (location['x'] + size['width']) * dpr
                        bottom = (location['y'] + size['height']) * dpr
                        
                        # 裁剪图片
                        img = Image.open(full_screenshot_path)
                        img = img.crop((left, top, right, bottom))
                        img.save(screenshot_path)
                        os.remove(full_screenshot_path)  # 清理临时文件
                        return True, screenshot_path
                        
                    except NoSuchElementException:
                        return False, f"未找到选择器为 {element_selector} 的元素"
                    except TimeoutException:
                        return False, f"等待元素 {element_selector} 超时"
                    except (ElementNotVisibleException, ElementNotInteractableException):
                        return False, f"元素 {element_selector} 不可见或无法交互"
                else:
                    # 截取整个页面
                    if self.driver.save_screenshot(screenshot_path):
                        return True, screenshot_path
                    else:
                        # 尝试使用JavaScript截图作为备选方案
                        screenshot_data = self.driver.execute_script("""
                            var canvas = document.createElement('canvas');
                            var ctx = canvas.getContext('2d');
                            canvas.width = window.innerWidth;
                            canvas.height = window.innerHeight;
                            ctx.drawWindow(window, 0, 0, window.innerWidth, window.innerHeight, 'rgb(255, 255, 255)');
                            return canvas.toDataURL('image/png');
                        """)
                        
                        # 保存JavaScript返回的截图数据
                        img_data = screenshot_data.replace('data:image/png;base64,', '')
                        with open(screenshot_path, 'wb') as f:
                            f.write(base64.b64decode(img_data))
                        return True, screenshot_path
                
            except WebDriverException as e:
                if attempt < max_attempts - 1:
                    # 重试前关闭并重新初始化驱动
                    self.close_browser()
                    time.sleep(2)
                    self.initialize_driver()
                    if self.current_url:
                        self.driver.get(self.current_url)
                        time.sleep(2)
                    continue
                return False, f"截图失败 (尝试 {attempt+1}/{max_attempts}): {str(e)}"
            except Exception as e:
                return False, f"截图处理失败: {str(e)}"
        
        return False, f"已尝试 {max_attempts} 次，均无法完成截图"
    
    def close_browser(self):
        """关闭浏览器"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                print(f"关闭浏览器时发生错误: {str(e)}")
            finally:
                self.driver = None
                self.cleanup_chrome_processes()
                return True, "浏览器已关闭"
        return False, "没有运行中的浏览器"
    
    def execute_script(self, script, *args):
        """执行JavaScript代码"""
        try:
            if not self.driver:
                return False, "浏览器未初始化或已关闭"
                
            result = self.driver.execute_script(script, *args)
            return True, result
        except Exception as e:
            return False, f"执行脚本失败: {str(e)}"
    
    def get_page_title(self):
        """获取当前页面标题"""
        try:
            if not self.driver:
                return False, "浏览器未初始化或已关闭"
                
            return True, self.driver.title
        except Exception as e:
            return False, f"获取标题失败: {str(e)}"
    
    def find_element(self, selector, by=By.CSS_SELECTOR, timeout=10):
        """查找单个元素"""
        try:
            if not self.driver:
                return False, "浏览器未初始化或已关闭"
                
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return True, element
        except TimeoutException:
            return False, f"超时未找到元素: {selector}"
        except NoSuchElementException:
            return False, f"未找到元素: {selector}"
        except Exception as e:
            return False, f"查找元素失败: {str(e)}"
    
    def click_element(self, selector, by=By.CSS_SELECTOR, timeout=10):
        """点击指定元素"""
        try:
            success, element = self.find_element(selector, by, timeout)
            if not success:
                return False, element  # 传递错误信息
                
            # 确保元素可见并点击
            WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, selector))
            ).click()
            return True, f"已点击元素: {selector}"
        except Exception as e:
            return False, f"点击元素失败: {str(e)}"
    
    def input_text(self, selector, text, by=By.CSS_SELECTOR, timeout=10):
        """在输入框中输入文本"""
        try:
            success, element = self.find_element(selector, by, timeout)
            if not success:
                return False, element  # 传递错误信息
                
            # 清除现有内容并输入新文本
            element.clear()
            element.send_keys(text)
            return True, f"已在元素 {selector} 中输入文本"
        except Exception as e:
            return False, f"输入文本失败: {str(e)}"
    
    def get_element_text(self, selector, by=By.CSS_SELECTOR, timeout=10):
        """获取元素的文本内容"""
        try:
            success, element = self.find_element(selector, by, timeout)
            if not success:
                return False, element  # 传递错误信息
                
            return True, element.text
        except Exception as e:
            return False, f"获取元素文本失败: {str(e)}"
    
    def scroll_to_bottom(self):
        """滚动到页面底部"""
        try:
            if not self.driver:
                return False, "浏览器未初始化或已关闭"
                
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            return True, "已滚动到页面底部"
        except Exception as e:
            return False, f"滚动失败: {str(e)}"
    
    def wait_for_page_load(self, timeout=30):
        """等待页面完全加载"""
        try:
            if not self.driver:
                return False, "浏览器未初始化或已关闭"
                
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script('return document.readyState') == 'complete'
            )
            return True, "页面已完全加载"
        except TimeoutException:
            return False, "等待页面加载超时"
        except Exception as e:
            return False, f"等待页面加载失败: {str(e)}"
