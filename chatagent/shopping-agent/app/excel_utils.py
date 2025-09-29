# Developer: Enjoy
# Function: shop agent
# Creation Date: 2025-09-21
# Version: 1.0
import pandas as pd
import os
from datetime import datetime
from .config import EXCELS_DIR

class ExcelUtils:
    def __init__(self):
        self.excels_dir = EXCELS_DIR
    
    def save_order_to_excel(self, order_data):
        """
        将订单数据保存到Excel文件
        order_data: 从OCR获取的订单数据字典
        """
        try:
            # 生成唯一的Excel文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            order_id = order_data.get('订单号', f'order_{timestamp}')
            excel_filename = f"order_{order_id}_{timestamp}.xlsx"
            excel_path = os.path.join(self.excels_dir, excel_filename)
            
            # 创建一个ExcelWriter对象
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 订单基本信息
                order_info = {
                    '订单号': [order_data.get('订单号')],
                    '订单日期': [order_data.get('订单日期')],
                    '总金额': [order_data.get('总金额')],
                    '支付方式': [order_data.get('支付方式')],
                    '收货人': [order_data.get('收货人')],
                    '联系电话': [order_data.get('联系电话')],
                    '收货地址': [order_data.get('收货地址')]
                }
                pd.DataFrame(order_info).to_excel(writer, sheet_name='订单信息', index=False)
                
                # 商品列表
                products = order_data.get('商品名称列表', [])
                if not isinstance(products, list):
                    products = []
                
                pd.DataFrame(products).to_excel(writer, sheet_name='商品明细', index=False)
            
            return True, excel_path
            
        except Exception as e:
            return False, f"保存Excel失败: {str(e)}"
    
    def get_excel_files(self):
        """获取所有保存的Excel文件列表"""
        try:
            excel_files = [f for f in os.listdir(self.excels_dir) if f.endswith('.xlsx')]
            # 按修改时间排序，最新的在前
            excel_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.excels_dir, x)), reverse=True)
            return True, excel_files
        except Exception as e:
            return False, f"获取Excel文件列表失败: {str(e)}"
    
    def read_excel_file(self, filename):
        """读取Excel文件内容"""
        try:
            excel_path = os.path.join(self.excels_dir, filename)
            if not os.path.exists(excel_path):
                return False, "文件不存在"
                
            # 读取订单信息
            order_info = pd.read_excel(excel_path, sheet_name='订单信息').to_dict('records')
            # 读取商品明细
            products = pd.read_excel(excel_path, sheet_name='商品明细').to_dict('records')
            
            return True, {
                'order_info': order_info[0] if order_info else {},
                'products': products
            }
        except Exception as e:
            return False, f"读取Excel文件失败: {str(e)}"
