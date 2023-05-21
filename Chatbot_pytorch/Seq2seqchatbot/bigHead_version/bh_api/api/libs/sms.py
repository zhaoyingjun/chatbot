from django.conf import settings
import json

from api.libs.ronglian_sms_sdk import SmsSDK

accId = settings.SMS_ACCOUNTSID
accToken = settings.SMS_ACCOUNTTOKEN
appId = settings.SMS_APPID

class SMS(object):
    """发送短信的辅助类"""
    def __new__(cls, *args, **kwargs):
        if not hasattr(SMS, "_instance"):
            cls._instance = super(SMS, cls).__new__(cls, *args, **kwargs)
            cls._instance.sms = SmsSDK(accId, accToken, appId)
        return cls._instance

    def send_template_sms(self, tid, mobile, datas):
        """发送模板短信"""
        # @param to 手机号码
        # @param datas 内容数据 格式为数组 例如：{'12','34'}，如不需替换请填 ''
        # @param temp_id 模板Id
        res_str  = self.sms.sendMessage(tid, mobile, datas)
        res_dict = json.loads(res_str)
        return res_dict.get('statusCode') == '000000'

if __name__ == '__main__':
    sms = SMS()
    # 以下手机号为虚拟号，可替换成自己的
    result = sms.send_template_sms(1,'18000000000', ('1234', 5))
    print(result)