from api.libs.sms import SMS
from api.settings import constants
from mycelery.main import app

import logging
log = logging.getLogger("django")

@app.task(name="send_sms")
def send_sms(mobile, sms_code):
    '''发送短信'''
    sms = SMS()
    res = sms.send_template_sms(constants.SMS_TEMPLATE_ID, mobile, (sms_code, constants.SMS_EXPIRE_TIME // 60), )
    if not res:
        log.error('用户注册短信发送失败！手机号：%s' % mobile)