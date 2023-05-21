from rest_framework.views import exception_handler
from django.db import DatabaseError
from rest_framework.response import Response
from rest_framework import status

import logging
logger = logging.getLogger('django')

def custom_exception_handler(exc, context):
    """
    自定义异常处理 同exception_handler(exc, context)
    :param exc: 异常类
    :param context: 抛出异常的上下文（本次请求的request对象+异常发送事件+行号等）
    :return: Response响应对象
    """
    response = exception_handler(exc, context)

    if response is None:
        view = context['view']
        if isinstance(exc, DatabaseError):
            logger.error('[%s] %s' % (view, exc))
            response = Response({'message': '服务器内部错误，请联系客户工作人员'}, status=status.HTTP_507_INSUFFICIENT_STORAGE)

    return response