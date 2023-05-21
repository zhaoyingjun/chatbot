import logging
import random
from django_redis import get_redis_connection
from rest_framework.generics import CreateAPIView, RetrieveUpdateAPIView, get_object_or_404, \
    ListAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework.views import APIView
from rest_framework import status

from api.settings import constants
from api.utils.algorithm import reply
from mycelery.sms.tasks import send_sms
from .models import User, Chat
from .serializers import UserModelSerializer, PersonModelSerializer, ChatLogModelSerializer,  \
    ChatSerializer
from .utils import get_user_by_account


log = logging.getLogger('django')

class UserAPIView(CreateAPIView):
    """注册用户"""
    queryset = User.objects.all()
    serializer_class = UserModelSerializer

class MobileAPIView(APIView):
    """验证手机号唯一性"""
    def get(self,request,mobile):
        user = get_user_by_account(mobile)
        if user is not None:
            return Response({"message":"手机号已经被注册过！"},status=status.HTTP_400_BAD_REQUEST) # 不写则默认200
        return Response({"message":"ok"})

class SMSAPIView(APIView):
    """
    发送短信
    """
    def get(self, request, mobile):
        """短信发送"""
        # 1. 判断手机号码是否在60秒内曾经发送过短信
        redis_conn = get_redis_connection("sms_code")
        ret = redis_conn.get("mobile_%s" % mobile)
        if ret is not None:
            return Response({"message": "对不起，60秒内已经发送过短信，请耐心等待"}, status=status.HTTP_400_BAD_REQUEST)

        # 2. 生成短信验证码
        sms_code = "%06d" % random.randint(1, 999999)

        # 3. 保存短信验证码到redis[使用事务把多条命令集中发送给redis]
        pipe = redis_conn.pipeline()
        pipe.multi()
        pipe.setex("sms_%s" % mobile, constants.SMS_EXPIRE_TIME, sms_code)
        pipe.setex("mobile_%s" % mobile, constants.SMS_INTERVAL_TIME, "_")
        pipe.execute()

        # 4. 调用sdk，发送短信
        try:
            send_sms.delay(mobile, sms_code)
        except:
             return Response({'message': '短信发送失败！'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 5. 响应发送短信的结果
        return Response({"result": "短信发送成功！"})

class PersonAPIView(RetrieveUpdateAPIView):
    """更新、获得用户个性化信息：昵称、头像。"""
    permissions_classes = [IsAuthenticated]
    serializer_class = PersonModelSerializer

    def get_object(self):
        username = self.kwargs.get('username')
        return get_object_or_404(User, username=username)

class ChatLogAPIView(ListAPIView):
    """查询所有聊天记录"""
    permissions_classes = [IsAuthenticated]
    serializer_class = ChatLogModelSerializer
    def get_queryset(self):
        username = self.kwargs.get('username')
        res = Chat.objects.filter(user=username).order_by('-created_time')
        return res

class DayChatLogAPIView(ListAPIView):
    '''查询单日所有聊天记录'''
    permissions_classes = [IsAuthenticated]
    serializer_class = ChatLogModelSerializer
    def get_queryset(self):
        username = self.kwargs.get('username')
        day = self.kwargs.get('created_day')
        res = Chat.objects.filter(user=username, created_day=day).order_by('created_time')
        return res

class AnonymousChatAPIView(APIView):
    '''游客每天可以聊天3次，不记录在db'''
    throttle_classes = [AnonRateThrottle]
    def post(self, request):
        answer = {"answer":reply(request.data.get('question'))}
        return Response(answer, status=status.HTTP_200_OK)

class ChatAPIView(CreateAPIView):
    """用户提问，并保存聊天记录"""
    permissions_classes = [IsAuthenticated]
    serializer_class = ChatSerializer
    queryset = Chat.objects.all()



