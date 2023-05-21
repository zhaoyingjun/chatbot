import re
from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from rest_framework_jwt.settings import api_settings
from django_redis import get_redis_connection

from bh_api.api.utils.algorithm import reply
from .models import User, Chat
from .utils import get_user_by_account

class UserModelSerializer(serializers.ModelSerializer):
    sms_code = serializers.CharField(min_length=4, max_length=6, required=True, write_only=True, help_text="短信验证码")
    token = serializers.CharField(max_length=1024, read_only=True, help_text="token认证字符串")

    class Meta:
        model = User
        fields = ["id", "username", "mobile", "password", "sms_code", "token"]
        extra_kwargs = {
            "id": {
                "read_only": True,
            },
            "username": {
                "read_only": True,
            },
            "password": {
                "write_only": True,
            },
            "mobile": {
                "write_only": True,
            }
        }

    def validate(self, attrs):
        mobile = attrs.get("mobile")
        sms_code = attrs.get("sms_code")
        password = attrs.get("password")

        if not re.match(r"^1[3-9]\d{9}$", mobile):
            raise serializers.ValidationError("对不起，手机号格式有误！")

        ret = get_user_by_account(mobile)
        if ret is not None:
            raise serializers.ValidationError("对不起，手机号已经被注册过！")

        redis_conn = get_redis_connection("sms_code")
        real_sms_code = redis_conn.get("sms_%s" % mobile)
        redis_conn.delete("sms_%s" % mobile)

        if real_sms_code.decode() != sms_code:
            raise serializers.ValidationError("对不起，短信验证码错误！本次验证码已失效，请重新发送！")

        return attrs

    def create(self, validated_data):
        validated_data.pop("sms_code")
        raw_password = validated_data.get("password")
        hash_password = make_password(raw_password)
        username = validated_data.get("mobile")
        user = User.objects.create(
            mobile=username,
            username=username,
            password=hash_password,
        )

        jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER

        payload = jwt_payload_handler(user)
        user.token = jwt_encode_handler(payload)
        return user


class PersonModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['avatar', 'nickname']


class ChatLogModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = "__all__"

class ChatSerializer(serializers.Serializer):
    answer = serializers.CharField(read_only=True)
    question = serializers.CharField(write_only=True)
    user = serializers.CharField(write_only=True)

    def create(self, validated_data):
         answer = reply(validated_data.get("question"))
         validated_data['answer'] = answer
         user_row = User.objects.get(username=validated_data.get('user'))
         validated_data['user'] = user_row
         obj = Chat.objects.create(**validated_data)
         return obj

