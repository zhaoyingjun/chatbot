from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """用户模型类"""
    mobile = models.CharField(max_length=11, unique=True, verbose_name='手机号')
    nickname = models.CharField(max_length=16, verbose_name='昵称', default='你')
    avatar = models.ImageField(upload_to='avatar',default='avatar/default.png', verbose_name='头像')
    class Meta:
        db_table = 'bh_users'
        verbose_name = '用户信息'
        verbose_name_plural = verbose_name

class Chat(models.Model):
    """聊天记录类"""
    question = models.CharField(verbose_name="提问", max_length=1024)
    answer = models.CharField(verbose_name="回答", max_length=1024)
    created_time = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    created_day = models.DateField(verbose_name="创建日", auto_now_add=True)
    user = models.ForeignKey(to="User",to_field='username',on_delete=models.CASCADE, related_name='user_chats')

    class Meta:
        db_table = 'bh_chats'
        verbose_name = '聊天记录'
        verbose_name_plural = verbose_name



