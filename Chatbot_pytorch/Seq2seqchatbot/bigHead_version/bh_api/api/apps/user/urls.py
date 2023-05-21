from django.urls import path, re_path
from rest_framework_jwt.views import obtain_jwt_token

from . import views

urlpatterns = [
    path(r'login/', obtain_jwt_token),
    path(r"", views.UserAPIView.as_view()),
    re_path(r"mobile/(?P<mobile>1[3-9]\d{9})/", views.MobileAPIView.as_view()),
    re_path(r'sms/(?P<mobile>1[3-9]\d{9})/', views.SMSAPIView.as_view() ),
    re_path(r'person/(?P<username>1[3-9]\d{9})/',views.PersonAPIView.as_view()),

    re_path(r'^chatlog/(?P<username>1[3-9]\d{9})/$', views.ChatLogAPIView.as_view()),
    re_path(r'chatlog/(?P<username>1[3-9]\d{9})/(?P<created_day>[1-9]\d{3}-\d{2}-\d[1-9])/',
            views.DayChatLogAPIView.as_view()),

    path(r'chat-tour/',views.AnonymousChatAPIView.as_view()),
    path(r'chat/', views.ChatAPIView.as_view()),
]
