from django.urls import path
from . import views
urlpatterns = [
    path(r"nav/",views.NavListAPIView.as_view()),
]