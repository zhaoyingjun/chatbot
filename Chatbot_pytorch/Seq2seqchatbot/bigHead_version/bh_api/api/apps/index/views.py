from rest_framework.generics import ListAPIView

from api.settings import constants
from .models import Nav
from .serializers import NavModelSerializer

class NavListAPIView(ListAPIView):
    """导航菜单视图"""
    queryset = Nav.objects.filter(is_show=True, is_deleted=False).order_by("orders","-id")[:constants.NAV_LENGTH]
    serializer_class = NavModelSerializer