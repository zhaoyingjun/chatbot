import xadmin
from xadmin import views

class BaseSetting(object):
    """xadmin的基本配置"""
    enable_themes = True
    use_bootswatch = True

xadmin.site.register(views.BaseAdminView, BaseSetting)

class GlobalSettings(object):
    """xadmin的全局配置"""
    site_title = "大头chatbot"
    site_footer = "天才公司"
    menu_style = "accordion"

xadmin.site.register(views.CommAdminView, GlobalSettings)

from .models import Nav
class NavModelAdmin(object):
    list_display=["title","link","is_site","is_show"]
xadmin.site.register(Nav, NavModelAdmin)