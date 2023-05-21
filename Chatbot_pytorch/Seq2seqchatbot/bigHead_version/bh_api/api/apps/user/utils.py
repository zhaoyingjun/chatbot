from .models import User
from django.db.models import Q  # 作多条件
from django.contrib.auth.backends import ModelBackend

def jwt_response_payload_handler(token, user=None, request=None):
    """
    自定义jwt认证成功，返回数据
    :param token 本次登录成功以后，返回的jwt
    :param user 本次登录成功以后，从db中从查询到的用户模型信息
    :param request 本次客户端的请求对象
    """
    return {
        'token': token,
        'id': user.id,
        'username': user.username
    }

def get_user_by_account(account):
    """
    根据帐号获取user对象
    :param account: 账号，可以是用户名username，也可以是手机号mobile，也可以是其他data
    :return: User对象 或者 None
    """
    try:
        user = User.objects.filter( Q(username = account) | Q(mobile = account)).first()
    except User.DoesNotExist:
        return None
    else:
        return user

class UsernameMobileAuthBackend(ModelBackend):
    """
    自定义用户名/手机号认证
1. 根据username参数，查找用户User对象，username参数可能是用户名，也可能是手机号
2. 若查找到User对象，调用User对象的check_password方法检查密码是否正确
    """
    def authenticate(self, request, username=None, password=None, **kwargs):
        user = get_user_by_account(username)
        if user is not None and user.check_password(password):
            return user
        else:
            return None