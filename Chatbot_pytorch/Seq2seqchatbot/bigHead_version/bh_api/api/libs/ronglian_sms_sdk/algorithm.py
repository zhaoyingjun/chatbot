import hashlib
import base64

def md5(plaintext):
    """MD5算法
    Args:
        olaintext: 明文字符串
    Returns:
        32位16进制字符串
    """
    md5 = hashlib.md5()
    md5.update(bytes(plaintext, encoding='utf-8'))
    return md5.hexdigest()

def base64Encoder(plaintext):
    """Base64加密算法
    Args:
        plaintext: 明文字符串
    Returns:
        加密后字符串
    """
    return base64.b64encode(bytes(plaintext, encoding='utf-8'))
