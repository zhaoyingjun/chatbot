from django.db import models

class Nav(models.Model):
    is_show = models.BooleanField(default=False, verbose_name="是否显示")
    orders = models.IntegerField(default=1, verbose_name="排序")
    is_deleted = models.BooleanField(default=False, verbose_name="是否删除")

    title = models.CharField(max_length=500, verbose_name="导航标题")
    link = models.CharField(max_length=500, verbose_name="导航链接")
    # 默认站内跳转
    is_site = models.BooleanField(default=False, verbose_name="是否是站外地址")

    class Meta:
        db_table = 'bh_nav'
        verbose_name = '导航菜单'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.title
