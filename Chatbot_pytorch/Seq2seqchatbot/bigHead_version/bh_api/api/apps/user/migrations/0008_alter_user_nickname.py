# Generated by Django 3.2.4 on 2023-02-21 15:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0007_alter_user_nickname'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='nickname',
            field=models.CharField(default='你', max_length=16, verbose_name='昵称'),
        ),
    ]
