from celery import Celery
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api.settings.dev')

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'api\\apps'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

app = Celery("bh")
app.config_from_object("mycelery.config")
app.autodiscover_tasks(["mycelery.sms","mycelery.cache"])

# 通过终端来启动celery
# celery -A mycelery.main worker --loglevel=info
# celery -A mycelery.main worker -l info -P eventlet