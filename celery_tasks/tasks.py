from celery import Celery
from rss_service import fetch_and_store

celery = Celery(
    "rss_tasks",
    broker="redis://localhost:6350/0",
    backend="redis://localhost:6350/0"
)

celery.config_from_object("celery_tasks.celery_config")

@celery.task(name="fetch_and_store_task")
def fetch_and_store_task():
    fetch_and_store()
    return "RSS data fetched successfully"

@celery.on_after_configure.connect
def setup_initial_task(sender, **kwargs):
    print("Initial fetch_and_store_task is running...")
    fetch_and_store_task.delay()
