from celery.schedules import crontab

beat_schedule = {
    "fetch-every-30-mins": {
        "task": "fetch_and_store_task",
        "schedule": 1800,  # 30 dakika
    },
}
timezone = "Europe/Istanbul"
