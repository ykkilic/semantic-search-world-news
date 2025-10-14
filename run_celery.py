# import subprocess
# import sys

# # Worker komutu
# worker_cmd = [
#     sys.executable, "-m", "celery",
#     "-A", "celery_tasks.tasks",
#     "worker",
#     "--loglevel=info"
# ]

# # Beat komutu
# beat_cmd = [
#     sys.executable, "-m", "celery",
#     "-A", "celery_tasks.tasks",
#     "beat",
#     "--loglevel=info"
# ]

# # Worker'ı arka planda başlat
# worker_proc = subprocess.Popen(worker_cmd)

# try:
#     # Beat'i ön planda çalıştır
#     subprocess.run(beat_cmd)
# finally:
#     # Ctrl+C ile durdurulduğunda worker'ı kapat
#     worker_proc.terminate()
#     worker_proc.wait()


import subprocess
import sys
import signal

# Worker komutu (macOS için solo pool)
worker_cmd = [
    sys.executable, "-m", "celery",
    "-A", "celery_tasks.tasks",
    "worker",
    "--loglevel=info",
    "--pool=solo"
]

# Beat komutu
beat_cmd = [
    sys.executable, "-m", "celery",
    "-A", "celery_tasks.tasks",
    "beat",
    "--loglevel=info"
]

# Worker'ı arka planda başlat
worker_proc = subprocess.Popen(worker_cmd)

def shutdown(signal_received, frame):
    print("SIGINT/SIGTERM alındı, worker kapatılıyor...")
    worker_proc.terminate()
    worker_proc.wait()
    sys.exit(0)

# Ctrl+C veya kill ile durdurmayı yakala
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

try:
    # Beat'i ön planda çalıştır
    subprocess.run(beat_cmd)
finally:
    # Beat kapandığında worker'ı da kapat
    worker_proc.terminate()
    worker_proc.wait()
