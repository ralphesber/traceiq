web: gunicorn --worker-class gevent -w 1 --worker-connections 50 server:app --bind 0.0.0.0:$PORT --timeout 120 --keep-alive 130
worker: python3 worker.py
