# Hugging Face Spaces용 Dockerfile (WhatBreath Flask 서버)
FROM python:3.10-slim

WORKDIR /app

# librosa가 오디오(webm 등)를 읽으려면 ffmpeg이 필요해요
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 나머지 파일 복사 (best_model.pth, .env는 .dockerignore로 별도 관리)
COPY . .

# Hugging Face Spaces는 7860 포트를 기본으로 사용해요
EXPOSE 7860

# Flask 앱을 gunicorn으로 실행 (app.py 안의 app 객체를 가리킴)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]