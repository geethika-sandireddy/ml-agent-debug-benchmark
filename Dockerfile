FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# requirements.txt points pip at the CPU-only torch index to keep the image small.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "-m", "server.app"]
