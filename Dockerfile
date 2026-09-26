FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless ca-certificates && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYTHONUNBUFFERED=1
ENV SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501 8080 7077
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
