FROM python:3.14-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

RUN sed -i 's/\r$//' src/run.sh

RUN chmod +x src/run.sh

CMD ["bash", "./src/run.sh"]
