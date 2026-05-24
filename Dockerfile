FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY . .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
-r requirements.txt

EXPOSE 7861

CMD ["python","app.py"]
