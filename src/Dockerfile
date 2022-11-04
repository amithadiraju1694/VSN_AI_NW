# syntax=docker/dockerfile:1
FROM python:3.9.6

# Location where 'CMD' needs to run
WORKDIR /src

# Copy from source to destination
COPY requirements.txt /src/requirements.txt

# Upgrade pip , just in case
RUN pip3 install --upgrade pip

# Run this command in terminal
RUN pip3 install --no-cache-dir -r /src/requirements.txt

# Copy files from src ( first ) to destination ( second, in image)
COPY . /src

CMD ["python3" ,"-m" , "uvicorn" , "main:app" , "--reload", "--host", "0.0.0.0", "--port", "8000"]

