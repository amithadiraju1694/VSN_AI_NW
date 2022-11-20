# syntax=docker/dockerfile:1
FROM python:3.9.6

# Location where 'CMD' needs to run
WORKDIR /VSN_AI_NW

# Copy from source to destination
COPY ./requirements.txt /VSN_AI_NW/
COPY ./main.py /VSN_AI_NW/

# Upgrade pip , just in case
RUN pip3 install --upgrade pip

# Run this command in terminal
RUN pip3 install --no-cache-dir -r ./requirements.txt

COPY ./src/ /VSN_AI_NW/

CMD ["python3" ,"-m" , "uvicorn" , "main:app" , "--reload", "--host", "0.0.0.0", "--port", "8000"]

