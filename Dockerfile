# syntax=docker/dockerfile:1
FROM python:3.9.7-slim

# Location where 'CMD' needs to run in docker, can provide new directory name
WORKDIR /VSN_AI_NW

# Copy Requirements file first from source to destination
COPY ./requirements.txt /VSN_AI_NW/requirements.txt

# Upgrade pip , just in case
RUN pip3 install --upgrade pip

# Run this command in terminal
RUN pip3 install --no-cache-dir -r /VSN_AI_NW/requirements.txt

# Copy everything else from source to destination
COPY ./vsn_nw/ /VSN_AI_NW/vsn_nw

CMD ["python3" ,"-m" , "uvicorn" , "vsn_nw.main:app" , "--reload", "--host", "0.0.0.0", "--port", "8000"]

