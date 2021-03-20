# Pull the base image with python 3.6 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.6

# Install OS packages for Pillow-SIMD
RUN yum -y install sudo gcc gcc-c++

# Copy the earlier created requirements.txt file to the container
COPY gpt-2/requirements.txt ./

RUN pip3 install --upgrade pip
RUN python3 -m pip install --upgrade setuptools
RUN pip3 install --no-cache-dir  --force-reinstall -Iv grpcio

# Install the python requirements from requirements.txt
RUN python3.6 -m pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./

# Model files
ADD models /models
ADD selector /selector
ADD sentiment /sentiment
ADD draft_autoreviewer /draft_autoreviewer

COPY gpt-2 gpt-2/
COPY ./*.py ./
COPY config_remote.json ./config.json

RUN pip3 install pytumblr

# Make sure app.py is up to date
RUN rm ./app.py
COPY app_latest.py ./app.py

# Set the CMD to your handler
CMD ["app.lambda_handler"]
