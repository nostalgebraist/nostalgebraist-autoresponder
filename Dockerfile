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

# Model files
ADD models /models
ADD selector /selector
ADD sentiment /sentiment
ADD draft_autoreviewer /draft_autoreviewer

COPY gpt-2 gpt-2/
COPY ./*.py ./
COPY experimental experimental/
COPY selector_model selector_model/
COPY util util/
COPY config_lambda.json ./config_lambda.json

# Make sure app.py is up to date
COPY app_latest.py ./app.py

# Set the CMD to your handler
CMD ["app.lambda_handler"]
