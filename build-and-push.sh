# Build
docker build -t lambda-nost-ar .

# Tag the image to match the repository name
docker tag lambda-nost-ar:latest 430946944860.dkr.ecr.us-west-2.amazonaws.com/lambda-nost-ar:latest

# Register docker to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 430946944860.dkr.ecr.us-west-2.amazonaws.com

# Push the image to ECR
docker push 430946944860.dkr.ecr.us-west-2.amazonaws.com/lambda-nost-ar:latest
