# Base image with Python 3.8 and CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python 3.8 and other dependencies
RUN apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3-pip python3.8-venv && \
    apt-get install -y build-essential

# Create a symlink for python3.8 to python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Copy the ladder_v2 directory contents into /app/ladder_v2 in the container
COPY fully_implemented_tools/ladder_v2/ /app/fully_implemented_tools/ladder_v2/

# Copy common and resources directories into the container
COPY fully_implemented_tools/common/ /app/fully_implemented_tools/common/
COPY fully_implemented_tools/resources/ /app/fully_implemented_tools/resources/


# Install Python dependencies from requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r /app/fully_implemented_tools/ladder_v2/requirements-docker.txt


# Run the training script (e.g., train.py)
# CMD ["python", "train.py"]
# Alternatively, specify your command later during `docker run`

# Set the working directory in the container
WORKDIR /app/fully_implemented_tools/ladder_v2