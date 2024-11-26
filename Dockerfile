# Use the official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
# Set the working directory in the container
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    openjdk-11-jdk \
    libvips-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python3", "__init__.py"]