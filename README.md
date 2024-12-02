# Slide Processing Application

This application processes whole slide images (WSI) from an S3 bucket, performs various analyses using machine learning models, and stores the results back in an S3 bucket. The application is designed to run inside a Docker container and can be deployed on AWS ECS.

## Features

- Downloads slide images from an S3 bucket.
- Processes slides using multiple machine learning models.
- Stores the results in JSON format in an S3 bucket.
- Supports GPU acceleration using NVIDIA CUDA.

## Prerequisites

- Docker
- AWS CLI
- AWS S3 bucket with slide images
- NVIDIA GPU and CUDA drivers (for GPU acceleration)


## Usage

### 1. Clone the Repository

```sh
git clone <repository-url>
cd <repository-directory>
```

### 2. Build the Docker Image
```sh
docker build -t slide-processing-app .
```

### 3. Run the Docker Container
```sh
docker run --gpus all -e INPUT_BUCKET=my-input-bucket -e OUTPUT_BUCKET=my-output-bucket -p 5000:5000 slide-processing-app
```

Replace my-input-bucket and my-output-bucket with the names of your S3 buckets.

#### Environment Variables
- `INPUT_BUCKET`: The name of the S3 bucket containing the input slide images.
- `OUTPUT_BUCKET`: The name of the S3 bucket where the results will be stored.