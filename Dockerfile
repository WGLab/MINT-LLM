FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y git wget gcc build-essential libgl1-mesa-glx

# Create working directory
WORKDIR /workspace/RareDxGPT

# Copy requirements
COPY requirements.txt /workspace/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r /workspace/requirements.txt

# Optional: If you want to copy the entire codebase into the container
# COPY . /workspace/RareDxGPT

# Set environment variables for distributed training
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

CMD [ "bash" ]
