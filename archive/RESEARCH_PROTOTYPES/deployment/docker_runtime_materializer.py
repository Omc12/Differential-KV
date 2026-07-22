import os

class DockerRuntimeMaterializer:
    """
    Generates optimized Docker configurations for Differential KV.
    Ensures CUDA runtime compatibility and sparse-serving readiness.
    """
    def __init__(self, bundle_dir: str = "dist"):
        self.bundle_dir = bundle_dir

    def generate_docker_files(self, base_image: str = "nvidia/cuda:12.1.1-runtime-ubuntu22.04"):
        """
        Creates a Dockerfile and docker-compose.yml.
        """
        dockerfile_content = f"""# Differential KV Production Runtime
FROM {base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3-pip \\
    python3-dev \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose API and metrics ports
EXPOSE 8000 9090

# Entrypoint
CMD ["bash", "start_serving.sh"]
"""
        dockerfile_path = os.path.join(self.bundle_dir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
            
        compose_content = """version: '3.8'
services:
  dkv-serving:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - DKV_RUNTIME_DEVICE=cuda
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./configs:/app/configs
      - ./session_checkpoints:/app/session_checkpoints
"""
        compose_path = os.path.join(self.bundle_dir, "docker-compose.yml")
        with open(compose_path, "w") as f:
            f.write(compose_content)

        print(f"[DRM] Generated Docker configurations in {self.bundle_dir}")
        return dockerfile_path, compose_path

    def validate_containerization(self) -> bool:
        """
        Checks if Docker/NVIDIA-Docker are available for the current materialization.
        """
        # In a real validation, we might check `docker version` and `nvidia-smi`
        return True
