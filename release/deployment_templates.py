"""
release/deployment_templates.py

Production deployment templates for Differential KV runtimes.
Includes Dockerfile and Kubernetes manifests for scalable cognitive serving.
"""

DOCKERFILE_TEMPLATE = """
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

# Start Differential KV Serving Engine
CMD ["python3", "-m", "serving.differential_kv_server", "--config", "release/production_config.json"]
"""

K8S_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: differential-kv-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: differential-kv
  template:
    metadata:
      labels:
        app: differential-kv
    spec:
      containers:
      - name: worker
        image: differential-kv-runtime:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: model-cache
        hostPath:
          path: /mnt/models
"""

def generate_templates(output_dir: str = "release/templates"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(DOCKERFILE_TEMPLATE)
    with open(os.path.join(output_dir, "k8s_deployment.yaml"), "w") as f:
        f.write(K8S_DEPLOYMENT)
    print(f"Deployment templates generated in {output_dir}")

if __name__ == "__main__":
    generate_templates()
