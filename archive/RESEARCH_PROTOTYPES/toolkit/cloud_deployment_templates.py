import logging

logger = logging.getLogger(__name__)

def generate_k8s_templates():
    """
    Outputs standard Kubernetes YAML manifests for deploying
    Differential KV at cloud scale.
    """
    logger.info("Generating Kubernetes Templates...")
    
    k8s_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cognition-cloud-node
spec:
  replicas: 10
  selector:
    matchLabels:
      app: diff-kv-runtime
  template:
    metadata:
      labels:
        app: diff-kv-runtime
    spec:
      containers:
      - name: diff-kv
        image: diff_kv/cloud_runtime:v1
        resources:
          limits:
            nvidia.com/gpu: 1
"""
    with open("k8s_deployment.yaml", "w") as f:
        f.write(k8s_deployment)
        
    logger.info("k8s_deployment.yaml generated.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_k8s_templates()
