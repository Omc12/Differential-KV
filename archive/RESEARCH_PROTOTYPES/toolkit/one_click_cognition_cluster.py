import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def setup_cluster():
    """
    A unified entry point for spinning up a complete, local test cluster
    of cognitive runtimes using Docker Compose.
    """
    logger.info("Initializing One-Click Cognition Cluster...")
    
    compose_content = """
version: '3.8'
services:
  orchestrator:
    image: dkv/orchestrator:latest
    ports:
      - "8000:8000"
  
  edge_node_1:
    image: dkv/runtime:latest
    environment:
      - ROLE=coder
      - ORCHESTRATOR_URL=http://orchestrator:8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
        
    logger.info("docker-compose.yml generated.")
    logger.info("Run `docker-compose up -d` to launch the cluster.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_cluster()
