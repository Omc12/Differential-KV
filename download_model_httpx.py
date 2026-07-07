import sys
import httpx
from huggingface_hub import set_client_factory, set_async_client_factory, snapshot_download

# Disable SSL verification for httpx clients
def client_factory() -> httpx.Client:
    return httpx.Client(verify=False)

def async_client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=False)

set_client_factory(client_factory)
set_async_client_factory(async_client_factory)

# Also disable warning logs from urllib3/httpx about unverified HTTPS requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

print("Starting download of mlx-community/Llama-3.2-3B-Instruct-4bit via mirror (SSL verify disabled)...")
try:
    snapshot_download(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        endpoint="https://hf-mirror.com"
    )
    print("DOWNLOAD_SUCCESSFUL")
except Exception as e:
    print(f"DOWNLOAD_FAILED: {e}")
    sys.exit(1)
