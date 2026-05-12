from fastapi import FastAPI, UploadFile, File, HTTPException
import logging

app = FastAPI(title="Manifold Exchange API")
logger = logging.getLogger(__name__)

# Mock database of manifolds
manifolds_db = {}

@app.post("/exchange/upload/{manifold_id}")
async def upload_manifold(manifold_id: str, file: UploadFile = File(...)):
    """Uploads a compressed manifold representation to the central exchange."""
    try:
        content = await file.read()
        manifolds_db[manifold_id] = content
        logger.info(f"Manifold {manifold_id} uploaded successfully. Size: {len(content)} bytes")
        return {"status": "success", "manifold_id": manifold_id, "size": len(content)}
    except Exception as e:
        logger.error(f"Failed to upload manifold: {e}")
        raise HTTPException(status_code=500, detail="Internal upload error")

@app.get("/exchange/download/{manifold_id}")
async def download_manifold(manifold_id: str):
    """Downloads a requested manifold representation."""
    if manifold_id not in manifolds_db:
        raise HTTPException(status_code=404, detail="Manifold not found on exchange")
        
    return {
        "manifold_id": manifold_id,
        "data_length": len(manifolds_db[manifold_id]),
        "message": "Manifold payload attached (mock)"
    }
