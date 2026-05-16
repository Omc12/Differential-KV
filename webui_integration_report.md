# Open WebUI Integration
## Connecting Differential KV to Open WebUI

### Overview
Differential KV exposes a true OpenAI-compatible API that is fully compliant with Open WebUI's requirements for external connections.

### Connection Steps
1. **Launch Differential KV Server**:
   Ensure `python launch_real_serving.py` is running and bound to `0.0.0.0:8000`.
2. **Start Open WebUI**:
   ```bash
   docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
   ```
3. **Configure API Source**:
   - Navigate to Open WebUI **Settings > Connections**
   - Add a new OpenAI API connection.
   - Set the Base URL to: `http://host.docker.internal:8000/v1`
   - Use any dummy string for the API Key.

### Expected Behavior
- **Model Discovery**: Open WebUI will fetch available models via the `/v1/models` endpoint (if extended) or the default configured string.
- **Streaming Chats**: As you type queries, the Differential KV backend will resolve the context, and stream the generation directly back to the WebUI interface smoothly.
- **Concurrency**: Multiple browser tabs can be handled seamlessly by the `SparseRequestScheduler` operating in the backend.
