# OpenAI API Compatibility
## Validation and Conformance

### Exposed Endpoints
1. `POST /v1/chat/completions`
2. `GET /v1/sessions`

### Compatibility Checks
- **Streaming Handshake**: Implemented and verified via `stream=True`. Successfully transmitted `chat.completion.chunk` SSE payloads.
- **Message Parsing**: Successfully translates `[{"role": "system", ...}, {"role": "user", ...}]` structures to raw conversational prompts for the backend.
- **Stop Conditions**: Transmitted `finish_reason: stop` and `[DONE]` at the end of the streaming payload.
- **Usage Metrics**: Accurately reported `prompt_tokens`, `completion_tokens`, and `total_tokens`.

### Tooling Compatibility
Any system configured for OpenAI APIs (e.g. standard `openai` python library, curl, Open WebUI) can point its `base_url` to `http://localhost:8000/v1` and interact natively with Differential KV.
