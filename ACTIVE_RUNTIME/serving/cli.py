#!/usr/bin/env python3
import os
import sys
import time
import uuid
import asyncio
import argparse
import httpx
import torch

# Configure environment defaults for macOS MPS if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Add parent directory to sys.path to resolve imports correctly
_runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

# ---------------------------------------------------------------------------
# ANSI Color Codes for Styling
# ---------------------------------------------------------------------------
COLOR_RESET = "\033[0m"
COLOR_USER = "\033[92m"       # Green
COLOR_AI = "\033[96m"         # Cyan
COLOR_SYSTEM = "\033[95m"     # Magenta
COLOR_WARNING = "\033[93m"    # Yellow
COLOR_ERROR = "\033[91m"      # Red
COLOR_DIM = "\033[90m"        # Dark Gray
COLOR_BOLD = "\033[1m"        # Bold

def print_system(msg):
    print(f"{COLOR_SYSTEM}{COLOR_BOLD}[System]{COLOR_RESET} {COLOR_SYSTEM}{msg}{COLOR_RESET}")

def print_warning(msg):
    print(f"{COLOR_WARNING}{COLOR_BOLD}[Warning]{COLOR_RESET} {COLOR_WARNING}{msg}{COLOR_RESET}")

def print_error(msg):
    print(f"{COLOR_ERROR}{COLOR_BOLD}[Error]{COLOR_RESET} {COLOR_ERROR}{msg}{COLOR_RESET}")

def print_metrics(msg):
    print(f"{COLOR_DIM}{msg}{COLOR_RESET}")

# ---------------------------------------------------------------------------
# Helper function for non-blocking input
# ---------------------------------------------------------------------------
async def async_input(prompt: str) -> str:
    import select
    def _read():
        sys.stdout.write(prompt)
        sys.stdout.flush()
        
        fd = sys.stdin.fileno()
        is_tty = sys.stdin.isatty()
        
        old_settings = None
        if is_tty:
            try:
                import termios
                old_settings = termios.tcgetattr(fd)
                new_settings = termios.tcgetattr(fd)
                # Disable IEXTEN to prevent Ctrl-O and Ctrl-R interception during copy-paste
                new_settings[3] &= ~termios.IEXTEN
                termios.tcsetattr(fd, termios.TCSANOW, new_settings)
            except Exception:
                pass
                
        try:
            # Wait for standard input to be ready
            try:
                r, _, _ = select.select([fd], [], [])
                if not r:
                    raise EOFError()
            except (AttributeError, OSError, select.error):
                # Fallback if select is not supported on stdin (e.g. Windows)
                line = sys.stdin.readline()
                if not line:
                    raise EOFError()
                return line
            
            # Read first chunk
            try:
                chunk = os.read(fd, 8192)
                if not chunk:
                    raise EOFError()
            except OSError:
                raise EOFError()
                
            chunks = [chunk]
            # Accumulate any immediately available chunks (pasted text)
            while True:
                try:
                    r, _, _ = select.select([fd], [], [], 0.01)
                    if r:
                        next_chunk = os.read(fd, 8192)
                        if not next_chunk:
                            break
                        chunks.append(next_chunk)
                    else:
                        break
                except (AttributeError, OSError, select.error):
                    break
                    
            # Decode and clean control characters
            text = b"".join(chunks).decode("utf-8", errors="replace")
            cleaned_chars = []
            for char in text:
                o = ord(char)
                if 32 <= o <= 126 or char in ('\n', '\r', '\t'):
                    cleaned_chars.append(char)
            return "".join(cleaned_chars)
            
        finally:
            if old_settings is not None:
                try:
                    import termios
                    termios.tcsetattr(fd, termios.TCSANOW, old_settings)
                except Exception:
                    pass

    try:
        raw_input = await asyncio.to_thread(_read)
        return raw_input.rstrip("\r\n")
    except EOFError:
        raise

# ---------------------------------------------------------------------------
# Client Mode Helper (talks to running OpenAI compatible API gateway)
# ---------------------------------------------------------------------------
async def run_client_mode(args):
    api_url = args.api_url.rstrip('/')
    session_id = f"cli-session-{uuid.uuid4().hex[:8]}"
    messages = []
    
    print_system(f"Connecting to DiffKV Gateway at {COLOR_BOLD}{api_url}{COLOR_RESET}...")
    print_system(f"Session ID: {COLOR_BOLD}{session_id}{COLOR_RESET}")
    print_system("Type /help to see all available commands.")
    print("-" * 80)
    
    while True:
        try:
            user_prompt = await async_input(f"\n{COLOR_USER}{COLOR_BOLD}User >{COLOR_RESET} ")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        user_prompt_stripped = user_prompt.strip()
        if not user_prompt_stripped:
            continue

        # Handle Slash Commands (only if single-line to avoid misinterpreting multi-line pasted text starting with /)
        if user_prompt_stripped.startswith("/") and "\n" not in user_prompt_stripped:
            cmd = user_prompt_stripped.split()[0].lower()
            if cmd in ["/exit", "/quit"]:
                print_system("Exiting client.")
                break
            elif cmd in ["/reset", "/new"]:
                messages = []
                # Notify server to delete session
                try:
                    async with httpx.AsyncClient() as client:
                        await client.delete(f"{api_url}/sessions/{session_id}")
                except Exception:
                    pass
                session_id = f"cli-session-{uuid.uuid4().hex[:8]}"
                print_system(f"Conversation reset. New Session ID: {COLOR_BOLD}{session_id}{COLOR_RESET}")
                continue
            elif cmd in ["/paste", "/multiline"]:
                print_system("Entering multiline paste mode. Paste your text below, then type 'EOF' or '///' on a new line to submit. Type '/cancel' to abort.")
                lines = []
                while True:
                    try:
                        line = await async_input("... ")
                    except (KeyboardInterrupt, EOFError):
                        print()
                        lines = []
                        break
                    stripped_line = line.strip()
                    if stripped_line == "/cancel":
                        print_system("Multiline input cancelled.")
                        lines = []
                        break
                    if stripped_line in ("EOF", "///"):
                        break
                    lines.append(line)
                if not lines:
                    continue
                user_prompt_stripped = "\n".join(lines).strip()
            elif cmd == "/help":
                print_system("Available commands:")
                print("  /reset, /new       : Clear chat history and start a new session.")
                print("  /paste, /multiline : Enter multiline mode for pasting papers or long text.")
                print("  /stats             : Fetch live runtime & memory stats from the server.")
                print("  /srl               : Fetch Semantic Routing Layer (SRL) details for current session.")
                print("  /exit, /quit       : Close the client.")
                continue
            elif cmd == "/stats":
                print_system("Fetching metrics from server...")
                try:
                    async with httpx.AsyncClient() as client:
                        r = await client.get(f"{api_url}/runtime_info")
                        if r.status_code == 200:
                            data = r.json()
                            print(f"\n{COLOR_BOLD}=== SERVER RUNTIME METRICS ==={COLOR_RESET}")
                            print(f"Model:           {data.get('model', 'N/A')}")
                            print(f"Device:          {data.get('device', 'N/A')}")
                            print(f"Serving Mode:    {data.get('serving_mode', 'N/A')}")
                            print(f"VRAM Allocated:  {data.get('vram_allocated_gb', 0.0):.3f} GB")
                            print(f"Process RSS:     {data.get('process_rss_gb', 0.0):.3f} GB")
                            
                            kv = data.get("kv_summary", {})
                            if kv:
                                print(f"Active Sessions: {kv.get('sessions', 0)}")
                                print(f"VRAM Saved:      {kv.get('vram_saved_mb', 0.0):.2f} MB")
                                print(f"Compressions:    {kv.get('total_compressions', 0)}")
                                print(f"Avg Cosine Sim:  {kv.get('avg_cosine_sim', 0.0):.4f}")
                                print(f"Avg Norm Drift:  {kv.get('avg_norm_drift', 0.0):.4f}")
                                
                                pager = kv.get("pager", {})
                                if pager:
                                    print(f"Pager Resident:  {pager.get('gpu_resident_mb', 0.0):.2f} MB")
                                    print(f"Total Evictions: {pager.get('total_evictions', 0)}")
                                    print(f"Total Reloads:   {pager.get('total_reloads', 0)}")
                            print("==============================\n")
                        else:
                            print_error(f"Failed to fetch stats: Server returned {r.status_code}")
                except Exception as e:
                    print_error(f"Failed to connect to gateway: {e}")
                continue
            elif cmd == "/srl":
                print_system(f"Fetching SRL statistics for session {session_id}...")
                try:
                    async with httpx.AsyncClient() as client:
                        r = await client.get(f"{api_url}/sessions/{session_id}/srl")
                        if r.status_code == 200:
                            data = r.json()
                            print(f"\n{COLOR_BOLD}=== SRL SESSION STATS ==={COLOR_RESET}")
                            print(f"Session ID:   {data.get('session_id')}")
                            print(f"SRL Enabled:  {data.get('srl_enabled')}")
                            print(f"SRL Built:    {data.get('srl_built')}")
                            if not data.get('srl_built'):
                                print(f"Reason:       {data.get('reason')}")
                            else:
                                print(f"K-Min/K-Max:  {data.get('k_min')} / {data.get('k_max')}")
                                print(f"Threshold:    {data.get('routing_threshold')}")
                                print(f"Cache Size:   {data.get('cache_size_tokens')} tokens")
                            print("=========================\n")
                        else:
                            print_error(f"Failed to fetch SRL info: Server returned {r.status_code}")
                except Exception as e:
                    print_error(f"Failed to connect to gateway: {e}")
                continue
            else:
                print_warning(f"Unknown command: {cmd}. Type /help for assistance.")
                continue

        # Add to history
        messages.append({"role": "user", "content": user_prompt_stripped})
        
        # Prepare streaming request to Gateway
        payload = {
            "model": args.model,
            "messages": messages,
            "stream": True,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "session_id": session_id
        }
        
        print(f"\n{COLOR_AI}{COLOR_BOLD}AI >{COLOR_RESET} ", end="", flush=True)
        
        start_time = time.time()
        first_token_time = None
        response_chunks = []
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", f"{api_url}/chat/completions", json=payload) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        print()
                        print_error(f"Server returned code {r.status_code}: {body.decode()}")
                        messages.pop()  # Remove last message on failure
                        continue
                        
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            data_str = line[len("data: "):].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json_loads(data_str)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    print(content, end="", flush=True)
                                    response_chunks.append(content)
                            except Exception:
                                pass
                                
            print() # end line
            
            # Print performance metrics
            end_time = time.time()
            if response_chunks:
                assistant_response = "".join(response_chunks)
                messages.append({"role": "assistant", "content": assistant_response})
                
                # Estimate tokens (approx 4 chars = 1 token)
                est_tokens = max(1, sum(len(c) for c in response_chunks) // 4)
                total_duration = end_time - start_time
                
                ttft_str = f"{(first_token_time - start_time) * 1000:.1f}ms" if first_token_time else "N/A"
                if first_token_time and end_time > first_token_time:
                    gen_duration = end_time - first_token_time
                    speed = est_tokens / gen_duration
                    speed_str = f"{speed:.1f} tok/s"
                else:
                    speed_str = "N/A"
                    
                print_metrics(f"[Metrics] TTFT: {ttft_str} | Speed: {speed_str} | Generated: ~{est_tokens} tokens | Duration: {total_duration:.2f}s")
            
        except Exception as e:
            print()
            print_error(f"Request failed: {e}")
            messages.pop()

def json_loads(s):
    import json
    return json.loads(s)


# ---------------------------------------------------------------------------
# Direct Mode Helper (runs batch engine directly in-process)
# ---------------------------------------------------------------------------
async def run_direct_mode(args):
    print_system("Starting DiffKV in Direct Mode. Loading wrappers and tokenizer...")
    
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from serving.production_session_manager import ProductionSessionManager

    # Auto-detect best device
    try:
        from native_core.mac_utils import get_best_device as _gbd
        _best_device = _gbd()
    except ImportError:
        _best_device = "cuda" if torch.cuda.is_available() else "cpu"
    print_system(f"Auto-selected device: {COLOR_BOLD}{_best_device}{COLOR_RESET}")

    # Wire platform settings
    if _best_device == "mps":
        if os.environ.get("DIFFKV_MPS_APPROXIMATE_ATTN") is None:
            os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
        if os.environ.get("DIFFKV_USE_TORCH_COMPILE") is None:
            os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
        print_system("Apple Silicon/MPS environment configuration:")
        print(f"  - DIFFKV_MPS_APPROXIMATE_ATTN = {os.environ.get('DIFFKV_MPS_APPROXIMATE_ATTN')}")
        print(f"  - DIFFKV_USE_TORCH_COMPILE     = {os.environ.get('DIFFKV_USE_TORCH_COMPILE')}")

    # Preset low configurations
    if args.preset == "low":
        print_system("Applying low preset auto-optimizations...")
        if not args.load_in_4bit and not args.load_in_8bit:
            if _best_device == "cuda":
                args.load_in_4bit = True
                print_system("CUDA + low preset: auto-enabling 4-bit weight quantization (bitsandbytes)")
            elif _best_device == "mps":
                print_system("MPS + low preset: FP16 active (eager weight execution)")
        if args.serving_mode != "lightweight":
            print_system(f"Adjusting serving_mode from '{args.serving_mode}' to 'lightweight' to prevent OOM")
            args.serving_mode = "lightweight"
        if args.rank == 32:
            print_system("Adjusting KV rank from 32 to 16")
            args.rank = 16

    quantization_config = None
    if (args.load_in_4bit or args.load_in_8bit) and _best_device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            if args.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print_system("Quantization: 4-bit NF4 enabled")
            elif args.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                print_system("Quantization: 8-bit LLM.int8 enabled")
        except ImportError:
            print_warning("bitsandbytes not installed, loading model in full/half precision.")

    if _best_device == "mps" and args.preset:
        try:
            from native_core.config import DiffKVConfig
            cfg = DiffKVConfig({"preset": args.preset})
            watermark = cfg.mps_watermark
            if watermark > 0.0:
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(watermark)
                os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(round(watermark * 0.8, 2))
                print_system(f"MPS watermarks: high={watermark}, low={round(watermark * 0.8, 2)}")
                torch.mps.set_per_process_memory_fraction(watermark)
        except Exception as e:
            print_warning(f"Could not configure MPS memory limits: {e}")

    print_system(f"Loading Model: {COLOR_BOLD}{args.model}{COLOR_RESET}...")
    wrapper = DiffKVHFWrapper(
        args.model,
        config={
            'rank':             args.rank,
            'micro_block_size': args.micro_block_size,
            'block_size':       args.micro_block_size,
            'serving_mode':     args.serving_mode,
            'mode':             'fp16',
            'quantization':     'int4' if args.load_in_4bit else ('int8' if args.load_in_8bit else None),
            'preset':           args.preset,
        },
        device=_best_device,
        quantization_config=quantization_config,
    )
    
    draft_wrapper = None
    if args.draft_model:
        print_system(f"Loading Speculative Draft Model: {args.draft_model}...")
        draft_wrapper = DiffKVHFWrapper(
            args.draft_model,
            config={
                'rank':             args.rank,
                'micro_block_size': args.micro_block_size,
                'block_size':       args.micro_block_size,
                'serving_mode':     args.serving_mode,
                'mode':             'fp16',
                'quantization':     'int4' if args.load_in_4bit else ('int8' if args.load_in_8bit else None),
                'preset':           args.preset,
            },
            device=_best_device,
            quantization_config=quantization_config,
        )

    print_system("Starting batching engine and session manager...")
    engine = ContinuousBatchEngine(wrapper, max_batch_size=args.batch_size, draft_wrapper=draft_wrapper)
    session_manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=args.max_resident_sessions,
    )

    # Start the continuous batching background loop
    engine.start()
    print_system("Batch engine loop started.")

    session_id = session_manager.create_session()
    messages = []
    
    print_system(f"Direct Session initialized: {COLOR_BOLD}{session_id}{COLOR_RESET}")
    print_system("Type /help to list commands.")
    print("-" * 80)

    try:
        while True:
            try:
                user_prompt = await async_input(f"\n{COLOR_USER}{COLOR_BOLD}User >{COLOR_RESET} ")
            except (KeyboardInterrupt, EOFError):
                print()
                break

            user_prompt_stripped = user_prompt.strip()
            if not user_prompt_stripped:
                continue

            # Handle Slash Commands (only if single-line to avoid misinterpreting multi-line pasted text starting with /)
            if user_prompt_stripped.startswith("/") and "\n" not in user_prompt_stripped:
                cmd = user_prompt_stripped.split()[0].lower()
                if cmd in ["/exit", "/quit"]:
                    print_system("Stopping engine and exiting...")
                    break
                elif cmd in ["/reset", "/new"]:
                    session_manager.delete_session(session_id)
                    if hasattr(engine, "cancel"):
                        engine.cancel(session_id)
                    session_id = session_manager.create_session()
                    messages = []
                    print_system(f"Session reset. New Session ID: {COLOR_BOLD}{session_id}{COLOR_RESET}")
                    continue
                elif cmd in ["/paste", "/multiline"]:
                    print_system("Entering multiline paste mode. Paste your text below, then type 'EOF' or '///' on a new line to submit. Type '/cancel' to abort.")
                    lines = []
                    while True:
                        try:
                            line = await async_input("... ")
                        except (KeyboardInterrupt, EOFError):
                            print()
                            lines = []
                            break
                        stripped_line = line.strip()
                        if stripped_line == "/cancel":
                            print_system("Multiline input cancelled.")
                            lines = []
                            break
                        if stripped_line in ("EOF", "///"):
                            break
                        lines.append(line)
                    if not lines:
                        continue
                    user_prompt_stripped = "\n".join(lines).strip()
                elif cmd == "/help":
                    print_system("Available commands:")
                    print("  /reset, /new       : Reset current conversation and release KV cache.")
                    print("  /paste, /multiline : Enter multiline mode for pasting papers or long text.")
                    print("  /stats             : Print model details, VRAM, and KV compression diagnostics.")
                    print("  /srl               : Print Semantic Routing Layer metadata for this session.")
                    print("  /exit, /quit       : Shutdown engine and exit.")
                    continue
                elif cmd == "/stats":
                    print_system("Analyzing engine diagnostics...")
                    
                    process = None
                    try:
                        import psutil
                        process = psutil.Process()
                    except ImportError:
                        pass
                        
                    rss_gb = (process.memory_info().rss / 1e9) if process else 0.0
                    
                    vram_allocated_gb = 0.0
                    if _best_device == "mps":
                        try:
                            vram_allocated_gb = torch.mps.current_allocated_memory() / 1e9
                        except Exception:
                            pass
                    elif _best_device == "cuda":
                        try:
                            vram_allocated_gb = torch.cuda.memory_allocated() / 1e9
                        except Exception:
                            pass
                            
                    kv_summary = {}
                    if wrapper.manager is not None:
                        try:
                            kv_summary = wrapper.manager.runtime_summary()
                        except Exception:
                            pass
                            
                    print(f"\n{COLOR_BOLD}=== DIRECT RUNTIME METRICS ==={COLOR_RESET}")
                    print(f"Model:           {args.model}")
                    print(f"Device:          {_best_device}")
                    print(f"Serving Mode:    {args.serving_mode}")
                    print(f"VRAM Allocated:  {vram_allocated_gb:.3f} GB")
                    print(f"Process RSS:     {rss_gb:.3f} GB")
                    
                    if kv_summary:
                        print(f"Active Sessions: {kv_summary.get('sessions', 0)}")
                        print(f"VRAM Saved:      {kv_summary.get('vram_saved_mb', 0.0):.2f} MB")
                        print(f"Compressions:    {kv_summary.get('total_compressions', 0)}")
                        print(f"Avg Cosine Sim:  {kv_summary.get('avg_cosine_sim', 0.0):.4f}")
                        print(f"Avg Norm Drift:  {kv_summary.get('avg_norm_drift', 0.0):.4f}")
                        
                        pager = kv_summary.get("pager", {})
                        if pager:
                            print(f"Pager Resident:  {pager.get('gpu_resident_mb', 0.0):.2f} MB")
                            print(f"Total Evictions: {pager.get('total_evictions', 0)}")
                            print(f"Total Reloads:   {pager.get('total_reloads', 0)}")
                    print("==============================\n")
                    continue
                elif cmd == "/srl":
                    print_system(f"Retrieving SRL state for session {session_id}...")
                    kv_mgr = wrapper.manager
                    if kv_mgr is None:
                        print_error("KV manager not initialized")
                        continue
                    srl_state = kv_mgr.get_srl_state(session_id)
                    session_config = getattr(kv_mgr, "session_configs", {}).get(session_id, {})
                    
                    print(f"\n{COLOR_BOLD}=== SRL SESSION STATS ==={COLOR_RESET}")
                    print(f"Session ID:   {session_id}")
                    print(f"SRL Enabled:  {session_config.get('srl_enabled', True)}")
                    if srl_state is None:
                        print("SRL Built:    False")
                        print("Reason:       SRL index has not been built yet (requires generation/prefill).")
                    else:
                        print("SRL Built:    True")
                        print(f"K-Min/K-Max:  {srl_state.k_min} / {srl_state.k_max}")
                        print(f"Threshold:    {srl_state.routing_threshold}")
                        print(f"Cache Size:   {getattr(srl_state, 'cache_size_tokens', 0)} tokens")
                    print("=========================\n")
                    continue
                else:
                    print_warning(f"Unknown command: {cmd}. Type /help for assistance.")
                    continue

            # Chat Completion sequence
            messages.append({"role": "user", "content": user_prompt_stripped})
            
            # Format inputs
            history = session_manager.get_history(session_id)
            full_context = list(history) if history else []
            # Append the new user prompt
            full_context.append(messages[-1])
            
            prompt = engine.tokenizer.apply_chat_template(full_context, tokenize=False, add_generation_prompt=True)
            
            payload = {
                "prompt":             prompt,
                "messages":           messages,
                "max_tokens":         args.max_tokens,
                "temperature":        args.temperature,
                "top_p":              args.top_p,
                "repetition_penalty": args.repetition_penalty,
            }

            print(f"\n{COLOR_AI}{COLOR_BOLD}AI >{COLOR_RESET} ", end="", flush=True)

            start_time = time.time()
            first_token_time = None
            response_chunks = []
            
            is_finished = False
            try:
                # Submit request to engine queue
                queue = await engine.submit(session_id, payload)
                
                while True:
                    chunk = await queue.get()
                    if "error" in chunk:
                        print()
                        print_error(f"Engine generation error: {chunk['error']}")
                        messages.pop()
                        break
                    
                    text_delta = chunk.get("text", "")
                    if text_delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        print(text_delta, end="", flush=True)
                        response_chunks.append(text_delta)
                        
                    if chunk.get("is_final"):
                        is_finished = True
                        break
                
                print() # newline
                
                if response_chunks:
                    result_text = "".join(response_chunks)
                    messages.append({"role": "assistant", "content": result_text})
                    
                    # Sync history to session manager
                    session_manager.clear_history(session_id)
                    for msg in messages:
                        session_manager.append_message(session_id, msg["role"], msg["content"])
                        
                    # Update session prefix registry for the next turn
                    try:
                        next_turn_messages = list(messages)
                        full_next_prompt = engine.tokenizer.apply_chat_template(
                            next_turn_messages, tokenize=False, add_generation_prompt=False
                        )
                        engine.update_session_token_prefix(session_id, full_next_prompt)
                    except Exception as prefix_err:
                        print_warning(f"Could not update session token prefix: {prefix_err}")

                    # Performance Metrics
                    end_time = time.time()
                    est_tokens = len(engine.tokenizer.encode(result_text, add_special_tokens=False))
                    total_duration = end_time - start_time
                    ttft_str = f"{(first_token_time - start_time) * 1000:.1f}ms" if first_token_time else "N/A"
                    
                    if first_token_time and end_time > first_token_time:
                        speed = est_tokens / (end_time - first_token_time)
                        speed_str = f"{speed:.1f} tok/s"
                    else:
                        speed_str = "N/A"
                        
                    print_metrics(f"[Metrics] TTFT: {ttft_str} | Speed: {speed_str} | Generated: {est_tokens} tokens | Duration: {total_duration:.2f}s")
                    
            except asyncio.CancelledError:
                if not is_finished:
                    if hasattr(engine, "cancel"):
                        engine.cancel(session_id, free_kv=False)
                raise
            except Exception as e:
                print()
                print_error(f"Generation failed: {e}")
                messages.pop()
                
    finally:
        # Stop continuous batching engine
        await engine.stop()
        print_system("Engine stopped.")

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DiffKV CLI: Interactive terminal interface for testing models running with DiffKV.")
    
    # Mode selection
    parser.add_argument('--api-url', type=str, default=None,
                        help="Gateway API Base URL (e.g. http://localhost:8000/v1). If specified, runs in Client Mode.")
    
    # Model configuration
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help="HuggingFace model ID or path.")
    parser.add_argument('--rank', type=int, default=32,
                        help="SVD rank for KV compression. Higher = better quality. CAP at head_dim.")
    parser.add_argument('--micro-block-size', type=int, default=256,
                        help="Tokens per compressed KV block. S=256.")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Maximum batch size for engine.")
    parser.add_argument('--serving-mode', type=str,
                        choices=['lightweight', 'balanced', 'performance', 'long-context', 'fused-sparse'],
                        default='balanced',
                        help="KV cache serving mode.")
    parser.add_argument('--preset', type=str,
                        choices=['low', 'mid', 'high'],
                        default='mid',
                        help="Hardware optimization preset.")
    
    # Quantization flags
    parser.add_argument('--load-in-4bit', action='store_true',
                        help="Load model weights in 4-bit NF4 quantization.")
    parser.add_argument('--load-in-8bit', action='store_true',
                        help="Load model weights in 8-bit quantization.")
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=2048,
                        help="Max tokens to generate per response.")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument('--top-p', type=float, default=0.9,
                        help="Top-p sampling probability.")
    parser.add_argument('--repetition-penalty', type=float, default=1.15,
                        help="Repetition penalty parameter.")
    
    # Session manager params
    parser.add_argument('--max-resident-sessions', type=int, default=4,
                        help="Maximum resident sessions in VRAM.")
    parser.add_argument('--draft-model', type=str, default=None,
                        help="Draft model for speculative decoding.")
    
    args = parser.parse_args()

    # Disable parallel tokenization warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Run client or direct mode
    if args.api_url is not None:
        try:
            asyncio.run(run_client_mode(args))
        except KeyboardInterrupt:
            print("\nExiting.")
    else:
        try:
            asyncio.run(run_direct_mode(args))
        except KeyboardInterrupt:
            print("\nExiting.")

if __name__ == '__main__':
    main()
