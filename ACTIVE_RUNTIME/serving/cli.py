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
# ANSI Color Codes for Styling (Sky Blue and Dark Blue Theme)
# ---------------------------------------------------------------------------
COLOR_RESET = "\033[0m"
COLOR_SKY_BLUE = "\033[38;5;81m"     # Sky Blue (accent color)
COLOR_DEEP_BLUE = "\033[38;5;27m"    # Deep/Dark Blue (theme accent)
COLOR_STEEL_BLUE = "\033[38;5;75m"   # Steel Blue (subtle)

COLOR_USER = COLOR_STEEL_BLUE        # Soft Steel Blue for User prompt
COLOR_AI = COLOR_SKY_BLUE            # Sky Blue for Model prompt
COLOR_SYSTEM = "\033[38;5;33m"       # Mid Blue for System messages
COLOR_WARNING = "\033[38;5;214m"      # Warm Gold/Orange for warnings
COLOR_ERROR = "\033[38;5;203m"        # Soft Red for errors
COLOR_DIM = "\033[90m"              # Dark Gray
COLOR_BOLD = "\033[1m"              # Bold

def print_system(msg):
    print(f"{COLOR_SYSTEM}{COLOR_BOLD}[System]{COLOR_RESET} {COLOR_SYSTEM}{msg}{COLOR_RESET}")

def print_warning(msg):
    print(f"{COLOR_WARNING}{COLOR_BOLD}[Warning]{COLOR_RESET} {COLOR_WARNING}{msg}{COLOR_RESET}")

def print_error(msg):
    print(f"{COLOR_ERROR}{COLOR_BOLD}[Error]{COLOR_RESET} {COLOR_ERROR}{msg}{COLOR_RESET}")

def print_metrics(msg):
    print(f"{COLOR_DIM}{msg}{COLOR_RESET}")

def get_display_model_name(model_path_or_name: str) -> str:
    if not model_path_or_name:
        return "AI"
    if model_path_or_name == "0.5b":
        return "Qwen2.5-0.5B"
    if model_path_or_name == "1.5b":
        return "Qwen2.5-1.5B"
    
    basename = os.path.basename(model_path_or_name)
    if basename.endswith(".gguf"):
        basename = basename[:-5]
        
    # Clean up Qwen naming or generic paths
    name = basename
    if name.lower().startswith("qwen"):
        parts = name.split("-")
        capitalized_parts = []
        for p in parts:
            if p.lower() == "qwen2.5":
                capitalized_parts.append("Qwen2.5")
            elif p.lower() == "qwen":
                capitalized_parts.append("Qwen")
            elif p.lower().endswith("b") and p[:-1].replace(".", "").isdigit():
                capitalized_parts.append(p[:-1] + "B")
            elif p.lower() == "instruct":
                capitalized_parts.append("Instruct")
            else:
                capitalized_parts.append(p.capitalize())
        name = "-".join(capitalized_parts)
    
    for suffix in ["-q4_k_m", "-q8_0", "_q4_k_m", "_q8_0"]:
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            
    return name

def show_banner():
    logo_lines = [
        " ██████╗  ██╗  ██╗██╗   ██╗",
        " ██╔══██╗ ██║ ██╔╝██║   ██║",
        " ██║  ██║ █████═╝ ╚██╗ ██╔╝",
        " ██║  ██║ ██╔═██╗  ╚████╔╝ ",
        " ██████╔╝ ██║  ██╗  ╚██╔╝  ",
        " ╚══════╝ ╚═╝  ╚═╝   ╚═╝   "
    ]
    # Color gradient from Sky Blue to Deep Blue
    colors = [81, 75, 69, 39, 33, 27, 26, 20]
    print()
    for line in logo_lines:
        colored_line = ""
        n_chars = len(line)
        for i, char in enumerate(line):
            color_idx = min(len(colors) - 1, int((i / n_chars) * len(colors)))
            color_code = colors[color_idx]
            colored_line += f"\033[38;5;{color_code}m{char}"
        print(colored_line + "\033[0m")
    print(f"      {COLOR_SKY_BLUE}DKV — High-Performance Inference{COLOR_RESET}")
    print(f"      {COLOR_DEEP_BLUE}Serving Engine & Runtime Context CLI{COLOR_RESET}\n")

class ThemedStdout:
    def __init__(self, original, buffer_stderr=False):
        self.original = original
        self.buffer_stderr = buffer_stderr
        self._buffer = []
        self._buffering_enabled = False
        self._skip_next_newline = False

    def enable_buffering(self):
        if self.buffer_stderr:
            self._buffering_enabled = True

    def disable_and_flush(self):
        if self.buffer_stderr:
            self._buffering_enabled = False
            if self._buffer:
                content = "".join(self._buffer)
                self._buffer.clear()
                self.original.write(content)
                self.original.flush()

    def write(self, data):
        if not data:
            return self.original.write(data)
            
        # Consume the trailing newline printed by Python's print() if previous message was filtered
        if data == "\n" and self._skip_next_newline:
            self._skip_next_newline = False
            return len(data)
            
        if data != "\n":
            self._skip_next_newline = False
            
        # Bypass buffering for engine-level [DKV] warnings/updates
        # so they print immediately when they occur (e.g. before prefill starts).
        is_engine_warning = "[DKV]" in data
        
        if self._buffering_enabled and not is_engine_warning:
            formatted_data = self._format_themed_output(data)
            if formatted_data == "" and data.strip() != "":
                self._skip_next_newline = True
            self._buffer.append(formatted_data)
            return len(data)
            
        formatted_data = self._format_themed_output(data)
        if formatted_data == "" and data.strip() != "":
            self._skip_next_newline = True
        return self.original.write(formatted_data)

    def _format_themed_output(self, data):
        # Don't double color if escape sequences are already present in system messages
        if "\033[" in data and not any(tag in data for tag in ["[System]", "[Warning]", "[Error]", "[DKV]"]):
            return data
            
        lines = data.split("\n")
        processed_lines = []
        is_verbose = os.environ.get("DKV_VERBOSE") == "1"
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                processed_lines.append(line)
                continue
                
            # Filter developer telemetry logs if not in verbose mode
            is_telemetry = any(tag in stripped for tag in [
                "[DKV BatchEngine]", 
                "[DKV Telemetry]", 
                "[DKV VRAM]", 
                "[DKV Step]"
            ])
            is_important = any(tag in stripped for tag in [
                "WARNING:", "Warning:", "ERROR:", "Error:", "[Warning]", "[Error]"
            ])
            
            if is_telemetry and not is_important and not is_verbose:
                continue

            # Check if this is a dashed separator line
            if len(stripped) >= 40 and all(c == '-' for c in stripped):
                processed_lines.append(f"\033[38;5;27m{line}\033[0m")
                continue
                
            # Check if this is a bullet point line (starts with spaces and "- ")
            # e.g. "  - DKV_MPS_APPROXIMATE_ATTN = 1"
            if line.startswith("  - ") or line.startswith("    - "):
                idx = line.find("- ")
                if idx != -1:
                    indent = line[:idx]
                    content = line[idx+2:]
                    processed_lines.append(f"{indent}\033[38;5;33m-\033[0m \033[38;5;75m{content}\033[0m")
                else:
                    processed_lines.append(line)
                continue
                
            replacements = [
                ("[DKV MLX Wrapper]", "\033[38;5;27m\033[1m[DKV MLX Wrapper]\033[0m\033[38;5;27m"),
                ("[DKV BatchEngine]", "\033[38;5;27m\033[1m[DKV BatchEngine]\033[0m\033[38;5;27m"),
                ("[DKV Telemetry]", "\033[38;5;27m\033[1m[DKV Telemetry]\033[0m\033[38;5;27m"),
                ("[DKV VRAM]", "\033[38;5;27m\033[1m[DKV VRAM]\033[0m\033[38;5;27m"),
                ("[DKV Step]", "\033[38;5;27m\033[1m[DKV Step]\033[0m\033[38;5;27m"),
                ("[DKV MLX]", "\033[38;5;81m\033[1m[DKV MLX]\033[0m\033[38;5;81m"),
                ("[DKV]", "\033[38;5;81m\033[1m[DKV]\033[0m\033[38;5;81m"),
                ("[System]", "\033[38;5;33m\033[1m[System]\033[0m\033[38;5;33m"),
                ("[Warning]", "\033[38;5;214m\033[1m[Warning]\033[0m\033[38;5;214m"),
                ("[Error]", "\033[38;5;203m\033[1m[Error]\033[0m\033[38;5;203m"),
                ("WARNING:", "\033[38;5;214m\033[1mWARNING:\033[0m\033[38;5;214m"),
                ("Warning:", "\033[38;5;214m\033[1mWarning:\033[0m\033[38;5;214m"),
                ("ERROR:", "\033[38;5;203m\033[1mERROR:\033[0m\033[38;5;203m"),
                ("Error:", "\033[38;5;203m\033[1mError:\033[0m\033[38;5;203m"),
            ]
            
            modified_line = line
            replaced = False
            for target, replacement in replacements:
                if target in modified_line:
                    modified_line = modified_line.replace(target, replacement)
                    replaced = True
                    
            if replaced:
                modified_line = modified_line + "\033[0m"
            processed_lines.append(modified_line)
            
        result = "\n".join(processed_lines)
        if result.strip() == "" and data.strip() != "":
            return ""
        return result

    def flush(self):
        return self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def flush(self):
        return self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)

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
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise EOFError()
            except OSError:
                raise EOFError()
                
            chunks = [chunk]
            total_bytes = len(chunk)
            paste_spinner_shown = False
            spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner_idx = 0

            # Accumulate any immediately available chunks (pasted text)
            # Use 50ms timeout (was 10ms) to safely capture large multi-chunk pastes
            # like research papers without premature read termination.
            while True:
                try:
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if r:
                        next_chunk = os.read(fd, 65536)
                        if not next_chunk:
                            break
                        chunks.append(next_chunk)
                        total_bytes += len(next_chunk)
                        # Show a paste-reading spinner for large inputs (>4KB = likely a paste)
                        if total_bytes > 4096:
                            frame = spinner_frames[spinner_idx % len(spinner_frames)]
                            spinner_idx += 1
                            sys.stdout.write(f"\r{COLOR_SYSTEM}{frame} Reading paste... {total_bytes // 1024}KB{COLOR_RESET}  ")
                            sys.stdout.flush()
                            paste_spinner_shown = True
                    else:
                        break
                except (AttributeError, OSError, select.error):
                    break

            if paste_spinner_shown:
                sys.stdout.write(f"\r{COLOR_SYSTEM}✓ Captured {total_bytes // 1024}KB ({total_bytes} chars){COLOR_RESET}\n")
                sys.stdout.flush()
                    
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
# Helper function for automatic, high-capacity pasting
# ---------------------------------------------------------------------------
async def run_raw_paste() -> str:
    import select
    def _read():
        fd = sys.stdin.fileno()
        is_tty = sys.stdin.isatty()
        
        old_settings = None
        if is_tty:
            try:
                import termios
                old_settings = termios.tcgetattr(fd)
                new_settings = termios.tcgetattr(fd)
                # Disable ICANON (canonical mode) and ECHO (terminal echoing)
                new_settings[3] &= ~termios.ICANON
                new_settings[3] &= ~termios.ECHO
                termios.tcsetattr(fd, termios.TCSANOW, new_settings)
            except Exception:
                pass
                
        try:
            # Phase 1: Wait for first byte to arrive (blocking select)
            try:
                r, _, _ = select.select([fd], [], [])
                if not r:
                    return ""
            except (AttributeError, OSError, select.error):
                return sys.stdin.read()
                
            chunks = []
            total_bytes = 0
            spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner_idx = 0

            # Phase 2: Read chunks with spinner; 200ms pause signals end of paste
            while True:
                try:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total_bytes += len(chunk)

                    # Animated spinner showing KB received
                    frame = spinner_frames[spinner_idx % len(spinner_frames)]
                    spinner_idx += 1
                    sys.stdout.write(f"\r{COLOR_SYSTEM}{frame} Receiving paste... {total_bytes // 1024}KB ({total_bytes} chars){COLOR_RESET}  ")
                    sys.stdout.flush()
                    
                    r, _, _ = select.select([fd], [], [], 0.20)
                    if not r:
                        break
                except OSError:
                    break
                except (AttributeError, select.error):
                    break

            if total_bytes > 0:
                est_tokens = total_bytes // 4
                sys.stdout.write(f"\r{COLOR_SYSTEM}✓ Captured {total_bytes // 1024}KB — ~{est_tokens:,} tokens ready to send{COLOR_RESET}\n")
                sys.stdout.flush()
                    
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

    return await asyncio.to_thread(_read)

# ---------------------------------------------------------------------------
# Prefill progress bar — runs in a real daemon thread so it ticks every 1s
# regardless of the asyncio event loop being blocked by MLX/PyTorch forward passes.
# ---------------------------------------------------------------------------
import threading as _threading

class PrefillBar:
    """Starts a daemon thread that animates a progress bar during prefill.

    Usage:
        bar = PrefillBar(n_chars)
        bar.start()
        ...do work...
        bar.stop()   # clears the line, joins the thread
    """
    def __init__(self, n_chars: int):
        self._est_tokens = max(1, n_chars // 4)
        self._stop_event = _threading.Event()
        self._thread = _threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        # Clear the bar line
        sys.stdout.write("\r" + " " * 90 + "\r")
        sys.stdout.flush()

    def _run(self):
        est_tokens = self._est_tokens
        bar_width = 28
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        # Estimate based on MPS 1.5B-4bit with 128-token chunks ≈ 60–80 tok/s net
        est_seconds = max(2.0, est_tokens / 70.0)
        start = time.time()
        idx = 0
        while not self._stop_event.is_set():
            elapsed = time.time() - start
            # Progress capped at 95% until we actually finish
            progress = min(0.95, elapsed / est_seconds)
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = int(progress * 100)
            frame = spinner_frames[idx % len(spinner_frames)]
            idx += 1
            sys.stdout.write(
                f"\r{COLOR_SYSTEM}{frame} Prefilling {est_tokens:,} tokens  "
                f"[{COLOR_SKY_BLUE}{bar}{COLOR_SYSTEM}] {pct}%  {elapsed:.0f}s{COLOR_RESET}  "
            )
            sys.stdout.flush()
            # Tick every 1 second — completely independent of the event loop
            self._stop_event.wait(timeout=1.0)

# Shim so existing call sites only need to swap asyncio.Event → PrefillBar
def _make_prefill_bar(n_input_chars: int):
    """Returns a started PrefillBar if input is large enough, else None."""
    if n_input_chars > 1024:
        bar = PrefillBar(n_input_chars)
        bar.start()
        return bar
    return None

def _stop_prefill_bar(bar):
    """Stop and clear the bar if it was started."""
    if bar is not None:
        bar.stop()


# ---------------------------------------------------------------------------
# Client Mode Helper (talks to running OpenAI compatible API gateway)
# ---------------------------------------------------------------------------
async def run_client_mode(args):
    api_url = args.api_url.rstrip('/')
    session_id = f"cli-session-{uuid.uuid4().hex[:8]}"
    messages = []
    
    show_banner()
    
    print_system(f"Connecting to DKV Gateway at {COLOR_BOLD}{api_url}{COLOR_RESET}...")
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

        # Handle Slash Commands (only if single-line to avoid misinterpreting multi-line pasted text starting with /).
        # Exception: /file explicitly supports a multi-line question suffix after the | separator.
        _first_tok = user_prompt_stripped.split()[0].lower() if user_prompt_stripped.split() else ""
        if user_prompt_stripped.startswith("/") and ("\n" not in user_prompt_stripped or _first_tok == "/file"):
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
            elif cmd == "/system":
                parts = user_prompt_stripped.split(maxsplit=1)
                if len(parts) > 1:
                    sys_prompt = parts[1]
                else:
                    print_system("Please provide the system prompt text after /system.")
                    continue
                messages = [m for m in messages if m["role"] != "system"]
                messages.insert(0, {"role": "system", "content": sys_prompt})
                print_system(f"System prompt set to: {COLOR_BOLD}{sys_prompt}{COLOR_RESET}")
                continue
            elif cmd in ["/paste", "/multiline"]:
                print_system("Raw paste mode active. Paste your text now (it will automatically submit when pasting completes).")
                pasted_text = await run_raw_paste()
                if not pasted_text.strip():
                    print_system("No text pasted. Cancelled.")
                    continue
                _est_tokens = len(pasted_text) // 4
                _limit = int(os.environ.get("DKV_MAX_INPUT_TOKENS", "32768"))
                if _est_tokens > _limit:
                    print_warning(
                        f"Input is ~{_est_tokens:,} tokens, which exceeds the limit of {_limit:,}. "
                        f"It will be truncated on submit. Raise the limit with: "
                        f"DKV_MAX_INPUT_TOKENS={_est_tokens + 2048} python serving/cli.py ..."
                    )
                else:
                    print_system(f"Captured ~{_est_tokens:,} tokens (limit: {_limit:,}). Submitting...")
                user_prompt_stripped = pasted_text.strip()
            elif cmd == "/file":
                parts = user_prompt_stripped.split(maxsplit=1)
                if len(parts) > 1:
                    subparts = parts[1].split("|", 1)
                    filepath = subparts[0].strip()
                    question_suffix = ""
                    if len(subparts) > 1:
                        question_suffix = subparts[1].strip()
                    
                    if not os.path.exists(filepath):
                        print_system(f"File not found: {COLOR_BOLD}{filepath}{COLOR_RESET}")
                        continue
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                            file_content = f.read()
                        if question_suffix:
                            file_content = f"{file_content}\n\n{question_suffix}"
                        pasted_text = file_content
                    except Exception as e:
                        print_system(f"Error reading file: {e}")
                        continue
                else:
                    print_system("Please provide a filepath. Usage: /file <path> | [optional question]")
                    continue

                _est_tokens = len(pasted_text) // 4
                _limit = int(os.environ.get("DKV_MAX_INPUT_TOKENS", "32768"))
                if _est_tokens > _limit:
                    print_warning(
                        f"Input is ~{_est_tokens:,} tokens, which exceeds the limit of {_limit:,}. "
                        f"It will be truncated on submit. Raise the limit with: "
                        f"DKV_MAX_INPUT_TOKENS={_est_tokens + 2048} python serving/cli.py ..."
                    )
                else:
                    print_system(f"Loaded file {COLOR_BOLD}{filepath}{COLOR_RESET} (~{_est_tokens:,} tokens). Submitting...")
                user_prompt_stripped = pasted_text.strip()
            elif cmd == "/help":
                print_system("Available commands:")
                print("  /reset, /new       : Clear chat history and start a new session.")
                print("  /system <prompt>   : Set a custom system prompt to guide the AI.")
                print("  /paste, /multiline : Enter multiline mode for pasting papers or long text.")
                print("                       Raise token limit: DKV_MAX_INPUT_TOKENS=65536 (default: 32768)")
                print("  /file <path>       : Load a text file directly from disk as the prompt.")
                print("                       Format: /file <path> | [optional question]")
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
        
        if hasattr(sys.stderr, "enable_buffering"):
            sys.stderr.enable_buffering()

        model_display = get_display_model_name(args.model)
        n_input_chars = len(user_prompt_stripped)
        
        start_time = time.time()
        first_token_time = None
        response_chunks = []

        # Show prefill progress bar for large inputs (>1KB ~ 250 tokens)
        # Runs in a daemon thread — ticks every 1s independent of the event loop.
        prefill_bar = _make_prefill_bar(n_input_chars)
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", f"{api_url}/chat/completions", json=payload) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        _stop_prefill_bar(prefill_bar)
                        prefill_bar = None
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
                                         # First token — stop the prefill bar and print AI prompt
                                         first_token_time = time.time()
                                         _stop_prefill_bar(prefill_bar)
                                         prefill_bar = None
                                         print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)
                                     print(content, end="", flush=True)
                                     response_chunks.append(content)
                             except Exception:
                                 pass

            # If we never got a token, stop bar and print AI prompt anyway
            _stop_prefill_bar(prefill_bar)
            prefill_bar = None
            if not first_token_time:
                print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)
                                 
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
        finally:
            if hasattr(sys.stderr, "disable_and_flush"):
                sys.stderr.disable_and_flush()

def json_loads(s):
    import json
    return json.loads(s)


# ---------------------------------------------------------------------------
# Direct Mode Helper (runs batch engine directly in-process)
# ---------------------------------------------------------------------------
async def run_direct_mode(args):
    show_banner()
    print_system("Starting DKV in Direct Mode. Loading wrappers and tokenizer...")
    
    from serving.hf_dkv_wrapper import DKVHFWrapper
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
        if os.environ.get("DKV_MPS_APPROXIMATE_ATTN") is None:
            os.environ["DKV_MPS_APPROXIMATE_ATTN"] = "1"
        if os.environ.get("DKV_USE_TORCH_COMPILE") is None:
            os.environ["DKV_USE_TORCH_COMPILE"] = "0"
        print_system("Apple Silicon/MPS environment configuration:")
        print(f"  - DKV_MPS_APPROXIMATE_ATTN = {os.environ.get('DKV_MPS_APPROXIMATE_ATTN')}")
        print(f"  - DKV_USE_TORCH_COMPILE     = {os.environ.get('DKV_USE_TORCH_COMPILE')}")

    if _best_device == "cuda":
        # ── CUDA ↔ MLX parity defaults (explicit env still wins) ──
        # V-side rebalancing before the joint K|V SVD, matching MLX's
        # v_scale_on.  It is already the code default in
        # compress_layer_blocks_gpu; set here too so the CLI surfaces it and a
        # profile/A-B can flip it with DKV_V_SCALE=0.
        os.environ.setdefault("DKV_V_SCALE", "1")
        print_system("CUDA ↔ MLX parity configuration:")
        print(f"  - DKV_V_SCALE              = {os.environ.get('DKV_V_SCALE')} (V rebalanced before joint SVD)")
        # CAD (relational accuracy) auto-enables for high/quality/max presets in
        # DKVHFWrapper — matching MLX (mlx_dkv_wrapper); reported there.
        # Streaming compression (long-context peak VRAM, DKV_STREAMING_COMPRESS)
        # and lower engagement (DKV_ENGAGE_THRESHOLD) are left opt-in: MLX
        # engages DKV from token 1 by design (slower short-context), while the
        # CUDA default bypasses to dense below the threshold for faster short
        # prompts.  Flip either explicitly to match MLX's always-on behavior.
        print(f"  - CAD                        = auto for high/quality/max preset (see wrapper)")

    # ── BEST DKV decode config (shared with the serving gateway) ──
    # setdefault the user-optimal serving policy: fast exact-dense for short prompts, DKV
    # sparse at >=8k, decompress-and-cache fast decode + adaptive bias when sparse. Single
    # source of truth in serving/decode_config.py (an explicit env still wins). Force DKV
    # always-on with DKV_COMPRESSED_DECODE=1.
    from serving.decode_config import apply_best_decode_defaults
    apply_best_decode_defaults(log=print_system)

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
        # `is None` = the user left it alone. The old `== 32` could not tell that
        # from an explicit --rank 32, so it silently overrode a deliberate choice.
        if args.rank is None:
            print_system("Adjusting KV rank to 16 (low preset)")
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
            from native_core.config import DKVConfig
            cfg = DKVConfig({"preset": args.preset})
            watermark = cfg.mps_watermark
            if watermark > 0.0:
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(watermark)
                os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(round(watermark * 0.8, 2))
                print_system(f"MPS watermarks: high={watermark}, low={round(watermark * 0.8, 2)}")
                torch.mps.set_per_process_memory_fraction(watermark)
        except Exception as e:
            print_warning(f"Could not configure MPS memory limits: {e}")

    print_system(f"Loading Model: {COLOR_BOLD}{args.model}{COLOR_RESET}...")
    wrapper = DKVHFWrapper(
        args.model,
        config={
            **({'rank': args.rank} if args.rank else {}),
            # Only pass a block size when the user actually asked for one. A
            # hardcoded default here shadows the runtime's, and the runtime's is the
            # one that was measured (MLX: 256 -> 1024 took linkbench 9/24 to 24/24 =
            # dense, and the pool from 0.95x of the KV it replaces to 0.28x). See
            # ACTIVE_RUNTIME/docs/cuda_port_record.md.
            **({'micro_block_size': args.micro_block_size,
                'block_size':       args.micro_block_size}
               if args.micro_block_size else {}),
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
        draft_wrapper = DKVHFWrapper(
            args.draft_model,
            config={
                **({'rank': args.rank} if args.rank else {}),
                # Only pass a block size when the user actually asked for one. A
                # hardcoded default here shadows the runtime's, and the runtime's is the
                # one that was measured (MLX: 256 -> 1024 took linkbench 9/24 to 24/24 =
                # dense, and the pool from 0.95x of the KV it replaces to 0.28x). See
                # ACTIVE_RUNTIME/docs/cuda_port_record.md.
                **({'micro_block_size': args.micro_block_size,
                    'block_size':       args.micro_block_size}
                   if args.micro_block_size else {}),
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

            # Handle Slash Commands (only if single-line to avoid misinterpreting multi-line pasted text starting with /).
            # Exception: /file explicitly supports a multi-line question suffix after the | separator.
            _first_tok = user_prompt_stripped.split()[0].lower() if user_prompt_stripped.split() else ""
            if user_prompt_stripped.startswith("/") and ("\n" not in user_prompt_stripped or _first_tok == "/file"):
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
                elif cmd == "/system":
                    parts = user_prompt_stripped.split(maxsplit=1)
                    if len(parts) > 1:
                        sys_prompt = parts[1]
                    else:
                        print_system("Please provide the system prompt text after /system.")
                        continue
                    messages = [m for m in messages if m["role"] != "system"]
                    messages.insert(0, {"role": "system", "content": sys_prompt})
                    print_system(f"System prompt set to: {COLOR_BOLD}{sys_prompt}{COLOR_RESET}")
                    continue
                elif cmd in ["/paste", "/multiline"]:
                    print_system("Raw paste mode active. Paste your text now (it will automatically submit when pasting completes).")
                    pasted_text = await run_raw_paste()
                    if not pasted_text.strip():
                        print_system("No text pasted. Cancelled.")
                        continue
                    _est_tokens = len(pasted_text) // 4
                    _limit = int(os.environ.get("DKV_MAX_INPUT_TOKENS", "32768"))
                    if _est_tokens > _limit:
                        print_warning(
                            f"Input is ~{_est_tokens:,} tokens, which exceeds the limit of {_limit:,}. "
                            f"It will be truncated on submit. Raise the limit with: "
                            f"DKV_MAX_INPUT_TOKENS={_est_tokens + 2048} python serving/cli.py ..."
                        )
                    else:
                        print_system(f"Captured ~{_est_tokens:,} tokens (limit: {_limit:,}). Submitting...")
                    user_prompt_stripped = pasted_text.strip()
                elif cmd == "/file":
                    parts = user_prompt_stripped.split(maxsplit=1)
                    if len(parts) > 1:
                        subparts = parts[1].split("|", 1)
                        filepath = subparts[0].strip()
                        question_suffix = ""
                        if len(subparts) > 1:
                            question_suffix = subparts[1].strip()
                        
                        if not os.path.exists(filepath):
                            print_system(f"File not found: {COLOR_BOLD}{filepath}{COLOR_RESET}")
                            continue
                        try:
                            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                                file_content = f.read()
                            if question_suffix:
                                file_content = f"{file_content}\n\n{question_suffix}"
                            pasted_text = file_content
                        except Exception as e:
                            print_system(f"Error reading file: {e}")
                            continue
                    else:
                        print_system("Please provide a filepath. Usage: /file <path> | [optional question]")
                        continue

                    _est_tokens = len(pasted_text) // 4
                    _limit = int(os.environ.get("DKV_MAX_INPUT_TOKENS", "32768"))
                    if _est_tokens > _limit:
                        print_warning(
                            f"Input is ~{_est_tokens:,} tokens, which exceeds the limit of {_limit:,}. "
                            f"It will be truncated on submit. Raise the limit with: "
                            f"DKV_MAX_INPUT_TOKENS={_est_tokens + 2048} python serving/cli.py ..."
                        )
                    else:
                        print_system(f"Loaded file {COLOR_BOLD}{filepath}{COLOR_RESET} (~{_est_tokens:,} tokens). Submitting...")
                    user_prompt_stripped = pasted_text.strip()
                elif cmd == "/help":
                    print_system("Available commands:")
                    print("  /reset, /new       : Reset current conversation and release KV cache.")
                    print("  /system <prompt>   : Set a custom system prompt to guide the AI.")
                    print("  /paste, /multiline : Enter multiline mode for pasting papers or long text.")
                    print("                       Raise token limit: DKV_MAX_INPUT_TOKENS=65536 (default: 32768)")
                    print("  /file <path>       : Load a text file directly from disk as the prompt.")
                    print("                       Format: /file <path> | [optional question]")
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
                    # Debug print blocks
                    try:
                        blocks = kv_mgr.get_streaming_blocks(session_id, 0)
                        print(f"DEBUG blocks count: {len(blocks)}")
                        for idx, b in enumerate(blocks):
                            print(f"  Block {idx}: anchor={b.anchor_idx}, state={getattr(b, 'state', 'NONE')}, pool_idx={getattr(b, 'pool_idx', 'NONE')}, skip={getattr(b, 'skip_compression', False)}")
                        print(f"DEBUG token_ids length: {len(kv_mgr._session_token_ids.get(session_id, []))}")
                        print(f"DEBUG pending_cpu_blocks: {getattr(kv_mgr, '_pending_cpu_blocks', 0)}")
                    except Exception as de:
                        print(f"DEBUG block printing error: {de}")

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
                        
                        # SAS and EQA-DR parameters
                        if hasattr(srl_state, "concept_tok_1"):
                            print(f"Concept Tok 1: {srl_state.concept_tok_1}")
                            print(f"Concept Tok 2: {srl_state.concept_tok_2}")
                        if hasattr(srl_state, "current_query_segment_id"):
                            print(f"Current Segment: {srl_state.current_query_segment_id}")
                        if hasattr(srl_state, "dynamic_anchors"):
                            print(f"Dynamic Anchors: {srl_state.dynamic_anchors}")
                        if hasattr(srl_state, "prompt_anchors"):
                            print(f"Prompt Anchors: {srl_state.prompt_anchors}")
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

            if hasattr(sys.stderr, "enable_buffering"):
                sys.stderr.enable_buffering()

            model_display = get_display_model_name(args.model)
            n_input_chars = len(user_prompt_stripped)

            start_time = time.time()
            first_token_time = None
            response_chunks = []

            # Show prefill progress bar for large inputs (>1KB ~ 250 tokens)
            # Runs in a daemon thread — ticks every 1s independent of the event loop.
            prefill_bar = _make_prefill_bar(n_input_chars)
            
            is_finished = False
            try:
                # Submit request to engine queue
                queue = await engine.submit(session_id, payload)
                
                while True:
                    chunk = await queue.get()
                    if "error" in chunk:
                        _stop_prefill_bar(prefill_bar)
                        prefill_bar = None
                        print()
                        print_error(f"Engine generation error: {chunk['error']}")
                        messages.pop()
                        break
                    
                    text_delta = chunk.get("text", "")
                    if text_delta:
                        if first_token_time is None:
                            # First token — stop prefill bar, print AI prompt header
                            first_token_time = time.time()
                            _stop_prefill_bar(prefill_bar)
                            prefill_bar = None
                            print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)
                        print(text_delta, end="", flush=True)
                        response_chunks.append(text_delta)
                        
                    if chunk.get("is_final"):
                        is_finished = True
                        break

                # Ensure bar is stopped (handles zero-token response edge case)
                _stop_prefill_bar(prefill_bar)
                prefill_bar = None
                if not first_token_time:
                    print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)
                
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
                if hasattr(sys.stderr, "disable_and_flush"):
                    sys.stderr.disable_and_flush()
                
    finally:
        # Stop continuous batching engine
        await engine.stop()
        print_system("Engine stopped.")

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    sys.stdout = ThemedStdout(sys.stdout)
    sys.stderr = ThemedStdout(sys.stderr, buffer_stderr=True)
    parser = argparse.ArgumentParser(description="DKV CLI: Interactive terminal interface for testing models running with DKV.")
    
    # Mode selection
    parser.add_argument('--api-url', type=str, default=None,
                        help="Gateway API Base URL (e.g. http://localhost:8000/v1). If specified, runs in Client Mode.")
    
    # Model configuration
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help="HuggingFace model ID or path.")
    parser.add_argument('--rank', type=int, default=None,
                        help="SVD rank for KV compression. Unset = the runtime's own default (32). Do NOT hardcode a value here: passing one explicitly overrides the runtime default and makes 'user asked for 32' indistinguishable from 'left at the default'. CAP at head_dim.")
    parser.add_argument('--micro-block-size', type=int, default=None,
                        help="Tokens per compressed KV block. Unset = the runtime's own measured default (1024 on MLX). Do NOT hardcode a value here: passing one explicitly overrides the runtime default, which is how a measured default silently fails to reach the CLI and the server.")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Maximum batch size for engine.")
    parser.add_argument('--serving-mode', type=str,
                        choices=['lightweight', 'balanced', 'performance', 'long-context', 'fused-sparse'],
                        default='balanced',
                        help="KV cache serving mode.")
    parser.add_argument('--preset', type=str,
                        choices=['low', 'mid', 'high', 'ultra'],
                        default='mid',
                        help="Quality/cost preset. 'ultra' keeps the most "
                             "spectrum and stores keys unrotated; it is the only "
                             "preset that matches dense on distractor retrieval, "
                             "and it costs decode and VRAM to do it.")

    parser.add_argument('--fastdc', action='store_true',
                        help="Decode-optimised path: capture the decode step as "
                             "a CUDA graph and replay it. Byte-identical output "
                             "where it engages (17.3 -> 10.2 s wall at 16k on "
                             "Qwen2.5-1.5B). It engages only when routing is "
                             "non-selective (compressed blocks <= K) and declines "
                             "to a verified no-op otherwise. NOT free on every "
                             "model: it moves per-layer work into the host loop, "
                             "so on wide models at long context, where it "
                             "declines, it costs ~9%.")
    
    # Quantization flags
    parser.add_argument('--load-in-4bit', action='store_true',
                        help="Load model weights in 4-bit NF4 quantization.")
    parser.add_argument('--load-in-8bit', action='store_true',
                        help="Load model weights in 8-bit quantization.")
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=16384,
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

    # --fastdc must reach the environment BEFORE the runtime is imported:
    # dkv_attention resolves _MUTATION_OUT once at import time. The wrapper is
    # imported lazily inside the run functions below, so setting it here is early
    # enough -- but move this and the flag becomes a dead knob that reads as
    # enabled while changing nothing, which this codebase has produced six times.
    if getattr(args, 'fastdc', False):
        os.environ['DKV_FAST_DECODE'] = '1'

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
