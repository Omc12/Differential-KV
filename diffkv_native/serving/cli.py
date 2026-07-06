#!/usr/bin/env python3
import os
import sys
import time
import uuid
import asyncio
import argparse
import subprocess
import threading
import codecs
import httpx

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
        " ██████╗  ██╗ ██████╗ ██████╗  ██╗  ██╗ ██╗   ██╗",
        " ██╔══██╗ ██║ ██╔═══╝ ██╔═══╝  ██║ ██╔╝ ██║   ██║",
        " ██║  ██║ ██║ █████╗  █████╗   █████═╝  ╚██╗ ██╔╝",
        " ██║  ██║ ██║ ██╔══╝  ██╔══╝   ██╔═██╗   ╚████╔╝ ",
        " ██████╔╝ ██║ ██║     ██║      ██║  ██╗   ╚██╔╝  ",
        " ╚══════╝  ╚═╝ ╚═╝     ╚═╝      ╚═╝  ╚═╝    ╚═╝   "
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
    print(f"      {COLOR_SKY_BLUE}Differential KV Cache — High-Performance Inference{COLOR_RESET}")
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
            
        # Bypass buffering for engine-level [DiffKV] warnings/updates
        # so they print immediately when they occur (e.g. before prefill starts).
        is_engine_warning = "[DiffKV]" in data
        
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
        if "\033[" in data and not any(tag in data for tag in ["[System]", "[Warning]", "[Error]", "[DiffKV]"]):
            return data
            
        lines = data.split("\n")
        processed_lines = []
        is_verbose = os.environ.get("DIFFKV_VERBOSE") == "1"
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                processed_lines.append(line)
                continue
                
            # Filter developer telemetry logs if not in verbose mode
            is_telemetry = any(tag in stripped for tag in [
                "[DiffKV BatchEngine]", 
                "[DiffKV Telemetry]", 
                "[DiffKV VRAM]", 
                "[DiffKV Step]"
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
            # e.g. "  - DIFFKV_MPS_APPROXIMATE_ATTN = 1"
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
                ("[DiffKV MLX Wrapper]", "\033[38;5;27m\033[1m[DiffKV MLX Wrapper]\033[0m\033[38;5;27m"),
                ("[DiffKV BatchEngine]", "\033[38;5;27m\033[1m[DiffKV BatchEngine]\033[0m\033[38;5;27m"),
                ("[DiffKV Telemetry]", "\033[38;5;27m\033[1m[DiffKV Telemetry]\033[0m\033[38;5;27m"),
                ("[DiffKV VRAM]", "\033[38;5;27m\033[1m[DiffKV VRAM]\033[0m\033[38;5;27m"),
                ("[DiffKV Step]", "\033[38;5;27m\033[1m[DiffKV Step]\033[0m\033[38;5;27m"),
                ("[DiffKV MLX]", "\033[38;5;81m\033[1m[DiffKV MLX]\033[0m\033[38;5;81m"),
                ("[DiffKV]", "\033[38;5;81m\033[1m[DiffKV]\033[0m\033[38;5;81m"),
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

            # Phase 2: Read chunks with animated spinner; 200ms pause signals end of paste
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
# regardless of the asyncio event loop being blocked by C++ binary I/O.
# ---------------------------------------------------------------------------
import threading as _threading

class PrefillBar:
    """Starts a daemon thread that animates a progress bar during prefill."""
    def __init__(self, n_chars: int):
        self._est_tokens = max(1, n_chars // 4)
        self._stop_event = _threading.Event()
        self._thread = _threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        sys.stdout.write("\r" + " " * 90 + "\r")
        sys.stdout.flush()

    def _run(self):
        est_tokens = self._est_tokens
        bar_width = 28
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        est_seconds = max(2.0, est_tokens / 70.0)
        start = time.time()
        idx = 0
        while not self._stop_event.is_set():
            elapsed = time.time() - start
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
            self._stop_event.wait(timeout=1.0)

def _make_prefill_bar(n_input_chars: int):
    if n_input_chars > 1024:
        bar = PrefillBar(n_input_chars)
        bar.start()
        return bar
    return None

def _stop_prefill_bar(bar):
    if bar is not None:
        bar.stop()


# ---------------------------------------------------------------------------
# Text reference normalization (copied from openai_compatible_api_gateway.py)
# ---------------------------------------------------------------------------
def _normalize_references(text: str) -> str:
    import re
    lines = text.split('\n')
    header_re = re.compile(r'\b(references?|bibliography|works\s+cited|reference\s+list|sources|citations)\b', re.IGNORECASE)
    header_idx = None
    for i, line in enumerate(lines):
        if len(line) <= 100 and header_re.search(line):
            header_idx = i
    
    ref_entry_re = re.compile(r'^(?:[iI]n\s+)?(?:\b[*\-•]\s*)?\[\d+\]')
    unambiguous_re = re.compile(r'^(?:\b[*\-•]\s*)?\[\d+\]')
    
    matching_indices = []
    unambiguous_indices = []
    for i, line in enumerate(lines):
        if header_idx is not None and i <= header_idx:
            continue
        stripped = line.strip()
        if ref_entry_re.match(stripped):
            matching_indices.append(i)
            if unambiguous_re.match(stripped):
                unambiguous_indices.append(i)
                
    if header_idx is not None and not matching_indices:
        matching_indices = []
        unambiguous_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ref_entry_re.match(stripped):
                matching_indices.append(i)
                if unambiguous_re.match(stripped):
                    unambiguous_indices.append(i)
        header_idx = None

    if not matching_indices:
        return text
        
    if header_idx is not None:
        ref_start_idx = header_idx + 1
    elif unambiguous_indices:
        ref_start_idx = unambiguous_indices[0]
    else:
        return text
                
    body = '\n'.join(lines[:ref_start_idx])
    ref_block = '\n'.join(lines[ref_start_idx:])
    
    pattern = re.compile(
        r'^\s*'
        r'(?:[iI]n\s+)?'
        r'(?:[*\-•]\s*)?'
        r'(\[\d+\])'
        r'(?:,\s*|\.\s*|\s+)?',
        re.MULTILINE
    )
    normalized_ref_block = pattern.sub(r'\1 ', ref_block)
    
    if body:
        return body + '\n' + normalized_ref_block
    return normalized_ref_block

# ---------------------------------------------------------------------------
# Format Chat messages for Qwen
# ---------------------------------------------------------------------------
def format_messages_as_chat(messages: list, add_generation_prompt: bool = True) -> str:
    result = ""
    has_system = any(msg.get("role") == "system" for msg in messages)
    if not has_system:
        result += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("system", "user", "assistant"):
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        result += "<|im_start|>assistant\n"
    return result

# ---------------------------------------------------------------------------
# Subprocess Wrapper (adapted from openai_compatible_api_gateway.py)
# ---------------------------------------------------------------------------
_SENTINEL_RESPONSE = b"__RESPONSE__"
_SENTINEL_FINISH   = b"__FINISH__"
_SENTINEL_READY    = b"__READY__"
_SENTINEL_CACHED_PREFIX = b"__CACHED__:"

class SubprocessWrapper:
    def __init__(self, binary_path, model_path):
        self.process = None
        self.binary_path = binary_path
        self.model_path = model_path
        self.cached_len = 0
        self.prev_prompt = ""

    def _read_stderr(self):
        try:
            for line in self.process.stderr:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                self.stderr_log.append(decoded)
                if len(self.stderr_log) > 100:
                    self.stderr_log.pop(0)
        except Exception:
            pass

    def start(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                pass
        
        print_system(f"Launching C++ subprocess: {self.binary_path} {self.model_path} -")
        
        self.verbose = os.environ.get("DIFFKV_VERBOSE") == "1"
        stderr_dest = None if self.verbose else subprocess.PIPE
        self.stderr_log = []
        
        self.process = subprocess.Popen(
            [self.binary_path, self.model_path, "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_dest,
            text=False,
            bufsize=0,
            env=os.environ,
        )

        if not self.verbose:
            self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self.stderr_thread.start()

        # Drain stdout until __READY__
        buf = b""
        while True:
            if self.process.poll() is not None:
                print_error("Subprocess failed to start (exited early). Stderr logs:")
                for line in self.stderr_log:
                    print(f"  {line}")
                raise RuntimeError("Subprocess exited early")

            chunk = os.read(self.process.stdout.fileno(), 4096)
            if not chunk:
                print_error("Subprocess failed to start (stdout closed). Stderr logs:")
                for line in self.stderr_log:
                    print(f"  {line}")
                raise RuntimeError("Subprocess failed to start (stdout closed)")
            buf += chunk
            if _SENTINEL_READY in buf:
                break
        print_system("C++ Native process started and ready.")

    def stop(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                pass
            self.process = None

    def _write_stdin(self, text: str):
        # Write in chunks to avoid overflowing the OS pipe buffer (65536 bytes).
        # A research paper encoded as a single \n-escaped line can exceed 100KB;
        # writing that in one shot blocks until the reader drains — but the reader
        # (C++ binary) is waiting for a newline that hasn't arrived yet, causing
        # a deadlock/crash. Chunked writes + flush prevent this.
        data = text.encode("utf-8")
        chunk_size = 32768
        offset = 0
        while offset < len(data):
            end = min(offset + chunk_size, len(data))
            self.process.stdin.write(data[offset:end])
            self.process.stdin.flush()
            offset = end

    def _is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ensure_alive(self):
        if not self._is_alive():
            print_warning("C++ process died — restarting...")
            self.start()

    def query_stream_into_queue(self, prompt: str, max_tokens: int,
                                out_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop,
                                cached_len: int = 0):
        self.ensure_alive()
        
        single_line = prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
        if cached_len > 0:
            stdin_payload = f"__CACHED__:{cached_len}\n{single_line}\n"
        else:
            stdin_payload = single_line + "\n"

        try:
            self._write_stdin(stdin_payload)
        except Exception as e:
            loop.call_soon_threadsafe(out_queue.put_nowait, {"error": f"stdin write error: {e}"})
            loop.call_soon_threadsafe(out_queue.put_nowait, None)
            return

        # Phase 1: Wait for __RESPONSE__
        buf = b""
        try:
            while True:
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    loop.call_soon_threadsafe(out_queue.put_nowait, {"error": "process exited before __RESPONSE__"})
                    loop.call_soon_threadsafe(out_queue.put_nowait, None)
                    return
                buf += chunk
                if _SENTINEL_RESPONSE in buf:
                    after = buf.split(_SENTINEL_RESPONSE, 1)[1]
                    if after.startswith(b"\n"):
                        after = after[1:]
                    buf = after
                    break
        except Exception as e:
            loop.call_soon_threadsafe(out_queue.put_nowait, {"error": f"read error during prefill: {e}"})
            loop.call_soon_threadsafe(out_queue.put_nowait, None)
            return

        # Signal that prefill is done
        loop.call_soon_threadsafe(out_queue.put_nowait, {"prefill_done": True, "text": ""})

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def _extract_cached_from_remainder(remainder: bytes) -> int:
            idx = remainder.find(_SENTINEL_CACHED_PREFIX)
            if idx == -1:
                return -1
            end = remainder.find(b"\n", idx + len(_SENTINEL_CACHED_PREFIX))
            val_bytes = remainder[idx + len(_SENTINEL_CACHED_PREFIX):] if end == -1 else remainder[idx + len(_SENTINEL_CACHED_PREFIX):end]
            try:
                return int(val_bytes.strip())
            except Exception:
                return -1

        def _read_cached_from_stdout(proc_stdout) -> int:
            buf = b""
            for _ in range(5):
                try:
                    chunk = os.read(proc_stdout.fileno(), 100)
                    if not chunk:
                        break
                    buf += chunk
                    idx = buf.find(_SENTINEL_CACHED_PREFIX)
                    if idx != -1:
                        if b"\n" in buf[idx + len(_SENTINEL_CACHED_PREFIX):]:
                            break
                except Exception:
                    break
            return _extract_cached_from_remainder(buf)

        if buf:
            if _SENTINEL_FINISH in buf:
                parts = buf.split(_SENTINEL_FINISH, 1)
                final_text = decoder.decode(parts[0], final=True)
                if final_text:
                    loop.call_soon_threadsafe(out_queue.put_nowait, {"text": final_text})
                remainder = parts[1] if len(parts) > 1 else b""
                new_cached = _extract_cached_from_remainder(remainder)
                if new_cached == -1:
                    new_cached = _read_cached_from_stdout(self.process.stdout)
                if new_cached >= 0:
                    loop.call_soon_threadsafe(out_queue.put_nowait, {"cached_len": new_cached})
                loop.call_soon_threadsafe(out_queue.put_nowait, None)
                return
            
            text = decoder.decode(buf)
            if text:
                loop.call_soon_threadsafe(out_queue.put_nowait, {"text": text})

        # Phase 2: Stream tokens until __FINISH__
        # Use small read chunks (64 B) so each os.read() returns ~1-2 tokens at most,
        # avoiding the burst-pause-burst pattern caused by reading 4096 B at a time
        # (which lets 50-100 tokens pile up in the OS pipe buffer before flushing).
        accumulated = b""
        _CHUNK = 64  # small enough to get ~1 token per read at typical token sizes
        try:
            while True:
                chunk = os.read(self.process.stdout.fileno(), _CHUNK)
                if not chunk:
                    break
                accumulated += chunk
                if _SENTINEL_FINISH in accumulated:
                    parts = accumulated.split(_SENTINEL_FINISH, 1)
                    final_text = decoder.decode(parts[0], final=True)
                    if final_text:
                        loop.call_soon_threadsafe(out_queue.put_nowait, {"text": final_text})
                    remainder = parts[1] if len(parts) > 1 else b""
                    new_cached = _extract_cached_from_remainder(remainder)
                    if new_cached == -1:
                        new_cached = _read_cached_from_stdout(self.process.stdout)
                    if new_cached >= 0:
                        loop.call_soon_threadsafe(out_queue.put_nowait, {"cached_len": new_cached})
                    break
                tail_len = len(_SENTINEL_FINISH) + 4
                if len(accumulated) > tail_len:
                    safe = accumulated[:-tail_len]
                    remaining = accumulated[-tail_len:]
                    text = decoder.decode(safe)
                    if text:
                        loop.call_soon_threadsafe(out_queue.put_nowait, {"text": text})
                    accumulated = remaining
        except Exception as e:
            loop.call_soon_threadsafe(out_queue.put_nowait, {"error": f"read error during generation: {e}"})

        loop.call_soon_threadsafe(out_queue.put_nowait, None)

# ---------------------------------------------------------------------------
# Client Mode Helper (talks to running native gateway)
# ---------------------------------------------------------------------------
async def run_client_mode(args):
    api_url = args.api_url.rstrip('/')
    messages = []
    
    show_banner()
    
    print_system(f"Connecting to Native DiffKV Gateway at {COLOR_BOLD}{api_url}{COLOR_RESET}...")
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
                print_system("Conversation reset.")
                continue
            elif cmd in ["/paste", "/multiline"]:
                print_system("Raw paste mode active. Paste your text now (it will automatically submit when pasting completes).")
                pasted_text = await run_raw_paste()
                if not pasted_text.strip():
                    print_system("No text pasted. Cancelled.")
                    continue
                print_system(f"Pasted {len(pasted_text)} characters successfully.")
                user_prompt_stripped = pasted_text.strip()
            elif cmd == "/help":
                print_system("Available commands:")
                print("  /reset, /new       : Clear chat history and start a new session.")
                print("  /paste, /multiline : Enter multiline mode for pasting papers or long text.")
                print("  /stats             : Print details about the current chat history length.")
                print("  /srl               : Fetch Semantic Routing Layer (SRL) details.")
                print("  /exit, /quit       : Close the client.")
                continue
            elif cmd == "/stats":
                print_system("Analyzing conversation metrics...")
                print(f"\n{COLOR_BOLD}=== CLIENT RUNTIME METRICS ==={COLOR_RESET}")
                print(f"Model:              {args.model}")
                print(f"Server API URL:     {api_url}")
                print(f"Turns in History:   {len([m for m in messages if m['role']=='assistant'])}")
                print(f"Approx Cache Tokens: (Calculated on server from history)")
                print("==============================\n")
                continue
            elif cmd == "/srl":
                print_warning("Semantic Routing Layer (SRL) is only supported in the ACTIVE_RUNTIME Python implementation.")
                continue
            else:
                print_warning(f"Unknown command: {cmd}. Type /help for assistance.")
                continue

        messages.append({"role": "user", "content": user_prompt_stripped})
        
        # Prepare streaming request to Gateway
        payload = {
            "model": args.model,
            "messages": messages,
            "stream": True,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature
        }
        
        if hasattr(sys.stderr, "enable_buffering"):
            sys.stderr.enable_buffering()

        model_display = get_display_model_name(args.model)
        n_input_chars = len(user_prompt_stripped)

        # Show prefill progress bar (thread-based, ticks every 1s)
        prefill_bar = _make_prefill_bar(n_input_chars)
        
        start_time = time.time()
        first_token_time = None
        response_chunks = []
        
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
                        
                    import json
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            data_str = line[len("data: "):].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    if first_token_time is None:
                                        # First token — stop the prefill bar, print AI prompt
                                        first_token_time = time.time()
                                        _stop_prefill_bar(prefill_bar)
                                        prefill_bar = None
                                        print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)
                                    print(content, end="", flush=True)
                                    response_chunks.append(content)
                            except Exception:
                                pass

            # Stop bar even if no tokens came
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

# ---------------------------------------------------------------------------
# Direct Mode Helper (spawns native C++ binary directly)
# ---------------------------------------------------------------------------
async def run_direct_mode(args):
    show_banner()
    native_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Setup GGUF model path
    model_arg = args.model
    if model_arg.endswith(".gguf") and os.path.exists(model_arg):
        model_path = os.path.abspath(model_arg)
    elif "0.5b" in model_arg.lower():
        model_path = os.path.join(native_root, "qwen2.5-0.5b-instruct.gguf")
    elif "1.5b" in model_arg.lower():
        model_path = os.path.join(native_root, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        if not os.path.exists(model_path):
            model_path = os.path.join(native_root, "qwen2.5-1.5b-instruct-q8_0.gguf")
    elif os.path.exists(model_arg):
        model_path = os.path.abspath(model_arg)
    else:
        model_path = os.path.join(native_root, "qwen2.5-0.5b-instruct.gguf")

    # Force single-threaded BLAS/LAPACK/OpenMP to avoid audio driver preemption and
    # system-wide lag.
    #
    # WHY (do NOT remove): the SVD compressor runs on a worker thread that sets itself
    # to QOS_CLASS_BACKGROUND (async_compressor.cpp) so it yields to the UI/audio/decode.
    # But multi-threaded Accelerate/veclib spawns its OWN internal worker threads to run
    # sgesdd, and those do NOT inherit the background QoS — they run at default priority
    # and saturate every performance core, starving the rest of the system → the whole
    # Mac lags during long-prompt prefill. (ACTIVE_RUNTIME does not need this because MLX
    # runs the forward on the GPU and its SVD volume is far smaller — matching the env var
    # does NOT match the behavior.)
    #
    # For lag-SAFE parallel SVD, raise the number of *compressor workers* instead
    # (DIFFKV_COMPRESSOR_THREADS=N) — those threads are background-QoS and yield to the
    # system, unlike BLAS-internal threads.
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Setup environment variables
    os.environ["DIFFKV_PRESET"] = args.preset
    if args.preset == 'low':
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "512"
    elif args.preset == 'high':
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "2048"
    else:
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "512"

    # Lag-safe parallel SVD: more *background-QoS* compressor workers (these yield to
    # decode/UI, unlike BLAS-internal threads) so prefill compression keeps up without
    # saturating the system. Default to a modest 4; user/env override is respected.
    if "DIFFKV_COMPRESSOR_THREADS" not in os.environ:
        os.environ["DIFFKV_COMPRESSOR_THREADS"] = "4"

    os.environ["DIFFKV_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["DIFFKV_USE_GPU"] = "1" if args.use_gpu else "0"
    os.environ["DIFFKV_MICRO_BLOCK_SIZE"] = str(args.micro_block_size)
    os.environ["DIFFKV_BINARY_PATH"] = os.path.abspath(args.binary_path)
    os.environ["DIFFKV_MODEL_PATH"] = model_path
    if hasattr(args, 'context') and args.context is not None:
        os.environ["DIFFKV_MAX_CTX_TK"] = str(args.context)
        print_system(f"Context override: {args.context} tokens (--context flag)")
    if "DIFFKV_MPS_APPROXIMATE_ATTN" not in os.environ:
        os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
    # Correctness is guaranteed by the spin-wait inside execute_metal_attention:
    # after [commandBuffer commit], the C++ polls MTLCommandBufferStatus until
    # Completed before returning — no race condition, no METAL_SYNC env var needed.
    os.environ["DIFFKV_TEMPERATURE"] = str(args.temperature)

    os.environ["DIFFKV_TOP_P"] = str(args.top_p)
    os.environ["DIFFKV_REPETITION_PENALTY"] = str(args.repetition_penalty)

    # Start wrapper
    wrapper = SubprocessWrapper(os.environ["DIFFKV_BINARY_PATH"], os.environ["DIFFKV_MODEL_PATH"])
    try:
        wrapper.start()
    except Exception as e:
        print_error(f"Failed to start C++ process: {e}")
        print_error(f"Make sure you have built diffkv_native and that the binary exists at {os.environ['DIFFKV_BINARY_PATH']}")
        return

    messages = []
    
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
                print_system("Stopping C++ binary and exiting...")
                break
            elif cmd in ["/reset", "/new"]:
                messages = []
                wrapper.cached_len = 0
                wrapper.prev_prompt = ""
                # Restart the process to fully clear binary side cache
                wrapper.start()
                print_system("Session and C++ context reset.")
                continue
            elif cmd in ["/paste", "/multiline"]:
                print_system("Raw paste mode active. Paste your text now (it will automatically submit when pasting completes).")
                pasted_text = await run_raw_paste()
                if not pasted_text.strip():
                    print_system("No text pasted. Cancelled.")
                    continue
                print_system(f"Pasted {len(pasted_text)} characters successfully.")
                user_prompt_stripped = pasted_text.strip()
            elif cmd == "/help":
                print_system("Available commands:")
                print("  /reset, /new       : Reset conversation and clear C++ KV cache.")
                print("  /paste, /multiline : Enter multiline mode for pasting papers or long text.")
                print("  /stats             : Print active C++ binary cache size and configuration.")
                print("  /srl               : Fetch Semantic Routing Layer (SRL) details.")
                print("  /exit, /quit       : Shutdown C++ process and exit.")
                continue
            elif cmd == "/stats":
                print_system("Retrieving stats...")
                print(f"\n{COLOR_BOLD}=== DIRECT RUNTIME METRICS ==={COLOR_RESET}")
                print(f"Binary Path:     {os.environ['DIFFKV_BINARY_PATH']}")
                print(f"Model Path:      {os.environ['DIFFKV_MODEL_PATH']}")
                print(f"GPU Active:      {os.environ['DIFFKV_USE_GPU'] == '1'}")
                print(f"Chunk Size:      {os.environ['DIFFKV_PREFILL_CHUNK_SIZE']}")
                print(f"Cached Tokens:   {wrapper.cached_len} tokens")
                print("==============================\n")
                continue
            elif cmd == "/srl":
                print_warning("Semantic Routing Layer (SRL) is only supported in the ACTIVE_RUNTIME Python implementation.")
                continue
            else:
                print_warning(f"Unknown command: {cmd}. Type /help for assistance.")
                continue

        messages.append({"role": "user", "content": user_prompt_stripped})
        user_prompt_formatted = format_messages_as_chat(messages)

        # Prefix Match Check
        cached_len = 0


        if wrapper.cached_len > 0 and wrapper.prev_prompt:
            if user_prompt_formatted.startswith(wrapper.prev_prompt):
                cached_len = wrapper.cached_len
                print_system(f"Reusing {cached_len} cached tokens in the C++ binary KV pool!")

        # Query Stream Queue
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        def producer():
            wrapper.query_stream_into_queue(
                user_prompt_formatted, args.max_tokens, queue, loop,
                cached_len=cached_len
            )
            
        threading.Thread(target=producer, daemon=True).start()

        if hasattr(sys.stderr, "enable_buffering"):
            sys.stderr.enable_buffering()

        model_display = get_display_model_name(args.model)
        n_input_chars = len(user_prompt_formatted)

        # Show prefill progress bar (thread-based, ticks every 1s)
        prefill_bar = _make_prefill_bar(n_input_chars)

        start_time = time.time()
        first_token_time = None
        response_chunks = []
        new_cached_len = -1
        
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                
                if isinstance(item, dict):
                    if "error" in item:
                        _stop_prefill_bar(prefill_bar)
                        prefill_bar = None
                        print()
                        print_error(f"C++ binary error: {item['error']}")
                        messages.pop()
                        break
                    if "prefill_done" in item:
                        # Prefill completed — stop bar, print AI prompt header
                        _stop_prefill_bar(prefill_bar)
                        prefill_bar = None
                        print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)
                        continue
                    if "cached_len" in item:
                        new_cached_len = item["cached_len"]
                        continue
                    text = item.get("text", "")
                    if text:
                        if first_token_time is None:
                            # First token but no prefill_done — stop bar and show header
                            first_token_time = time.time()
                            _stop_prefill_bar(prefill_bar)
                            prefill_bar = None
                            print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)
                        print(text, end="", flush=True)
                        response_chunks.append(text)
                else:
                    text = str(item)
                    if text:
                        if first_token_time is None:
                            first_token_time = time.time()
                            _stop_prefill_bar(prefill_bar)
                            prefill_bar = None
                            print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)
                        print(text, end="", flush=True)
                        response_chunks.append(text)

            # Ensure bar always stops (no-token edge case)
            _stop_prefill_bar(prefill_bar)
            prefill_bar = None
            if not first_token_time:
                print(f"\n{COLOR_AI}{COLOR_BOLD}{model_display} >{COLOR_RESET} ", end="", flush=True)

            print() # end line

            if response_chunks:
                assistant_response = "".join(response_chunks)
                normalized_response = _normalize_references(assistant_response)
                
                # If reference normalization made edits, print a note or update response
                messages.append({"role": "assistant", "content": normalized_response})
                
                # Store prompt + normalized response as the new prev_prompt
                wrapper.prev_prompt = format_messages_as_chat(messages, add_generation_prompt=False)
                if new_cached_len >= 0:
                    wrapper.cached_len = new_cached_len

                # Print metrics
                end_time = time.time()
                est_tokens = max(1, sum(len(c) for c in response_chunks) // 4) # rough estimate
                total_duration = end_time - start_time
                ttft_str = f"{(first_token_time - start_time) * 1000:.1f}ms" if first_token_time else "N/A"
                if first_token_time and end_time > first_token_time:
                    speed = est_tokens / (end_time - first_token_time)
                    speed_str = f"{speed:.1f} tok/s"
                else:
                    speed_str = "N/A"
                    
                print_metrics(f"[Metrics] TTFT: {ttft_str} | Speed: {speed_str} | Generated: ~{est_tokens} tokens | Duration: {total_duration:.2f}s")
        finally:
            if hasattr(sys.stderr, "disable_and_flush"):
                sys.stderr.disable_and_flush()

    wrapper.stop()
    print_system("Engine stopped.")

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    sys.stdout = ThemedStdout(sys.stdout)
    sys.stderr = ThemedStdout(sys.stderr, buffer_stderr=True)
    parser = argparse.ArgumentParser(description="diffkv_native CLI: Interactive terminal interface for native C++ DiffKV.")
    
    # Mode selection
    parser.add_argument('--api-url', type=str, default=None,
                        help="Gateway API Base URL (e.g. http://localhost:8000/v1). If specified, runs in Client Mode.")
    
    # Model config
    parser.add_argument('--model', type=str, default='1.5b',
                        help="Model path or shortcut ('0.5b' or '1.5b') to locate local GGUF files. "
                             "'1.5b' resolves to the q4_k_m GGUF (q8_0 fallback).")
    
    # Executable config
    parser.add_argument('--binary-path', type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/diffkv_native')),
                        help="Path to built diffkv_native C++ executable.")
    parser.add_argument('--use-gpu', type=int, choices=[0, 1], default=1,
                        help="Enable GPU/Metal execution in C++ binary.")
    parser.add_argument('--micro-block-size', type=int, default=256,
                        help="Micro block size for KV compression.")
    parser.add_argument('--preset', type=str, choices=['low', 'mid', 'high'], default='mid',
                        help="Optimization preset (influences chunk prefill size). Use --context to override token budget.")
    parser.add_argument('--context', type=int, default=None,
                        help="Override context token budget (e.g. 22000 for a 20k-token prompt). "
                             "Overrides the preset's default limit. Leave unset to use preset default.")
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help="Max tokens to generate (pass a higher value for long-form output).")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument('--top-p', type=float, default=0.9,
                        help="Top-p sampling probability.")
    parser.add_argument('--repetition-penalty', type=float, default=1.15,
                        help="Repetition penalty parameter.")

    args = parser.parse_args()

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
