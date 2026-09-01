# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare jsonl with field `input` and `outputs`.
{
    "index" int,
    "input": str,
    "outputs": [str],
}

python prepare.py \
    --save_dir ./ \
    --benchmark synthetic \
    --task niah_single_1 \
    --tokenizer_path tokenizer.model \
    --tokenizer_type nemo \
    --max_seq_length 4096 \
    --model_template_type base \
    --num_samples 10 \
"""
import os
import sys
import argparse
import importlib
import subprocess
import time
import yaml
from pathlib import Path
from template import Templates
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    # nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--num_samples", type=int, default=500, help='maximum number of samples we want to test')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--model_template_type", type=str, default='base', help='Options in `template.py`')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')
parser.add_argument("--prepare_for_ns", action='store_true')

args = parser.parse_args()

def main():
    start_time = time.time()
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    # Add templates
    assert args.model_template_type in Templates, print(f'{args.model_template_type} is not found in {Templates.keys()}')
    model_template = Templates[args.model_template_type]

    if args.prepare_for_ns:  
        from tokenizer import select_tokenizer
        TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)
        model_template_token = len(TOKENIZER.text_to_tokens(model_template))
        model_template = Templates['base']

    task_template = config['template']
    
    # Add answer prefix for all models
    answer_prefix = config['answer_prefix'] if 'answer_prefix' in config else ''

    config['template'] = model_template.format(task_template=task_template) + answer_prefix

    # Split task into multiple chunks 
    chunks = [(args.num_samples // args.chunk_amount) + (1 if i < args.num_samples % args.chunk_amount else 0) for i in range(args.chunk_amount)]
    num_samples = chunks[args.chunk_idx]
    pre_samples = sum(chunks[:args.chunk_idx])
    
    random_seed = args.random_seed + args.chunk_idx

    
    save_file = args.save_dir / args.task / f"{args.subset}.jsonl"
    file_exists = False
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            data = f.readlines()
        if len(data) == args.num_samples: file_exists = True



    if not file_exists:
        try:
            script = os.path.join(curr_folder, args.benchmark, f"{config['task']}.py")
            # LOCAL PORT FIX (argument list instead of a shell string). Upstream
            # builds this as one shell command, which on Windows breaks three
            # ways: bare `python` hits the Store alias stub; unquoted paths split
            # on the space in the repo directory ("Differential KV"); and
            # --template holds literal newlines, so cmd.exe tears it apart and
            # the generator runs with a collapsed template that never
            # substitutes {context} -- producing 26-token samples for a
            # 4096-token request, silently. Same script, same arguments, same
            # values; only the launch mechanism changes.
            command = [sys.executable, script,
                       "--save_dir", str(args.save_dir),
                       "--save_name", str(args.task),
                       "--subset", str(args.subset),
                       "--tokenizer_path", str(args.tokenizer_path),
                       "--tokenizer_type", str(args.tokenizer_type),
                       "--max_seq_length", str(args.max_seq_length),
                       "--tokens_to_generate", str(config['tokens_to_generate']),
                       "--num_samples", str(num_samples),
                       "--random_seed", str(random_seed)]
            for _k, _v in config['args'].items():
                command += [f"--{_k}", str(_v)]
            if args.remove_newline_tab:
                command += ["--remove_newline_tab"]
            if config['task'] == 'qa':
                command += ["--pre_samples", str(pre_samples)]
            command += ["--template", config['template']]
            if args.prepare_for_ns:
                command += ["--model_template_token", str(model_template_token)]

            print(" ".join(command[:6]), "...")
            result = subprocess.run(command,
                                    shell=False,
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            
            if result.returncode == 0:
                print("Output:")
                print(result.stdout)
            else:
                print("Error:")
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            print("Error output:", e.stderr)

        # A generation that produced nothing must not report success. Upstream
        # prints this unconditionally after the except block, so a dead child
        # reads exactly like a live one, and the empty directory it leaves is
        # taken downstream as "zero samples".
        _n = 0
        if os.path.exists(save_file):
            with open(save_file, encoding="utf-8") as _f:
                _n = sum(1 for _l in _f if _l.strip())
        if _n < 1:
            raise SystemExit(
                f"RULER generation FAILED for {args.task} @ {args.max_seq_length}: "
                f"{save_file} is missing or empty.")
        print(f"Prepare {args.task} with lines: {_n} to {save_file}")
        print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")
    else:
        print(f"Skip preparing {args.task} with lines: {args.num_samples} to {save_file} (file exists)")
    
if __name__ == '__main__':
    main()
