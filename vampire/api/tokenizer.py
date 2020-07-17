import argparse
import json
import os
import sys
import urllib
from typing import Any, Iterable, List, Dict, Optional
import time
import spacy
from spacy.tokenizer import Tokenizer
from tokenizers import (BertWordPieceTokenizer, ByteLevelBPETokenizer,
                        CharBPETokenizer, SentencePieceBPETokenizer)
from tqdm import tqdm
from transformers import AutoTokenizer
import multiprocessing
import numpy as np
import shutil
import uuid

EOS = 50256


def split_docs(tokens: np.array) -> List[np.array]:
    idx = np.nonzero(tokens == EOS)[0]
    docs = np.split(tokens, idx)
    docs = [doc[1:] for doc in docs if len(doc) > 1]
    return docs

def load_huggingface_tokenizer(tokenizer_path: str) -> (Any, bool):
    if os.path.isdir(tokenizer_path):
        with open(os.path.join(tokenizer_path, 'config.json'), 'r') as f:
                config = json.load(f)
        tokenizer_type = config['tokenizer_type']
        tokenizer = {'SP': SentencePieceBPETokenizer,
                     'BBPE': ByteLevelBPETokenizer,
                     'CharBPE': CharBPETokenizer,
                     'BERT': BertWordPieceTokenizer}[tokenizer_type]
        if tokenizer_type in ['SP', 'BBPE', 'CharBPE']:
            vocab_file = [x for x in os.listdir(tokenizer_path) if 'vocab.json' in x][0]
            merges_file = [x for x in os.listdir(tokenizer_path) if 'merges.txt' in x][0]
            tokenizer = tokenizer(vocab_file=os.path.join(tokenizer_path, vocab_file),
                                merges_file=os.path.join(tokenizer_path, merges_file))
        else:
            vocab_file = [x for x in os.listdir(tokenizer_path) if 'vocab.txt' in x][0]
            tokenizer = tokenizer(vocab_file=os.path.join(tokenizer_path, vocab_file))
        is_transformers_tokenizer = False
    else:
        is_transformers_tokenizer = True
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer, is_transformers_tokenizer

def load_tokenizer(tokenizer: str) -> Any:
    is_transformers_tokenizer = False
    if tokenizer == "spacy":
        nlp = spacy.load('en_core_web_sm')
        tokenizer = Tokenizer(nlp.vocab)
    elif tokenizer == 'scispacy':
        nlp = spacy.load('en_core_sci_sm')
        tokenizer = Tokenizer(nlp.vocab)
    elif tokenizer == 'whitespace':
        tokenizer = lambda x: x.split()
    else:
        tokenizer, is_transformers_tokenizer = load_huggingface_tokenizer(tokenizer)
    return tokenizer, is_transformers_tokenizer



class TokenizerManager(object):
    def __init__(self, tokenizer_type: str) -> None:
        self.tokenizer_type = tokenizer_type
        self.tokenizer, self.is_transformers_tokenizer = load_tokenizer(tokenizer_type)

    def _tokenize_line(self, line: str, lower: bool, remove_wordpiece_indicator: bool, return_ids: bool) -> str:
        if self.tokenizer_type in ['spacy', 'scispacy']:
            tokens = list(map(str, self.tokenizer(line)))
        elif self.tokenizer_type in ['whitespace']:
            tokens = self.tokenizer(line)
        else:
            if self.is_transformers_tokenizer:
                if return_ids:
                    tokens = self.tokenizer.tokenize(line, add_special_tokens=True)
                    tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                if return_ids:
                    tokens = self.tokenizer.encode(line).ids
                else:
                    tokens = self.tokenizer.encode(line).tokens
        if return_ids:
            tokens = list(map(str, tokens))
        if remove_wordpiece_indicator:
            tokens = [token.lstrip("\u2581").replace('</w>', '') for token in tokens]
        joined_tokens = ' '.join(tokens)
        if lower:
            joined_tokens = joined_tokens.lower()
        return joined_tokens
    

    def tokenize(self, line: str, lower: bool=False, is_json: bool=False, print_it: bool=False, remove_wordpiece_indicator=False, return_ids=False) -> Optional[str]:
        if not line.isspace():
            if is_json:
                orig_json = json.loads(line)
                line = orig_json['text']
            else:
                orig_json = None
                line = line.strip()
            tokens = self._tokenize_line(line, lower, remove_wordpiece_indicator, return_ids)
            if orig_json:
                orig_json['text'] = tokens
                tokens = json.dumps(orig_json)
            if print_it:
                print(tokens)
            return tokens
        else:
            return None



def count_lines(input_file: str) -> int:
    with (open(input_file, "r")) as f: 
        num_lines = 0
        for line in f:
            num_lines += 1
    return num_lines


class Consumer(multiprocessing.Process):

    def __init__(self,
                 tokenizer: Any,
                 task_queue: multiprocessing.JoinableQueue,
                 result_queue: multiprocessing.JoinableQueue,
                 num_consumers: int,
                 num_tasks: int,
                 worker_id: int,
                 pos: int,
                 add_tqdm: bool,
                 num_add_tqdm: int,
                 silent: bool) -> None:
        multiprocessing.Process.__init__(self)
        self.tokenizer = tokenizer
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.add_tqdm = add_tqdm
        self.worker_id = worker_id
        if add_tqdm:
            self.pbar = tqdm(total=num_tasks // num_consumers, position=pos + 1, disable=silent, leave=False)
            self.pbar.set_description(f"Tokenizer {worker_id}")

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                self.result_queue.put(None)
                break
            elif next_task == "newline":
                self.task_queue.task_done()
                self.result_queue.put("newline")
            else:
                answer = next_task(self.tokenizer)
                self.task_queue.task_done()
                self.result_queue.put(answer)
            if self.add_tqdm:
                self.pbar.update()


class Writer(multiprocessing.Process):

    def __init__(self,
                 result_queue: multiprocessing.JoinableQueue,
                 num_writers: int,
                 num_tasks: int,
                 output_dir: str,
                 worker_id: int,
                 pos: int,
                 add_tqdm: bool,
                 num_add_tqdm: int,
                 silent: bool) -> None:
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.add_tqdm = add_tqdm
        self.worker_id = worker_id
        self.file_ = os.path.join(output_dir, str(worker_id) + ".tmp")
        if add_tqdm:
            self.pbar = tqdm(total=num_tasks // num_writers, position=pos + 1 + num_add_tqdm, disable=silent, leave=False)
            self.pbar.set_description(f"Writer {worker_id}")

    def run(self):
        proc_name = self.name
        with open(self.file_, 'w+') as f:
            while True:
                next_task = self.result_queue.get()
                if next_task is None:
                    # Poison pill means shutdown
                    self.result_queue.task_done()
                    break
                elif next_task == "newline":
                    f.write("\n")
                else:
                    f.write(next_task + "\n")
                self.result_queue.task_done()
                if self.add_tqdm:
                    self.pbar.update()

class Task(object):
    def __init__(self, line: str, is_json: bool, lower: bool, return_ids: bool, remove_wordpiece_indicator: bool) -> None:
        self.line = line
        self.is_json = is_json
        self.lower = lower
        self.return_ids = return_ids
        self.remove_wordpiece_indicator = remove_wordpiece_indicator

    def __call__(self, tokenizer: Any):
        tokens = tokenizer.tokenize(self.line,
                                    is_json=self.is_json,
                                    lower=self.lower,
                                    print_it=False,
                                    remove_wordpiece_indicator=self.remove_wordpiece_indicator,
                                    return_ids=self.return_ids)
        return tokens

class MultiprocessTokenizer(object):

    def __init__(self, tokenizer: str, num_workers: int, worker_tqdms: int, silent: bool) -> None:
        self.tokenizer = TokenizerManager(tokenizer) 
        self.num_workers = num_workers
        self.worker_tqdms = worker_tqdms
        self.task_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.JoinableQueue()
        self.silent = silent
        self.identifier = uuid.uuid4().hex

    def get_vocab_size(self):
        return self.tokenizer.tokenizer.get_vocab_size()

    def get_vocab(self):
        return self.tokenizer.tokenizer.get_vocab()

    def launch_producers_and_consumers(self, num_tasks, output_dir):
        if self.worker_tqdms == -1:
            tqdms = np.array([1] * self.num_workers)
        else:
            tqdms = np.array([0] * self.num_workers)
            tqdms[:self.worker_tqdms] = 1

        num_add_tqdm = np.sum(tqdms)

        tokenizers = [
            Consumer(tokenizer=self.tokenizer,
                    task_queue=self.task_queue,
                    result_queue=self.result_queue, 
                    num_consumers=self.num_workers,
                    num_tasks=num_tasks,
                    worker_id=worker_id,
                    pos=pos,
                    add_tqdm=add_tqdm,
                    num_add_tqdm=num_add_tqdm,
                    silent=self.silent)
            for pos, (worker_id, add_tqdm) in enumerate(zip(range(self.num_workers), tqdms))
        ]

        writers = [
            Writer(result_queue=self.result_queue,
                num_writers=self.num_workers,
                num_tasks=num_tasks,
                output_dir=output_dir,
                worker_id=worker_id,
                pos=pos,
                add_tqdm=add_tqdm,
                num_add_tqdm=num_add_tqdm,
                silent=self.silent)
            for pos, (worker_id, add_tqdm) in enumerate(zip(range(self.num_workers), tqdms))
        ]
        return tokenizers, writers

    def mk_tmpdir(self, input_file):
        tmp_dir = f"/tmp/tokenization/{self.identifier}"
        if os.path.isdir(tmp_dir):
            for file in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, file))
            os.rmdir(tmp_dir)
        os.makedirs(tmp_dir)
        return tmp_dir
    

    def enqueue(self, input_file, num_tasks, json, lower, silent, return_ids, remove_wordpiece_indicator):
        # Enqueue jobs
    
        with open(input_file, "r") as f:
            for line in tqdm(f, total=num_tasks, position=0, leave=False, desc="Enqueue", disable=silent):
                if not line.isspace():
                    self.task_queue.put(Task(line, json, lower, return_ids, remove_wordpiece_indicator))
                else:
                    self.task_queue.put("newline")
        # Add a sentinel for each consumer 
        for i in range(self.num_workers):
            self.task_queue.put(None)
    
    def write_to_file(self, output_dir, output_file):
        shards = [os.path.join(output_dir, file_) for file_ in os.listdir(output_dir) if ".tmp" in file_]
        if os.path.isfile(output_file):
            os.remove(output_file)

        with open(output_file, 'a+') as outfile:
            for fname in tqdm(shards, desc="File Merge", disable=self.silent):
                with open(fname, "r") as infile:
                    for line in infile:
                        outfile.write(line)

    def run(self, input_file: str, output_file: str, json: bool=False, lower: bool=False, return_ids: bool=False, remove_wordpiece_indicator: bool=False):
        tmp_dir = self.mk_tmpdir(input_file)
        num_lines = count_lines(input_file)
        tokenizers, writers = self.launch_producers_and_consumers(num_lines, tmp_dir)
        for worker in tokenizers:
            worker.start()
        for worker in writers:
            worker.start()
        self.enqueue(input_file, num_lines, json, lower, self.silent, return_ids, remove_wordpiece_indicator)

        # Wait for all of the tasks to finish
        self.task_queue.join()
        self.result_queue.join()
        self.write_to_file(tmp_dir, output_file)

class VampireTokenizer(object):

    def __init__(self, tokenizer: str="spacy") -> None:
        self._tokenizer = tokenizer

    def pretokenize(self,
                    input_file: str,
                    output_file: str,
                    num_workers: int=1,
                    worker_tqdms: int=1,
                    is_json: bool=False,
                    lower: bool=False,
                    return_ids: bool=False,
                    remove_wordpiece_indicators: bool=False,
                    silent: bool=False) -> None:
        tok = MultiprocessTokenizer(self._tokenizer,
                                    num_workers,
                                    worker_tqdms,
                                    silent)
        tok.run(input_file, output_file, is_json, lower, return_ids, remove_wordpiece_indicators)
        return