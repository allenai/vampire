import typing
from typing import  Any
import json
import os
from multiprocessing import Process, Queue
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from spacy.tokenizer import Tokenizer
import spacy
from tqdm.auto import tqdm
import time

nlp = spacy.load("en")


class TokenizingWorker(Process):
    def __init__(
        self,
        pbar: Any,
        is_json: bool,
        queue_in: Queue, # Queue where text comes in
        queue_out: Queue, #Queue where tokens go
        tokenizer_type: str = "just_spaces",

    ):
        super(TokenizingWorker, self).__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.is_json = is_json
        self.pbar = pbar
        if tokenizer_type == "just_spaces":
            tokenizer = SpacyWordSplitter()
            self.tokenizer = lambda text: list(map(str, tokenizer.split_words(text)))
        elif tokenizer_type == "spacy":
            tokenizer = Tokenizer(nlp.vocab)
            self.tokenizer = lambda text: list(map(str, tokenizer(text)))

    def run(self):
        for line in iter(self.queue_in.get, None):

            if self.is_json:
                text = json.loads(line)["text"]
            else:
                text = line
            tokens = self.tokenizer(text)
            while self.queue_out.full():
                time.sleep(0.01)
            self.queue_out.put(" ".join(tokens),block=False,)
            self.pbar.update()




def multi_proc_data_loader(data_path: str,  tokenizer_type: str = "just_spaces"):
    num_processes = max(1, os.cpu_count() - 1)
    queue_in = Queue()
    queue_out = Queue(maxsize=10000)
    workers =[]
    is_json = data_path.endswith(".jsonl") or data_path.endswith(".json")


    pbar = tqdm()

    for _ in range(num_processes):  # minus one if the main processus is CPU intensive
        worker = TokenizingWorker(
            pbar=pbar,
            is_json=is_json, queue_in=queue_in, queue_out=queue_out,tokenizer_type=tokenizer_type
        )
        workers.append(worker)
        worker.start()
    with (open(data_path, "r")) as f:
        for line in f:
            queue_in.put(line)

    for worker in workers:
        #ensure each worker gets a None which tells it to stop
        queue_in.put(None)
    alive = any(map(lambda x:x.is_alive(),workers))
    res=[]
    while alive:
        while not queue_out.empty():
             tokens  =queue_out.get(block=False)
             res.append(tokens)
        alive = any(map(lambda x: x.is_alive(), workers))
        if alive:
            time.sleep(0.01)
    return  res





