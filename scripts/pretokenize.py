from tokenizers import SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer, BertWordPieceTokenizer
import sys
import os
import argparse
import json
from tqdm import tqdm
from vampire.common.util import load_huggingface_tokenizer
import spacy
from spacy.tokenizer import Tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help='tokenizer type (spacy or path to pretrained tokenizer)')
    parser.add_argument("--json", action='store_true', help='is input file json?')
    parser.add_argument("--lower", action='store_true', help='lowercase?')
    parser.add_argument("--silent", action='store_true', help='if set, will silence TQDM')

    args = parser.parse_args()

    if args.tokenizer == "spacy":
        nlp = spacy.load('en_core_web_sm')
        tokenizer = Tokenizer(nlp.vocab)
    else:
        tokenizer = load_huggingface_tokenizer(args.tokenizer)

    for line in tqdm(sys.stdin, disable=args.silent):        
        if args.json:
            orig_json = json.loads(line)
            line = orig_json['text']
        if args.tokenizer == 'spacy':
            tokens = list(map(str, tokenizer(line)))
        else:
            tokens = tokenizer.encode(line).tokens
        line = ' '.join(tokens)
        if args.lower:
            line = line.lower()
        if args.json:
            orig_json['text'] = line
            print(json.dumps(orig_json))
        else:
            print(line)
