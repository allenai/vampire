from tokenizers import BPETokenizer, ByteLevelBPETokenizer, BertWordPieceTokenizer
import os
import json
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="input text file")
    parser.add_argument("--tokenizer_type", type=str, choices=['BPE', 'BBPE', "BERT"], help='one of BPE, BBPE, BERT')
    parser.add_argument("--serialization_dir", help='path to output BPE model')
    parser.add_argument("--vocab_size", help='YTTM vocab size', type=int, default=10000)
    args = parser.parse_args()
    # Initialize a tokenizer
    
    tokenizer = {
                'BPE': BPETokenizer,
                'BBPE': ByteLevelBPETokenizer,
                'BERT': BertWordPieceTokenizer
                }[args.tokenizer_type]

    tokenizer = tokenizer()

    # Then train it!
    tokenizer.train([args.input_file], vocab_size=args.vocab_size)
    if not os.path.isdir(args.serialization_dir):
        os.makedirs(args.serialization_dir)
    tokenizer.save(args.serialization_dir, 'tokenizer')
    with open(os.path.join(args.serialization_dir, "config.json"), "w+") as f:
        config = vars(args)
        json.dump(config, f)