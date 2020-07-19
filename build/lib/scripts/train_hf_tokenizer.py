from tokenizers import SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer, BertWordPieceTokenizer
import os
import json
import sys
import argparse
import glob
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=False, help="input text file, use '-' for stdin")
    parser.add_argument("--tokenizer-type", type=str, choices=['SP', 'BBPE', "CharBPE", "BERT"], help='one of BPE, CharBPE, BBPE, BERT')
    parser.add_argument("--serialization-dir", help='path to output model')
    parser.add_argument("--vocab-size", help='vocab size', type=int, default=52000)
    parser.add_argument("--min-frequency", help='min frequency', type=int, default=0)
    parser.add_argument("--special-tokens", help='special tokens', nargs="+", type=str, default=['<unk>'])
    args = parser.parse_args()
    # Initialize a tokenizer
    tokenizer = {
                'SP': SentencePieceBPETokenizer,
                "CharBPE": CharBPETokenizer,
                'BBPE': ByteLevelBPETokenizer,
                'BERT': BertWordPieceTokenizer
                }[args.tokenizer_type]
    tokenizer = tokenizer()
    # Then train it!
    special_tokens = args.special_tokens + ["<unk>"] if "<unk>" not in args.special_tokens else args.special_tokens
    tokenizer.train(args.input_file, vocab_size=args.vocab_size, special_tokens=special_tokens)
    if not os.path.isdir(args.serialization_dir):
        os.makedirs(args.serialization_dir)
    tokenizer.save(args.serialization_dir, 'tokenizer')
    with open(os.path.join(args.serialization_dir, "config.json"), "w+") as f:
        config = vars(args)
        json.dump(config, f)