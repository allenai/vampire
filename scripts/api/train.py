import argparse
from vampire.api import VampireTokenizer
from vampire.api import preprocess_data
from vampire.api import VampireModel
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=81)
    parser.add_argument('--kld-clamp', type=int, default=10000)
    parser.add_argument('--train-file', type=str)
    parser.add_argument('--dev-file', type=str, required=False)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--tokenizer', type=str, required=False, default='spacy')
    parser.add_argument('--serialization-dir', type=str)
    parser.add_argument('--seed', type=int, default=np.random.randint(0,10000000))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()
    
    tokenizer = VampireTokenizer(args.tokenizer)
    tokenizer.pretokenize(input_file=args.train_file,
                          output_file=args.train_file + ".tok.jsonl",
                          num_workers=20,
                          worker_tqdms=20)
    tokenizer.pretokenize(input_file=args.dev_file,
                          output_file=args.dev_file + ".tok.jsonl",
                          num_workers=20,
                          worker_tqdms=20)        
    preprocess_data(train_path=args.train_file + ".tok.jsonl",
                    dev_path=args.dev_file + ".tok.jsonl",
                    serialization_dir=args.data_dir,
                    tfidf=True, 
                    vocab_size=args.vocab_size)                               
    vampire = VampireModel.from_params(args.data_dir,
                                       args.kld_clamp,
                                       args.hidden_dim,
                                       args.vocab_size,
                                       ignore_npmi=False)
    vampire.fit(args.data_dir,
                args.serialization_dir,
                seed=args.seed,
                cuda_device=args.device)