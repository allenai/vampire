import argparse
from vampire.api import VampireTokenizer
from vampire.api import preprocess_data
from vampire.api import VampireModel
import json
import torch
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=str,  required=True)
    parser.add_argument('--tokenizer', type=str, required=False, default='spacy')
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str,  required=True)
    parser.add_argument('--device', type=int, default=-1)

    args = parser.parse_args()
    tokenizer = VampireTokenizer(args.tokenizer)                            
    vampire = VampireModel.from_pretrained(pretrained_archive_path=args.archive,
                                           cuda_device=args.device,
                                           for_prediction=True)
    ids = []
    vectors = []
    for line in tqdm(args.input_file):
        line = json.loads(line)
        out = vampire.predict(line, scalar_mix=True)
        ids.append(line['index'])
        vectors.append(out)
    torch.save((torch.cat(ids,0).cpu(), torch.cat(vectors, 0).cpu()), args.output_file)