from vampire.api.tokenizer import VampireTokenizer
import argparse
import multiprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", help='path to input file', required=True)
    parser.add_argument("--tokenizer", help='tokenizer type (spacy or path to pretrained tokenizer)', required=True)
    parser.add_argument("--json", action='store_true', help='is input file json?')
    parser.add_argument("--lower", action='store_true', help='lowercase?')
    parser.add_argument("--silent", action='store_true', help='if set, will silence TQDM')
    parser.add_argument("--num-workers", type=int, default=1, help='how many workers?')
    parser.add_argument("--worker-tqdms", type=int, default=-1, help='how many per-worker tqdms to display?')
    parser.add_argument("--output-file", default=None, help='path to output file')
    parser.add_argument("--ids",  action='store_true', help='tokenize to ids')
    parser.add_argument("--remove-wordpiece-indicator",  action='store_true', help='if set, will remove wordpiece indicator when tokenizing')

    args = parser.parse_args()
    args_dict = vars(args)
    if args_dict['num_workers'] == -1:
        args_dict['num_workers'] = multiprocessing.cpu_count()
    input_file = args_dict.pop('input_file')
    output_file = args_dict.pop('output_file')
    is_json = args_dict.pop('json')
    lower = args_dict.pop('lower')
    ids = args_dict.pop('ids')
    remove_wordpiece_indicator = args_dict.pop('remove_wordpiece_indicator')
    tokenizer = VampireTokenizer(**args_dict)
    tokenizer.pretokenize(input_file, output_file, is_json, lower, ids, remove_wordpiece_indicator)