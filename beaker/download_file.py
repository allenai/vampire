from beaker import beaker_client
import argparse
import os

TOKEN = os.environ['BEAKER_CLIENT_TOKEN']
ADDRESS = "http://beaker-internal.allenai.org"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        help='name of dataset (e.g. ds_nc05x1bc54o5)',
                        required=True)
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        help='filename',
                        required=True)
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='output dir',
                        required=True)
    args = parser.parse_args()

    client = beaker_client.Client(ADDRESS, token=TOKEN)

    ds = beaker_client.Dataset(client, args.dataset)

    output = {}
    for file_ in ds.files:
        output[file_.path.replace('/', '', 1)] = file_
    
    files = ", ".join(output.keys())
    if not output.get(args.file):
        raise argparse.ArgumentTypeError(f"{args.file} not available. Available files: {files}")
    
    print(f"getting {args.file}...")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output[args.file].download(output_dir=args.output_dir)