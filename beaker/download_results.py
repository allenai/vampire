import argparse
import os
import sys

import boto3

from beaker import beaker_client

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
TOKEN = os.environ['BEAKER_CLIENT_TOKEN']
ADDRESS = "http://beaker-internal.allenai.org"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        help='name of dataset (e.g. ds_nc05x1bc54o5)',
                        required=True)
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='output dir',
                        required=True)
    parser.add_argument('-s',
                        '--s3',
                        type=str,
                        help='send output to s3 at this bucket',
                        required=False)
    args = parser.parse_args()

    client = beaker_client.Client(ADDRESS, token=TOKEN)

    ds = beaker_client.Dataset(client, args.dataset)

    output = {}
    for file_ in ds.files:
        output[file_.path.replace('/', '', 1)] = file_
    
    files = ", ".join(output.keys())
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    for file in ['model.tar.gz', 'vocabulary/vae.txt', 'vocabulary/vae.bgfreq.json']:
        print(f"getting {file}...")
        output[file].download(output_dir=args.output_dir)

    if args.s3:
        bucket_name = args.s3
        try:
            s3_client = boto3.client('s3', region_name = 'us-west-2')
            bucket = s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'us-west-2'})
        except s3_client.exceptions.BucketAlreadyOwnedByYou:
            s3 = boto3.resource('s3', region_name = 'us-west-2')
            bucket = s3.Bucket(args.s3)

        for file in ['model.tar.gz', 'vae.txt', 'vae.bgfreq.json']:
            print(f"Uploading {file} to Amazon S3 bucket {bucket_name}")
            bucket.upload_file(os.path.join(args.output_dir, file), file)
