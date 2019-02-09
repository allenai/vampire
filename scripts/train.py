import os
import argparse
import subprocess
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='path to directory containing reference counts and vocab',
                        required=True)
    parser.add_argument('-x',
                        '--seed',
                        type=int,
                        help='seed',
                        required=False,
                        default=42)
    parser.add_argument('-c', '--config', type=str, help='training config', required=True)
    parser.add_argument('-s', '--serialization_dir', type=str, help='model serialization directory', required=True)
    
    args = parser.parse_args()

    os.environ['SEED'] = str(args.seed)

    if os.path.exists(args.serialization_dir) and args.override:
        print(f"overriding {args.serialization_dir}")
        shutil.rmtree(args.serialization_dir)

    allennlp_command = [
                "allennlp",
                "train",
                "--include-package",
                "vae.modules.token_embedders.vae_token_embedder",
                "--include-package",
                "vae.models.classifier",
                "--include-package",
                "vae.models.unsupervised",
                "--include-package",
                "vae.models.joint_semi_supervised",
                "--include-package",
                "vae.data.dataset_readers.semisupervised_text_classification_json",
                "--include-package",
                "vae.data.tokenizers.regex_and_stopword_filter",
                "--include-package",
                "vae.common.allennlp_bridge",
                args.config,
                "-s",
                args.serialization_dir,
            ]
    subprocess.run(" ".join(allennlp_command), shell=True, check=True)
