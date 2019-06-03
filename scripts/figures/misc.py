"""
miscellaneous scripts for the paper
"""

# sample over vampires
import glob
import json
configs = glob.glob("logs/vampire_yahoo_search/*/trial/config*")
configs = [(x, json.load(open(x, 'r'))) for x in configs]
hidden_dims = [(x, y['model']['vae']['encoder']['hidden_dims'][0]) for x,y in configs]
hidden_dims = [(x.replace('/trial/config.json', ''), y) for x,y in hidden_dims]
hidden_dims = [ (x, y) for x,y in hidden_dims if x + "/trial" in glob.glob(x + "/*")]
hidden_dims = [ (x, y) for x,y in hidden_dims if x + "/trial/model.tar.gz" in glob.glob(x + "/trial/*")]
hidden_dims = [" ".join([str(y) for y in x]) for x in hidden_dims]


# join on VAMPIRE search
import pandas as pd
df = pd.read_json("/home/suching/vampire/logs/hatespeech_classifier_search/results.jsonl", lines=True)
df1 = pd.read_json("/home/suching/vampire/logs/vampire_hatespeech_search/results.jsonl", lines=True)
df['vampire_directory'] = df['model.input_embedder.token_embedders.vampire_tokens.model_archive'].str.replace('model.tar.gz', '')
master = df.merge(df1, left_on = 'vampire_directory', right_on='directory')
master.to_json("hyperparameter_search_results/hatespeech_vampire_classifier_search.jsonl", lines=True, orient='records')

