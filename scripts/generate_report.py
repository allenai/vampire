import itertools
import pandas as pd


def get_stats(basic, vae, glove, in_elmo,  throttle, df):
    sub_df = df.loc[(df.env_ADD_BASIC == basic) & (df.env_ADD_VAE == vae) & (df.env_ADD_GLOVE == glove) & (df.env_ADD_ELMO == in_elmo) & (df.env_THROTTLE == throttle)]
    sub_df = sub_df.drop_duplicates(subset=["env_SEED"])
    if sub_df.env_SEED.any():
        cur_throttle = sub_df.env_THROTTLE.unique()[0]
        out_df = sub_df[['metric_best_validation_accuracy']]
        out_df.columns = [cur_throttle]
        out_df.index = sub_df.env_SEED
        out_df = out_df.transpose()
        out_df['mean_acc'] = out_df.ix[cur_throttle].mean()
        out_df['std'] = out_df.ix[cur_throttle].std()
        out_df['median'] = out_df.ix[cur_throttle].median()
        out_df['max_'] = out_df.ix[cur_throttle].max()
        out_df['min_'] = out_df.ix[cur_throttle].min()
        out_df['+vae'] = sub_df.env_ADD_VAE.unique()[0]
        out_df['+glove'] = sub_df.env_ADD_GLOVE.unique()[0]
        out_df['+basic'] = sub_df.env_ADD_BASIC.unique()[0]
        out_df['+in-domain-elmo-frozen'] = sub_df.env_ADD_ELMO.unique()[0]
        # out_df['+out-domain-elmo-frozen'] = sub_df.env_ADD_OUT_DOMAIN_ELMO.unique()[0]
    else:
        out_df = None
    
    return out_df

if __name__ == '__main__':
    # df = pd.read_csv("/Users/suching/Downloads/Yahoo_final_1 (1).csv")
    df = pd.read_csv("/Users/suching/Downloads/AGNews_final_1 (2).csv")
    # df_elmo_ood = pd.read_csv("/Users/suching/Downloads/AGNews_final_1_OOD_ELMO.csv")
    # df.rename(columns={'env_ADD_ELMO': 'env_ADD_IN_DOMAIN_ELMO'})
    # df_elmo_ood.rename(columns={'env_ADD_ELMO': 'env_ADD_OUT_DOMAIN_ELMO'})

    # df = pd.concat([df, df_elmo_ood], 0)
    # df = pd.read_csv("/Users/suching/Downloads/IMDB_final_1.csv")
    master = pd.DataFrame()
    basic_options = [1, 0]
    vae_options = [1, 0]
    glove_options = [1, 0]
    elmo_options = [1, 0]
    throttle_options = [200, 500, 2500, 5000, 10000]

    options = [basic_options, vae_options, glove_options, elmo_options, throttle_options]

    for combo in list(itertools.product(*options)):
        stats = get_stats(combo[0], combo[1], combo[2], combo[3], combo[4], df)
        if stats is not None:
            master = pd.concat([master, stats], 0)
    # assert master.shape[0] == 4 * len(throttle_options)
    master.to_excel("/Users/suching/Github/vae/results/excel/agnews.xlsx")
    print(master)