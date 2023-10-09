import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt

def get_eda_metrics(eda_signal, show = False):
    df, info = nk.eda_process(eda_signal, sampling_rate=1000, method='neurokit')
    tonic = df['EDA_Tonic'].mean()
    info_df = pd.DataFrame.from_dict(info, orient = 'columns')
    n_scr = info_df.shape[0]
    metrics = pd.DataFrame.from_dict(info, orient = 'columns').drop(columns = ['sampling_rate','SCR_Onsets','SCR_Peaks','SCR_Recovery']).mean().to_frame().T
    metrics.insert(0, 'N_SCR_per_min', n_scr)
    metrics.insert(0, 'Tonic', tonic)

    if show:
        nk.eda_plot(df)
        plt.show()

    return metrics
