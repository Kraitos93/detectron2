import pandas as pd
import plotly.express as px
import sys
import os


def transform_model(model):
    if model=="soda_bottle_hq_new":
        return "SD1 dataset"
    elif model=="soda_bottle_hq_full_new":
        return "full dataset"
    elif model=="soda_bottle_hq_SD2_new":
        return "SD2 dataset"
    elif model=="soda_bottle_hq_SD3_new":
        return "SD3 dataset"
    elif model=="soda_bottle_chq_new":
        return "color change"
    elif model=="soda_bottle_bghq_new":
        return "background change"
    elif model=="soda_bottle_bgohq_new":
        return "background and orientation change"
    else:
        return "data"

if __name__ == "__main__":
    args = sys.argv
    model = args[1]
    df = pd.read_csv(os.path.join(os.getcwd(), 'csv_files', model + '.csv'))
    df_long=pd.melt(df, id_vars=['Epoch'], value_vars=['AP50', 'AP75', 'mAP'])
    fig = px.line(df_long, x='Epoch', y = 'value', title='Model quality (%s)' % (transform_model(model)))
    fig.write_image(os.path.join(os.getcwd(), 'csv_files', 'images', model + '.png'))
