import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./data/val_data.tsv', sep='\t')
    for text in df.values:
        print(text)
        print()
    print('Datapoints:', len(df))