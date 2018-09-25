import numpy as np
import pandas as pd


def process_data(data, T):
    '''
    data: df[total_T, len(x)]  ->  processed_data: (dict(train=train_x, val=val_x, test=test_x),
                                                    dict(train=train_y, val=val_y, test=test_y))
    '''

    def split_data(data, T, val_size=0.1, test_size=0.1):
        '''
        data: df[total_T, len(x)]  ->  splited_data : (df[train_T, len(x)], df[val_T, len(x)], df[test_T, len(x)])
        '''
        ntest = int(round(len(data) * (1 - test_size)) - T)
        nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)) - T)
        df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
        splited_data = (df_train, df_val, df_test)
        return splited_data

    def rnn_data(data, T, labels=False):
        '''
        data: df[total_T, len(x)]  ->  rnn_data: array[total_T-T, T, len(x)] if labels=False
        data: df[total_T, len(x)]  ->  rnn_data: array[total_T-T, 1, len(y)] if labels=True
        '''
        rnn_df = []
        for i in range(len(data) - T):
            if labels:
                try:
                    rnn_df.append(data.iloc[i + T].values)
                except AttributeError:
                    rnn_df.append(data.iloc[i + T])
            else:
                data_ = data.iloc[i:i + T]
                data_ = data_.values
                rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
        rnn_data = np.array(rnn_df)
        return rnn_data

    def prepare_data(data, T, labels=False, val_size=0.05, test_size=0.05):
        df_train, df_val, df_test = split_data(data, T, val_size, test_size)
        return (rnn_data(df_train, T, labels=labels),
                rnn_data(df_val, T, labels=labels),
                rnn_data(df_test, T, labels=labels))

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data, T)
    train_y, val_y, test_y = prepare_data(data, T, labels=True)
    processed_data = (dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y))
    return processed_data

# # test process_data()
# data = np.linspace(0, 10, 1000)
# data = np.sin(data)
# p_data = process_data(data, 5)
# print(p_data[0]['train'].shape)
# print(p_data[1]['train'].shape)
# print(p_data[0]['test'].shape)
# print(p_data[1]['test'].shape)

def generate_batches(data, batch_size):
    len_data = len(data[0])
    num_batch = len_data // batch_size
    for i in range(num_batch+1):
        x = data[0][batch_size*i:batch_size*(i+1)]
        y = data[1][batch_size*i:batch_size*(i+1)]
        yield (x, y)

def generate_epochs(data, num_epochs, batch_size):
    for i in range(num_epochs):
        yield generate_batches(data, batch_size)


