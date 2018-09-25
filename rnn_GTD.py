import numpy as np
import tensorflow as tf
import time
from LayerNormalizedLSTMCell import LayerNormalizedLSTMCell
import matplotlib.pyplot as plt
import pandas as pd
from utils import process_data, generate_epochs, generate_batches

# # sin data for testing:
# data = np.linspace(0, 100, 10000)
# data = np.sin(data)
# data_type = 'Sin'

# GTD data for real prediction:
df = pd.read_csv('data/day_count.csv')
df = df.sort_values(by='day')
df.info(verbose=True)
day = df['day']
count = df['count']
data = np.array(count.values)
data_type = 'GTD'

num_steps = 30
input_size = 1
output_size = 1
batch_size = 64
rnn_state_size = 32
dense_state_size = [10, 1]
num_classes = 1
learning_rate = 1e-4
keep_prob = 1
num_layers = 1

data = process_data(data, T=num_steps)
data_train = (data[0]['train'], data[1]['train'])
data_test = (data[0]['test'], data[1]['test'])


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def build_multilayer_lnlstm_graph_with_dynamic_rnn(
        cell_type=None,
        input_size=input_size,
        output_size=output_size,
        rnn_state_size=rnn_state_size,
        dense_state_size=dense_state_size,
        num_steps=num_steps,
        num_layers=num_layers,
        learning_rate=learning_rate,
        keep_probs=[1.0, 1.0]
):
    reset_graph()
    x = tf.placeholder(tf.float32, [None, num_steps, input_size], name='x')
    y = tf.placeholder(tf.float32, [None, output_size], name='y')
    if cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(rnn_state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(rnn_state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(rnn_state_size)
    elif cell_type == 'BasicRNN':
        cell = tf.nn.rnn_cell.BasicRNNCell(rnn_state_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(rnn_state_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_probs[0])
    if num_layers != 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_probs[1])

    init_state = cell.zero_state(tf.shape(x)[0], tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)
    rnn_output = tf.unstack(rnn_outputs, axis=1)[-1]
    output = tf.layers.Dense(dense_state_size[0], activation=tf.nn.relu)(rnn_output)
    output = tf.layers.Dense(dense_state_size[1])(output)
    prediction = output
    total_loss = tf.reduce_mean(tf.square(y - prediction))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x=x,
        y=y,
        init_state=init_state,
        final_state=final_state,
        total_loss=total_loss,
        train_step=train_step,
        prediction=prediction,
        saver=tf.train.Saver()
    )


def train_rnn(g, num_epochs, batch_size=batch_size, verbose=True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []

        for idx, epoch in enumerate(generate_epochs(data_train, num_epochs, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for x, y in epoch:
                steps += 1
                feed_dict = {g['x']: x, g['y']: y}
                training_loss_, _ = sess.run([g['total_loss'], g['train_step']], feed_dict=feed_dict)
                training_loss += training_loss_

            if verbose:
                print('epoch: {0}的平均损失值： {1}'.format(idx, training_loss/steps))
            training_losses.append(training_loss/steps)
        if isinstance(save, str):
            g['saver'].save(sess, save)
    return training_losses

def test(g, save):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, save)
        predictions = []
        ground_truth = []
        for x, y in generate_batches(data=data_test, batch_size=batch_size):
            feed_dict = {g['x']: x, g['y']: y}
            prediction_batch, loss = sess.run([g['prediction'], g['total_loss']], feed_dict=feed_dict)

            prediction_batch = np.squeeze(prediction_batch)
            predictions.append(prediction_batch)
            y_batch = np.squeeze(y)
            ground_truth.append(y_batch)
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    return predictions, ground_truth

def predict(g, save):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, save)
        predictions = []
        ground_truth = []
        first_input = True
        for x_, y in generate_batches(data=data_test, batch_size=1):
            if first_input:
                x = x_
            else:
                prediction = prediction_batch.reshape([1, 1, 1])
                x = np.concatenate((x[:, 1:, :], prediction), axis=1)
            feed_dict = {g['x']: x, g['y']: y}
            prediction_batch, loss = sess.run([g['prediction'], g['total_loss']], feed_dict=feed_dict)
            prediction_batch = np.squeeze(prediction_batch)
            predictions.append(prediction_batch)
            y_batch = np.squeeze(y)
            ground_truth.append(y_batch)
            first_input = False
    # predictions = np.concatenate(predictions, axis=0)
    # ground_truth = np.concatenate(ground_truth, axis=0)
    return predictions, ground_truth


# 训练
num_epochs = 200
num_layers = 1
keep_probs = [0.9, 0.9]


g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type='LSTM', num_layers=num_layers,
                                                   learning_rate=learning_rate)
start_time = time.time()
losses_lstm = train_rnn(g=g, num_epochs=num_epochs, save='weights/{0}_{1}layers_{2}epochs'.format('LSTM', num_layers, num_epochs))
duration_lstm = time.time()-start_time
print("LSTM训练耗时：", duration_lstm)


# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type=None, num_layers=num_layers,
#                                                    learning_rate=learning_rate)
# start_time = time.time()
# losses_basic = train_rnn(g=g, num_epochs=num_epochs, save='weights/{0}_{1}layers_{2}epochs'.format(cell_type, num_layers, num_epochs))
# duration_basic = time.time()-start_time
# print("BasicRNN训练耗时：", duration_basic)
#
# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type='LN_LSTM', num_layers=num_layers,
#                                                    learning_rate=learning_rate)
# start_time = time.time()
# losses_lnlstm = train_rnn(g=g, num_epochs=num_epochs, save='weights/{0}_{1}layers_{2}epochs'.format(LN_LSTM, num_layers, num_epochs))
# duration_lnlstm = time.time()-start_time
# print("LN_LSTM训练耗时：", duration_lnlstm)
#
# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(num_layers=num_layers,
#                                                    learning_rate=learning_rate, keep_probs=keep_probs)
# start_time = time.time()
# losses_dropoutlstm = train_rnn(g=g, num_epochs=num_epochs, save='weights/{0}_{1}layers_{2}epochs'.format('Dropout_LSTM', num_layers, num_epochs))
# duration_dropoutlstm = time.time()-start_time
# print("Dropout_LSTM训练耗时：", duration_dropoutlstm)
#
# g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type='LN_LSTM', num_layers=num_layers,
#                                                    learning_rate=learning_rate, keep_probs=keep_probs)
# start_time = time.time()
# losses_dropoutlnlstm = train_rnn(g=g, num_epochs=num_epochs, save='weights/{0}_{1}layers_{2}epochs'.format('Dropout_LN_LSTM', num_layers, num_epochs))
# duration_dropoutlnlstm = time.time()-start_time
# print("Dropout_LN_LSTM训练耗时：", duration_dropoutlnlstm)
#
#
# plt.title('loss curves of basicRNN, LSTM, LN_LSTM, Dropout_LSTM, Dropout_LN_LSTM')
# plt.plot(losses_basic, color='green', label='losses_basic({}s)'.format(duration_basic))
# plt.plot(losses_lstm, color='red', label='losses_lstm({}s)'.format(duration_lstm))
# plt.plot(losses_lnlstm, color='blue', label='losses_lnlstm({}s)'.format(duration_lnlstm))
# plt.plot(losses_dropoutlstm, color='yellow', label='losses_dropoutlstm({}s)'.format(duration_dropoutlstm))
# plt.plot(losses_dropoutlnlstm, color='cyan', label='losses_dropoutlnlstm({}s)'.format(duration_dropoutlnlstm))
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('figs/网络结构对比_{0}层{1}epochs.png'.format(num_layers, num_epochs))
# plt.show()


# 测试
cell_type = 'LSTM'
num_layers = 1
num_epochs = 200

g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type=cell_type, num_layers=num_layers, keep_probs=[1.0, 1.0])
tests, truth = test(g=g, save='weights/{0}_{1}layers_{2}epochs'.format(cell_type, num_layers, num_epochs))
g = build_multilayer_lnlstm_graph_with_dynamic_rnn(cell_type=cell_type, num_layers=num_layers, keep_probs=[1.0, 1.0])
preds, _ = predict(g=g, save='weights/{0}_{1}layers_{2}epochs'.format(cell_type, num_layers, num_epochs))
plt.title('prediction of average kills in GTD')
plt.plot(preds, color='red', label='prediction')
plt.plot(tests, color='blue', label='test')
plt.plot(truth, color='green', label='ground truth')
plt.legend()
plt.xlabel('day')
plt.ylabel('kill')
plt.savefig('figs/{4}_prediction_{2}{0}层{1}epochs_T={3}.png'.format(num_layers, num_epochs, cell_type, num_steps, data_type))
plt.show()