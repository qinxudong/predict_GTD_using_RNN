# predict_GTD_using_RNN
Time series prediction based on Global Terrorism Database using RNN.


1. Testing the prediction effectiveness using a sine curve data.

num_steps = 50
![img](https://github.com/qinxudong/predict_GTD_using_RNN/blob/master/figs/prediction_LSTM1%E5%B1%8250epochs_T%3D50.png)

num_steps = 200
![img](https://github.com/qinxudong/predict_GTD_using_RNN/blob/master/figs/prediction_LSTM1%E5%B1%8250epochs_T%3D200.png)

'test' means using data of num_steps to predict data of one step, 'prediction' means using data of num_steps as starting data to predict the long rest data.
As you can see, the algorithm can converge easily, but not able to predict long-time data. It appear to be the same when using real data extracted from GTD.


2. Prediction effectiveness using GTD data.

num_steps = 30, epochs=50
![img](https://github.com/qinxudong/predict_GTD_using_RNN/blob/master/figs/GTD_prediction_LSTM1%E5%B1%8250epochs_T%3D30.png)

num_steps = 30, epochs=200
![img](https://github.com/qinxudong/predict_GTD_using_RNN/blob/master/figs/GTD_prediction_LSTM1%E5%B1%82200epochs_T%3D30.png)

num_steps = 370, epochs=50
![img](https://github.com/qinxudong/predict_GTD_using_RNN/blob/master/figs/GTD_prediction_LSTM1%E5%B1%8250epochs_T%3D370.png)

I only trained the model 200 epochs at most, promisingly, it will get better when the number of epochs grows, especially for test results.
The results of prediction is even poorer than before, which is reasonable because the data is far more complicated.
According the results of testing and prediction, this model can be used to predict one day data by inputting several days or even one year data.
For the issues requiring long-time prediction, sequence2sequence models may be worth to try.
