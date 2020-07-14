from flask import Flask
from flask import request, Response,render_template
from keras.models import load_model
import tensorflow as tf
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pandas import Series
from pandas import concat
from pandas import DataFrame
model=load_model('wind.h5')
lstm_model=model
global graph
graph = tf.get_default_graph()
app = Flask(__name__)
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=1)
    return yhat[0,0]
waste=[]
@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route('/login', methods=["POST"])
def hello_world1():
    global model
    global lstm_model
    batch_size_exp = 1
    epoch_exp = 7
    neurons_exp = 10
    predict_values_exp = 12
    lag_exp = 24
    series = pd.read_csv('T1.csv', index_col="Date/Time")
    del series['LV ActivePower (kW)']
    del series['Wind Speed (m/s)']
    del series['Wind Direction (°)']
    with graph.as_default():
        for i in range(0, 12):
            series = series[:-1]
        raw_values = series.values
        diff_values = difference(raw_values, 1)
        supervised = timeseries_to_supervised(diff_values, lag_exp)
        supervised_values = supervised.values
        train, test = supervised_values[0:-predict_values_exp], supervised_values[-predict_values_exp:]
        scaler, train_scaled, test_scaled = scale(train, test)
        predictions = list()
        expectations = list()
        test_pred = list()
        global waste
        waste=[]
        for i in range(len(test_scaled)):
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            test_pred = [yhat] + test_pred
            if i + 1 < len(test_scaled):
                test_scaled[i + 1] = np.concatenate((test_pred, test_scaled[i + 1, i + 1:]), axis=0)
            yhat = invert_scale(scaler, X, yhat)
            yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
            predictions.append(yhat)
            expected = raw_values[len(train) + i + 1]
            expectations.append(expected)
            waste.append(yhat)
            s='Hour=%d, Predicted=%f' % (i + 1, yhat)
        pyplot.plot(predictions, label="Predicted")
        pyplot.savefig('static/img/foo.png')
        return render_template("some.html")
@app.route('/something', methods=["POST"])
def TextFormat():
    superi="<head><style>#customers {font-family: \"Trebuchet MS\", Arial, Helvetica, sans-serif;border-collapse: collapse;width: 100%;}#customers td, #customers th {border: 1px solid #ddd;padding: 8px;}#customers tr:nth-child(even){background-color: #f2f2f2;}#customers tr:hover {background-color: #ddd;}#customers th {padding-top: 12px;padding-bottom: 12px;text-align: left;background-color: #4CAF50;color: white;}body{background-color: white;padding:0;margin:0;}</style></head><body><table style=\"width:100%;height=100%\" id=\"customers\" border=\"2\"><tr><th style=\"color:white\">Hour</th><th style=\"color:white\">Predicted</th></tr></body>"
    for i in range(len(waste)):
        superi+="<tr><td style=\"color:black\">"+str(i+1)+"</td><td style=\"color:black\">"+str(waste[i][0])+"</td></tr>"
    return(superi)
if __name__ == '__main__':
    app.run(debug=True)
