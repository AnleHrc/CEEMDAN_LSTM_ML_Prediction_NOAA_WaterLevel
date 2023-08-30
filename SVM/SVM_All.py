import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import math
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score as r2, mean_squared_error
# 忽略警告
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


def xgboost_all_file(file_name,month):
    # 通过第一个_和第二个_截取站点名词
    first_underscore_index = file_name.find("_")
    second_underscore_index = file_name.find("_", first_underscore_index + 1)
    site_name = file_name[first_underscore_index + 1:second_underscore_index]
    print('For '+site_name)

    df = pd.read_csv('../Data/NOAA/All Station/{}'.format(file_name))

    # 将日期和时间合并为一个列，并将其设置为DataFrame的索引
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time (GMT)'])
    df.set_index('DateTime', inplace=True)
    df = df[['Wind Speed (m/s)', 'Wind Dir (deg)', 'Wind Gust (m/s)', 'Air Temp (°C)', 'Baro (mb)', 'Water Level(m)']]
    # 将数据分割为训练集和测试集
    train_months = [1, 2, 4, 5, 7, 8, 10, 11]
    test_months = [month]



    ### 数据归一化
    # %%
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df)
    # %%
    dataset

    train = dataset[df.index.month.isin(train_months)]
    test = dataset[df.index.month.isin(test_months)]
    # # %%
    # def splitData(var, per_test):
    #     num_test = int(len(var) * per_test)
    #     train_size = int(len(var) - num_test)
    #     train_data = var[0:train_size]
    #     test_data = var[train_size:train_size + num_test]
    #     return train_data, test_data
    #
    # # %%
    # train, test = splitData(dataset, 0.2)
    # %%
    print('train_len:', len(train), 'test_len:', len(test))
    # %%
    train.shape[1]

    # %%
    def createXY(data, n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(data)):
            dataX.append(data[i - n_past:i, 0:data.shape[1]])
            dataY.append(data[i, 5])
        return np.array(dataX), np.array(dataY)

    # %%
    # dataX = [[1,2],
    #          [2,3],
    #          [2,4]]
    # dataX = np.array(dataX)
    # dataX

    # %%
    n_past = 1
    trainX, trainY = createXY(train, n_past)
    testX, testY = createXY(test, n_past)
    # %%
    print('train Shape---', trainX.shape)
    print('trainY Shape---', trainY.shape)
    # %%
    print('testX Shape---', testX.shape)
    print('testY Shape---', testY.shape)


    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1] * trainX.shape[2]))


    testX = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))


    ### SVM

    from sklearn.svm import SVR
    svr = SVR(kernel='rbf')
    model = svr.fit(trainX, trainY)
    # %%
    pre = model.predict(testX)
    # %%
    pre
    # %%
    pre = np.array(pre).reshape(-1, 1)

    prediction_copies_array = np.repeat(pre, trainX.shape[1], axis=-1)  # 将一个数组prediction在最后一个轴上（即axis=-1）进行复制，重复8次，并将

    pre = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(pre), trainX.shape[1])))[:,
          5]  # 进行逆变换但是，只需要最后一列

    test_data = np.array(testY).reshape(-1, 1)

    prediction_copies_test_data = np.repeat(test_data, trainX.shape[1],
                                            axis=-1)  # 将一个数组prediction在最后一个轴上（即axis=-1）进行复制，重复8次，并将
    prediction_copies_test_data
    test_data = scaler.inverse_transform(np.reshape(prediction_copies_test_data, (len(test_data), trainX.shape[1])))[:,
                5]  # 进行逆变换但是，只需要最后一列

    from sklearn.metrics import r2_score as r2
    from matplotlib import pyplot as plt
    y = np.array(test_data)
    y_pred = pre
    MSE = mean_squared_error(y, y_pred)
    RMSE = math.sqrt(MSE)
    # MAE=mean_absolute_error(y, pre)
    # MAPE = metrics.mean_absolute_percentage_error(y, pre)
    MAE = np.mean(np.abs(y - y_pred))
    # MAPE = np.mean(np.abs((y - y_pred) / y))
    print("rmse :", RMSE)
    print("mae :", MAE)
    # print("mape :", MAPE)
    print('R² :', r2(y, y_pred))
    plt.plot(y, color='red', label='Real Value')
    plt.plot(y_pred, color='blue', label='Pred Value')
    plt.title('Prediction Water Level')
    plt.xlabel('Time')
    plt.ylabel('Detail Value')
    plt.legend()
    plt.show()

    data = {
        'Metric': ['R2', 'RMSE', 'MAE'],
        'Value': [r2(y, y_pred),RMSE,MAE, ]
    }

    results_seaon = pd.DataFrame(data)
    # 保存DataFrame为CSV文件
    results_seaon.to_csv(site_name+'_metrics{}'.format(month)+'.csv', index=False)
    # y = pd.DataFrame(y)
    # y.to_csv('../ResultAnaAndProcess/NOAA_WL/WS_WD_WG_WL/SVM/' + 'SVM_{}'.format(site_name) + '_True' + '.csv',
    #               index=False)
    # y_pred = pd.DataFrame(y_pred)
    # y_pred.to_csv('../ResultAnaAndProcess/NOAA_WL/WS_WD_WG_WL/SVM/' + 'SVM_{}'.format(site_name) + '_Pre' + '.csv',
    #              index=False)
    # %%
    # y_true = pd.DataFrame(y)
    # y_true.to_csv('../LSTM/LSTM_Result_Collection_SW_TA_LW_VPD_PA_WS_RH/PredictionResult/SW_TA_LW_VPD_PA_WS_RH/'+'SW_SVMYTrue'+'.csv',index=False)
    # y_pre = pd.DataFrame(y_pred)
    # y_pre.to_csv('../LSTM/LSTM_Result_Collection_SW_TA_LW_VPD_PA_WS_RH/PredictionResult/SW_TA_LW_VPD_PA_WS_RH/'+'SW_SVMYTPre'+'.csv',index=False)
    # %%
    from matplotlib import pyplot as plt

    # plt.plot(y, color='red', label='Real Value')
    # plt.plot(y_pre, color='blue', label='Pred Value')
    # plt.title('Prediction SW_IN_F_{}'.format(site_name))
    # plt.xlabel('Time')
    # plt.ylabel('Detail Value')
    # plt.legend()
    # plt.show()

csv_path = '../Data/NOAA/All Station'
csv_files = glob.glob(os.path.join(csv_path,'*.csv'))
moth = [3,6,9,12]
for file in csv_files:
    for moths in moth:
        xgboost_all_file(os.path.basename(file),moths)
    print("-------------------------")