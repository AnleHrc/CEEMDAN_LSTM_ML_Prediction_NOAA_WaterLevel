import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import importlib
import sys
importlib.reload(sys)
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
plt.rcParams['font.sans-serif'] = 'SimHei'  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['figure.dpi'] = 200  # 图像分辨率
plt.rcParams['text.color'] = 'black'  # 文字颜色
plt.style.use('ggplot')
print(plt.style.available)  # 可选的plt绘图风格
'''
['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']
'''
# 忽略警告
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


def xgboost_all_file(file_name,month):
    # 通过第一个_和第二个_截取站点名词
    first_underscore_index = file_name.find("_")
    second_underscore_index = file_name.find("_", first_underscore_index + 1)
    site_name = file_name[first_underscore_index + 1:second_underscore_index]
    print('For '+site_name)


    # %%
    df = pd.read_csv('../Data/NOAA/All Station/{}'.format(file_name))

    # 将日期和时间合并为一个列，并将其设置为DataFrame的索引
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time (GMT)'])
    df.set_index('DateTime', inplace=True)
    df = df[['Wind Speed (m/s)', 'Wind Dir (deg)', 'Wind Gust (m/s)', 'Air Temp (°C)', 'Baro (mb)', 'Water Level(m)']]
    # 将数据分割为训练集和测试集
    train_months = [1, 2, 4, 5, 7, 8, 10, 11]
    test_months = [month]
    # dataset.drop('TIMESTAMP_START',axis=1,inplace=True) #axis=1 表示按列删除，inplace=True 表示在原 DataFrame 上进行修改
    # # %%
    # '''
    # ['Date', 'Time (GMT)', 'Wind Speed (m/s)', 'Wind Dir (deg)',
    #    'Wind Gust (m/s)', 'Air Temp (°C)', 'Baro (mb)', 'Water Level(m)']
    # '''
    # dataset = dataset[['Wind Speed (m/s)', 'Air Temp (°C)', 'Water Level(m)']]

    # dataset = dataset[
    #     ['SW_IN_F_F', 'SW_IN_F', 'TA_F_F', 'TA_F']]
    # %%
    # dataset.head()
    # %% md
    # 数据归一化
    # %%

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df)
    # %%
    # dataset[0:20]

    # %% md
    # 转换数据格式
    # %%
    # def splitData(var, per_test):
    #     num_test = int(len(var) * per_test)
    #     train_size = int(len(var) - num_test)
    #     train_data = var[0:train_size]
    #     test_data = var[train_size:train_size + num_test]
    #     return train_data, test_data
    #
    # # %% md
    # ## 数据集划分
    # # %%
    # training, testing = splitData(dataset, 0.2)
    training = dataset[df.index.month.isin(train_months)]
    testing = dataset[df.index.month.isin(test_months)]
    # %%
    training
    # %%
    testing
    # %%
    '''
    dataset: 数据集
    n_past: 所使用过去的样本数量（使用过去的多少个样本来推算下一天）
    '''

    def createXY(dataset, n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i, 5])
        return np.array(dataX), np.array(dataY)

    # %%
    ### 创建用于训练的时间序列数据
    # %%
    n_past = 1
    trainX, trainY = createXY(training, n_past)
    testX, testY = createXY(testing, n_past)
    # %% md
    ### 打印训练集和测试集的形状
    # %%
    print('trainX Shape---', trainX.shape)
    print('trainY Shape---', trainY.shape)
    # %%
    print('testX Shape---', testX.shape)
    print('testY Shape---', testY.shape)
    # %%
    trainX.shape[1]
    # %%
    # trainX = trainX.reshape(21007, 30)#
    # %% md
    #### 重塑为RF可以识别的形状
    # %%
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1] * trainX.shape[2]))

    trainX.shape
    # %%
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))

    testX.shape
    # %% md
    ## 随机森拉
    # %%
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=-1, random_state=50, max_features=1.0,
                                  min_samples_leaf=10)
    # %%
    model.fit(trainX, trainY)
    # %%
    pre = model.predict(testX)
    pre = np.array(pre).reshape(-1, 1)
    prediction_copies_array = np.repeat(pre, trainX.shape[1], axis=-1)  # 将一个数组prediction在最后一个轴上（即axis=-1）进行复制，重复8次，并将

    pre = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(pre), trainX.shape[1])))[:,5]  # 进行逆变换但是，只需要最后一列

    test_data = np.array(testY).reshape(-1, 1)

    prediction_copies_test_data = np.repeat(test_data, trainX.shape[1],axis=-1)  # 将一个数组prediction在最后一个轴上（即axis=-1）进行复制，重复8次，并将

    # %%
    test_data = scaler.inverse_transform(np.reshape(prediction_copies_test_data, (len(test_data), trainX.shape[1])))[:,5]  # 进行逆变换但是，只需要最后一列

    # %%
    from sklearn.metrics import r2_score as r2
    y = np.array(test_data)
    y_pred = pre
    MSE = mean_squared_error(y, y_pred)
    RMSE = math.sqrt(MSE)
    # MAE=mean_absolute_error(y, pre)
    # MAPE = metrics.mean_absolute_percentage_error(y, pre)
    MAE = np.mean(np.abs(y - y_pred))
    MAPE = np.mean(np.abs((y - y_pred) / y))
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
        'Value': [r2(y, y_pred), RMSE, MAE, ]
    }

    results_seaon = pd.DataFrame(data)
    # 保存DataFrame为CSV文件
    results_seaon.to_csv(site_name + '_metrics{}'.format(month) + '.csv', index=False)
    # y = pd.DataFrame(y)
    # y.to_csv('../ResultAnaAndProcess/NOAA_WL/WL_WS_AT/RF/' + 'RF_{}'.format(site_name) + '_True' + '.csv',index=False)
    # y_pred = pd.DataFrame(y_pred)
    # y_pred.to_csv('../ResultAnaAndProcess/NOAA_WL/WL_WS_AT/RF/' + 'RF_{}'.format(site_name) + '_Pre' + '.csv',index=False)

csv_path = '../Data/NOAA/All Station'
csv_files = glob.glob(os.path.join(csv_path,'*.csv'))
moth = [3, 6, 9, 12]
for file in csv_files:
    for moths in moth:
        xgboost_all_file(os.path.basename(file), moths)
    print("-------------------------")