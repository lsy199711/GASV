from matplotlib import pyplot as plt
import lasio
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from model.model import one_cnn_model, three_cnn_model, bp_model
import openpyxl
from sklearn.svm import SVR
from dataset.DataSet import dataset_lab
from dataset.Normalization import Normalization, Anti_Normalization, Normalization_lab, Anti_Normalization_lab
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import tree
from sklearn import model_selection
#35作训练，23作预测


if __name__ == '__main__':

    # 读取初始化相关数据
    fileName_23 = '../井/WY23.las'
    fileName_35 = "../井/WY35.las"
    las_23 = lasio.read(fileName_23)
    b = 6
    Para_VVR, df_Para, Aim, de_Aim, data, train_data_length = dataset_lab(b)
    NL_input_data, NL_output_data = Normalization_lab(data, train_data_length)
    yhat_array = np.empty((Aim.shape[0], 1))
    loo = LeaveOneOut()
    loo.get_n_splits(NL_input_data)

    for
        yhat, model, history = three_cnn_model(NL_train_X_data, NL_train_Y_data, NL_test_X_data, 50)

        NL_train_X_data, NL_train_Y_data, NL_test_X_data = NL_train_X_data.reshape(-1, data.shape[1] - 1), NL_train_Y_data.reshape(-1, 1), NL_test_X_data.reshape(-1, data.shape[1] - 1)
        # dtr = tree.DecisionTreeRegressor(random_state=32)
        # yhat = dtr.fit(NL_train_X_data, NL_train_Y_data).predict(NL_test_X_data).reshape(-1, 1)
        #
        # rfr = RandomForestRegressor(random_state=80)
        # yhat = rfr.fit(NL_train_X_data, NL_train_Y_data).predict(NL_test_X_data).reshape(-1, 1)
        # svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.01)
        # yhat = svr_rbf.fit(NL_train_X_data, NL_train_Y_data).predict(NL_test_X_data).reshape(-1, 1)
        # print("系数： ", ev)
        # print('yhat: ', yhat)
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs = range(1, len(loss) + 1)
        #
        # # plt.title('loss')
        # plt.figure()
        # plt.plot(epochs, loss, 'blue', label=las_23.keys()[b] + ' loss')
        # plt.legend(loc="best")
        # plt.plot(epochs, val_loss, 'red', label=las_23.keys()[b] + ' val_loss')
        NL_test_X_data = NL_test_X_data.reshape(-1, Num_Var)
        yhat = Anti_Normalization_lab(data, NL_test_X_data, yhat)
        # print("yhat: ", yhat)
        yhat_array[i][0] = yhat
    print("yhat_array: ", yhat_array)
    print("Aim: ", Aim)
    print('mse: ', mean_squared_error(Aim, yhat_array))
    print('平均相对误差: ', np.average(np.abs(Aim - yhat_array) / yhat_array, axis=0))
    print('R2: ', r2_score(Aim, yhat_array))
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)

    # plt.title('loss')
    # plt.figure()
    # plt.plot(epochs, loss, 'blue', label=las_23.keys()[b] + ' loss')
    # plt.legend(loc="best")
    # plt.plot(epochs, val_loss, 'red', label=las_23.keys()[b] + ' val_loss')
    # plt.close()
    # plt.scatter(Aim, yhat_array, alpha=0.6)
    # plt.plot((0, 5.4), (0, 5.4), ls='--', c='k', label="1:1 line")
    # plt.xlabel("true/(m^3)")
    # plt.ylabel("predict/(us/m)")
    # plt.show()
    # depth = [item.value for item in list(worksheet.columns)[0]]
    fig_2 = plt.figure(figsize=(20, 9))  # 作图画布大小(每循环一次，就会建立新的画布）
    # for f in range(3):
    #     ax = fig_2.add_subplot(1, 4, f + 1)  # 图布一行len(a)列，画在第f+1块
    #     ax.plot(Para_VVR[train_data_length:, f], depth)
    #     ax.set_xlabel(df_Para.keys()[f])
    #     ax.xaxis.tick_top()
    #     ax.invert_yaxis()
    #     plt.tick_params(labelsize=6)
    ax = fig_2.add_subplot(1, 1, 1)
    ax.plot(Aim, 'o-', linewidth=2, label = 'true_value')
    ax.plot(yhat_array, 'o-', color="red", linewidth=2, label = 'predict_value')
    # plt.xticks(np.arange(1, 38 , 1))
    ax.set_xlabel(las_23.keys()[b])
    plt.tick_params(labelsize=6)
#     # plt.savefig(r"D:\software\pycharm\PyCharm 2019.3.3\projects\predict_GASV\结果\onewey\图片\两口井3to5\onecon/" + name + ".png")
#     plt.savefig("lab_bp_loocv_carve_predict")
plt.show()
# with open(r'D:\software\pycharm\PyCharm 2019.3.3\projects\predict_GASV\结果\onewey\class\Aim_predict_nonsort.txt', 'w') as f:
#     np.savetxt(f, Aim_data, fmt="%.6f")
# resullt: mse:  0.19096958874525233
# R2:  0.9763681438005949