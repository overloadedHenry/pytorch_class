import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def polynomial(x:np.ndarray, w:np.ndarray, b:np.ndarray):
    return x @ w.T + b.T

if __name__ == '__main__':

    #生成1000个五维向量,数值在0-100之间,浮点数
    x = np.random.rand(1000, 5) * 100
    print(x.shape)
    print(x)
    w = np.array([[1.035, 2.009, 3.981, 2.776, 5.821]])
    b = np.array([[2.1]])
    print(w.T.shape, b.T.shape)
    y = polynomial(x, w, b)
    print(y.shape)
    # 生成dataframe
    df = pd.DataFrame(np.concatenate([x, y], axis=1), columns=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
    print(df.head())
    # 保存为csv文件
    df.to_csv('data.csv', index=True)

    # 生成测试集
    x_test = np.random.rand(100, 5) * 100
    y_test = polynomial(x_test, w, b)
    df_test = pd.DataFrame(np.concatenate([x_test, y_test], axis=1), columns=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
    df_test.to_csv('test.csv', index=True)

