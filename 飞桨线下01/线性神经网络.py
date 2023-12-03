import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def first_order_linear_func(x: np.ndarray):
    return 2.7862 * x + 3.1415


def first_order_linear_func_with_noise(x: np.ndarray):
    return 2.7862 * x + 3.1415 + np.random.normal(0, 0.1, x.shape)


if __name__ == '__main__':
    # 1. 生成数据
    x = np.linspace(0, 10, 100)
    y = first_order_linear_func_with_noise(x)
    # 2. 绘制数据
    plt.scatter(x, y)
    plt.show()
    # 3. 拟合数据
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    print(model.coef_, model.intercept_)
    # 4. 绘制拟合结果
    plt.scatter(x, y)
    plt.plot(x, model.predict(x), color='r')
    plt.show()

    # # 5. 使用Pytorch实现线性回归
    # class LinearRegressionModel(nn.Module):
    #     def __init__(self):
    #         super(LinearRegressionModel, self).__init__()
    #         self.linear = nn.Linear(1, 1)  # 输入和输出的维度都是1
    #
    #     def forward(self, x):
    #         out = self.linear(x)
    #         return out
    #
    # model = LinearRegressionModel()
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    #
    # num_epochs = 1000
    # for epoch in range(num_epochs):
    #     inputs = torch.from_numpy(x.astype(np.float32))
    #     labels = torch.from_numpy(y.astype(np.float32))
    #
    #     outputs = model(inputs)
    #     loss = criterion(outputs, labels)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     if (epoch + 1) % 20 == 0:
    #         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # # 6. 绘制拟合结果
    # plt.scatter(x, y)
    # plt.plot(x, model(torch.from_numpy(x.astype(np.float32))).data.numpy(), color='r')
    # plt.show()
    #打印参数
    # print(model.state_dict())
    #打印模型结构
    # print(model)
    #保存模型