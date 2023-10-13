import numpy as np
from scipy import integrate
import time

def CrossCorr_1(func1, func2, start, end, step, x):
    """
    :param func1: Function
    :param func2: Function
    :param start: Integral start
    :param end: Integral end
    :param step: step:control the accuracy of the integral
    :return: corr: cross correlation value
    """

    res = []
    for tau in np.arange(start, end, step): # 跑循环进行积分
        res.append(func1(tau) * func2(x + tau) * step)

    corr = np.sum(np.array(res))

    return corr


def CrossCorr_2(func1, func2, x, start, end):
    """
    :param func1:
    :param func2:
    :param x:
    :param start:
    :param end:
    :return:
    """
    corr = integrate.quad(lambda tau: func1(tau) * func2(tau + x), start, end)
    return corr


def CrossCorr_3(func1, func2, start, end, step, x):
    tau_values = np.arange(start, end, step) # ndarray[]
    f_values = func1(tau_values)
    g_values = func2(x + tau_values)

    corr = np.sum(f_values * g_values) * step

    return corr


def func1(x):
    return x


def func2(x):
    return x ** 2

# 定义卷积运算函数
def CrossCorr_4(func1, func2, start, end, step, x):
    """
    :param func1:
    :param func2:
    :param start:
    :param end:
    :param step:
    :return:
    """
    res = []
    for tau in np.arange(start, end, step):
        res.append(func1(tau) * func2(x - tau) * step)

    corr = np.sum(np.array(res))

    return corr

if __name__ == '__main__':
    # 记录代码执行时间的间隔
    start = time.time()
    print(CrossCorr_3(func1, func2, -10, 10, 1e-5, 2))
    end = time.time()
    print(end - start)


    # print(CrossCorr_3(func1, func2, -10, 10, 1e-6, 2))
    # print(CrossCorr_2(func1, func2, x=3, start=-10, end=10))
    #1.4160540103912354
    #0.0009920597076416016
    #0.02169179916381836