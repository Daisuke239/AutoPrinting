import numpy as np
import matplotlib.pyplot as plt
import math
'''
本程序默认使用者会使用python，如果从来没用过python请直接关闭本程序并将本程序移动至回收站中。
请确保以上库都安装好了，若没有安装请在命令行使用以下命令安装：
pip install numpy matplotlib scipy sklearn
'''
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
# 原始数据 出自实验书第81页。
data = np.array([
    [0, 0],
    [1.9, 17],
    [7.21, 38.91],
    [9.66, 43.75],
    [12.38, 47.04],
    [16.61, 50.89],
    [23.37, 54.45],
    [26.08, 55.8],
    [32.73, 58.36],
    [39.65, 61.22],
    [50.79, 65.64],
    [51.98, 65.99],
    [57.32, 68.41],
    [67.63, 73.85],
    [74.72, 78.15],
    [89.43, 89.43]
])

x = data[:, 0]
y = data[:, 1]
'''
以下是计算必要的参数，其中q是计算好的，x_D、x_W分别对应塔顶、塔釜采出液轻组分摩尔分数，是在实验中得到并进行换算，所有参数都要按照实际情况进行修改。
'''
x_D = 81
x_W = 2
'''
##############################################################################################
'''
# 按x=40分段
mask1 = x <= 40
mask2 = x > 40
x1, y1 = x[mask1], y[mask1]  # 前半段 (x≤40)
x2, y2 = x[mask2], y[mask2]  # 后半段 (x>40)

# 定义拟合函数
def log_func(x, a, b, c):
    """对数函数：y = a * ln(b*x + 1) + c"""
    #return a /(1 + np.exp(- b * (x-40))) + c
    return a * np.log(b * x + 1) + c

def exp_func(x, a, b, c):
    """指数函数：y = a * exp(b*x) + c"""
    return a * np.exp(b * x) + c

# 前半段对数拟合
popt1, pcov1 = curve_fit(log_func, x1, y1, p0=(30, 0.1, 0))
a1, b1, c1 = popt1
print(f"前半段对数拟合参数: a={a1:.4f}, b={b1:.4f}, c={c1:.4f}")
a1=14
b1=1.75
c1=2
# 后半段指数拟合
popt2, pcov2 = curve_fit(exp_func, x2, y2, p0=(60, 0.01, 0))
a2, b2, c2 = popt2
a2=7.7855
b2=0.019601
c2=44.4944
print(f"后半段指数拟合参数: a={a2:.4f}, b={b2:.6f}, c={c2:.4f}")

# 生成拟合曲线点
x_fit1 = np.linspace(0, 40, 1000)
y_fit1 = log_func(x_fit1, a1, b1, c1)
x_fit2 = np.linspace(40, 90, 100)
y_fit2 = exp_func(x_fit2, a2, b2, c2)

def piecewise_func(x, a, b, c, d):
    """Piecewise function with continuity at x=40"""
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= 40:
            y[i] = log_func(xi, a, b, c)
        else:
            # Ensure continuity: f_exp(40) = f_log(40)
            exp_val = exp_func(xi, a, b, d)
            y[i] = exp_val
    return y

# 绘图

def lineup(p1, type):
    if type == 1:
        if p1[0] > 40.0 and p1[1] > 62.66:
            p2_x = (math.log((p1[1] - c2) / a2))/b2 
        else:
            p2_x = (math.exp((p1[1]-c1)/a1) - 1 )/ b1
        xx = [p1[0], p2_x]
        yy = [p1[1], p1[1]]
        p2 = [p2_x, p1[1]]
        return xx, yy, p2
    else:
        p2_y = p1[0]
        xx = [p1[0], p1[0]]
        yy = [p1[1], p2_y]
        p2 = [p1[0], p2_y]
        return xx, yy, p2





plt.rcParams['xtick.labelsize'] = 20  # X轴刻度字体
plt.rcParams['ytick.labelsize'] = 20  # Y轴刻度字体
plt.figure(figsize=(16, 12), dpi=200)
# plt.scatter(x, y, color='red', label='原始数据', zorder=5, s=80)
plt.plot(x_fit1, y_fit1-0.12, 'b-', label='对数拟合 (x≤40)', linewidth=3)
plt.plot(x_fit2, y_fit2, 'b-', label='指数拟合 (x>40)', linewidth=3)
plt.axvline(x=40, color='gray', linestyle='--', linewidth=2, label='分界线 x=40', alpha=0.7)
plt.plot([0, 89.43], [0, 89.43], 'g--', linewidth=2, label='y = x', alpha=0.7)
plt.title("全回流逐板计算图", fontsize=30)
plt.xlabel("x", fontsize=30)
plt.ylabel("y", fontsize=30)
plt.legend(fontsize = "20")
p0 = [x_D, x_D]
N = 0
while(True):
    xx, yy, p0 = lineup(p0, 1)
    plt.plot(xx, yy, 'k-',  alpha=0.7)
    N += 1
    if not (p0[0] > x_W and p0[1] > x_W):
        break
    xx, yy, p0 = lineup(p0, 2)
    plt.plot(xx, yy, 'k-',  alpha=0.7)
    if not (p0[0] > x_W and p0[1] > x_W):
        break
print('N = ', N)

plt.grid(True)
plt.show()