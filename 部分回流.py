import numpy as np
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
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
以下是计算必要的参数，其中q是计算好的，x_F、x_D、x_W、R是在实验中得到并进行换算的，所有参数都要按照实际情况进行修改。

注意：程序为部分回流时最优加料位置加料的计算，若需要实际加料位置加料的计算，请按照下面的注释修改代码。
'''

q = 1.131
x_F = 8.46
x_D = 78.06
x_W = 2.26
R =3.7

'''
##############################################################################################
'''

def f1(x,a,b,c):
    # 第一个函数的定义
    return (a * math.log(b * x + 1) + c)
 
def f2(x,a,b,c):
    # 第二个函数的定义
    return (a * math.exp(b * x) + c)

def f_q(x, q, x_F):
    q = q
    return (q/(q-1)*x - x_F/(q-1))
def f_x(x):
    return x

def f3(x,R,x_D):
    return (R/(R+1)*x + x_D/(R+1))




initial_guess = [30]  # 多起点提高成功率
def equation(x):
    return f3(x, R, x_D) - f_q(x, q, x_F)  #  
# 求解交点
solutions = []
for guess in initial_guess:
    root = fsolve(equation, guess)[0]
    if abs(equation(root)) < 1e-7:  # 验证解的精度
        solutions.append(round(root, 4))
        
# 去重并输出结果
unique_solutions = sorted(set(solutions))

x_q = float(unique_solutions[0])
y_q = float(f3(np.array(unique_solutions), R, x_D))

def f4(x):
    return ((y_q-x_W)/(x_q-x_W))*x + (1-(y_q-x_W)/(x_q-x_W))*x_W

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
x_fit1 = np.linspace(0, 40, 100)
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

def lineup(p1, type, N):
    if type == 1:
        if p1[0] > 40 and p1[1] > 60:
            p2_x = (math.log((p1[1] - c2) / a2))/b2 
        else:
            p2_x = (math.exp((p1[1]-c1)/a1) - 1 )/ b1
        xx = [p1[0], p2_x]
        yy = [p1[1], p1[1]]
        p2 = [p2_x, p1[1]]
        return xx, yy, p2
    else:
        # if N <= 13:
        #     p2_y = f3(p1[0],R,x_D)
        # else:
        #     p2_y = f4(p1[0])

        '''
        若需要实际加料位置加料的计算，把上面四行注释取消，注释下面一行。
        '''
        p2_y = min(f3(p1[0],R,x_D), f4(p1[0]))
                   

        xx = [p1[0], p1[0]]
        yy = [p1[1], p2_y]
        p2 = [p1[0], p2_y]
        return xx, yy, p2




plt.rcParams['xtick.labelsize'] = 20  # X轴刻度字体
plt.rcParams['ytick.labelsize'] = 20  # Y轴刻度字体
plt.figure(figsize=(16, 12), dpi=200)
# plt.scatter(x, y, color='red', label='原始数据', zorder=5, s=80)
plt.plot(x_fit1, y_fit1-0.12, 'k-',  linewidth=3)
plt.plot(x_fit2, y_fit2, 'k-',  linewidth=3)
plt.plot([0, 89.43], [0, 89.43], 'k-', linewidth=2, alpha=0.7)
plt.plot([x_D, x_q], [x_D, y_q], 'k-', linewidth=1,  alpha=0.7)
plt.plot([x_W, x_q], [x_W, y_q], 'k-', linewidth=1,  alpha=0.7)
plt.plot([x_F/(q-1)/(q/(q-1)-1), 30], [x_F/(q-1)/(q/(q-1)-1), f_q(30,q,x_F)], 'k-', linewidth=1)
# plt.plot([25.009, 30], [25.009, f_q(30,q,x_F)], 'k-', linewidth=1)
plt.xlim((0, 100))
plt.ylim((0, 100))
'''
若需要实际加料位置加料的计算，把下面一行注释取消。
'''
# plt.plot([0, x_q], [f3(0, R, x_D), y_q], 'k--', linewidth=1)

plt.title("部分回流最优加料位置加料", fontsize=30) # 记得修改标题
plt.xlabel("x", fontsize=30)
plt.ylabel("y", fontsize=30)
plt.legend(fontsize = "20")
p0 = [x_D, x_D]
N = 0
while(True):
    xx, yy, p0 = lineup(p0, 1, N)
    plt.plot(xx, yy, 'k-',  alpha=0.7)
    N += 1
    if not (p0[0] > x_W and p0[1] > x_W):
        break
    xx, yy, p0 = lineup(p0, 2, N)
    plt.plot(xx, yy, 'k-',  alpha=0.7)
    if not (p0[0] > x_W and p0[1] > x_W):
        break
print('N = ', N)
plt.grid(True)
plt.show()