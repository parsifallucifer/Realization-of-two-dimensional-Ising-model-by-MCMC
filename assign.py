import matplotlib.pyplot as plt
import numpy as np
import copy

x1 = []
x2 = []
y1 = []
y2 = []
dE = [[]] * 3
dM = [[]] * 3
A = 2

def Spin(S, T, Matrix_size):
    K = 1
    for i in range(Matrix_size):
        for j in range(Matrix_size):
            E = getEnergy(i, j, S, Matrix_size)
            delta_E = -E*2
            if delta_E <= 0:
                S[i,j] = - S[i,j]
            # 几率翻转
            elif np.exp((- delta_E)/(K * T)) > np.random.random():
                S[i,j] = - S[i,j]
    return S

def getEnergy(i, j, S, Size):
    J = 0.5
    H = 0
    top = i-1 if i>0 else Size-1
    bottom = i+1 if i<(Size-1) else 0
    left = j-1 if j>0 else Size-1
    right = j+1 if j<(Size-1) else 0
    E_loc_former = -1.0 * S[i,j] * (J * (S[top,j] + S[bottom,j] + S[i,left] + S[i,right]) +  H)
    return E_loc_former


def getAllEnergy(S, Size):
    totalE = 0
    for i in range(Size):
        for j in range(Size):
            totalE += getEnergy(i, j, S, Size)
    totalE = totalE / 100
    return totalE

def BuiltMatrix(Matrix_size, Temperature):
    M = []
    E = []
    C = [-1, 1]
    S = np.random.choice(C, (Matrix_size, Matrix_size))
    for numi in range(1050):
        S = Spin(S, Temperature,Matrix_size)
        m = abs(sum(sum(S))) / (Matrix_size ** 2)
        M.append(m)
        e = getAllEnergy(S, Matrix_size)
        E.append(e)
    newM = sum(M[50:]) / 1000
    newE = sum(E[50:]) / 1000
    if Temperature < 0.77 and Temperature > 0.73:
        dE[0] = copy.deepcopy(E[50:])
        dM[0] = copy.deepcopy(M[50:])
    elif Temperature < 1.25 and Temperature > 1.15:
        dE[1] = copy.deepcopy(E[50:])
        dM[1] = copy.deepcopy(M[50:])
    elif Temperature < 1.77 and Temperature > 1.73:
        dE[2] = copy.deepcopy(E[50:])
        dM[2] = copy.deepcopy(M[50:])

    y1.append(newE)
    y2.append(newM)
    return S

for t in np.arange(0.1, 6.0, 0.05):
    newS = BuiltMatrix(10, t)
    x1.append(t)
    x2.append(t)
plt.figure(num = 1)
plt.title(u'E & T')
plt.xlabel('T')
plt.ylabel('E')
plt.scatter(x1, y1, s=10, c="#ff1212", marker='o')
plt.figure(num = 2)
plt.title(u'M & T')
plt.xlabel('T')
plt.ylabel('M')
plt.scatter(x2, y2, s=10, c="#ff1212", marker='o')

def Mix(E_origin):
    if len(E_origin) % 2 == 1:
        E_origin.append(E_origin[-1])
        print("True")
    for t in np.arange(0, len(E_origin), 2):
        E_origin[int(t / 2)] = (E_origin[t] + E_origin[t + 1]) / 2
    E_origin = copy.deepcopy(E_origin[:round(len(E_origin)/2)])
    return E_origin

for w in range(0, 3):
    x3 = []
    x4 = []
    y3 = []
    y4 = []
    newdE = copy.deepcopy(dE[w])
    newdM = copy.deepcopy(dM[w])
    for i in range(1, 21):
        sum1 = 0
        sum2 = 0
        if len(newdE) == 1 and len(newdM) == 1:
            break;
        if len(newdE) > 1:
            print(newdE)
            newdE = Mix(newdE)
            print(len(newdE))
            print(newdE)
            for j in range(len(newdE)):
                sum1 += (newdE[j] - np.mean(newdE)) ** 2
            delta1 = (sum1 / ((len(newdE) - 1) * len(newdE))) ** 0.5
            x3.append(i)
            y3.append(delta1)
        if len(newdM) > 1:
            newdM = Mix(newdM)
            for j in range(len(newdM)):
                sum2 += (newdM[j] - np.mean(newdM)) ** 2
            delta2 = (sum2 / ((len(newdM) - 1) * len(newdM))) ** 0.5
            x4.append(i)
            y4.append(delta2)

    plt.figure(num = w*2 + 3)
    plt.title(u'deltaE & t')
    plt.xlabel('t')
    plt.ylabel('delta')
    plt.scatter(x3, y3, s=10, c="#ff1212", marker='o')

    plt.figure(num = w*2 + 4)
    plt.title(u'deltaM & t')
    plt.xlabel('t')
    plt.ylabel('delta_M')
    plt.scatter(x4, y4, s=10, c="#ff1212", marker='o')


"""
for i in range(1, 11):
    if len(M_origin) == 1:
        break;
    sum = 0
    for t in range(0, len(M_origin), 2):
        if t == len(M_origin) - 1:
            M_origin[int(len(M_origin) / 2)] = E[t]
        else:
            M_origin[int(t / 2)] = (M_origin[t] + M_origin[t + 1]) / 2
    M_origin = copy.deepcopy(M_origin[:round(len(M_origin))])
    for j in range(len(M_origin)):
        sum += (M_origin[j] - np.mean(M_origin)) ** 2
    delta2 = (sum / ((len(M_origin) - 1) * len(M_origin))) ** 0.5
    x4.append(i)
    y4.append(delta2)
"""
"""ss
plt.figure(num = 3)
plt.title(u'deltaE & t')
plt.xlabel('t')
plt.ylabel('delta')
plt.scatter(x3, y3, s=10, c="#ff1212", marker='o')

plt.figure(num = 4)
plt.title(u'deltaM & t')
plt.xlabel('t')
plt.ylabel('delta_M')
plt.scatter(x4, y4, s=10, c="#ff1212", marker='o')
"""
plt.show()
