# -*- coding: utf-8 -*-
"""
The Singular Binomial Point Method
"""
import numpy as np
import datetime as dt
import decimal as dc
print(dc.getcontext())
#dc.getcontext().prec=60 #将dc.Decimal的小数点位数设成50

class SBPModel:
    def __init__(self, h, S0, K, n, T, sigma, r):  # h是HW模型的参数，K strike price，n步数，T存续期
        '''
        self.h = dc.Decimal(h)
        self.S0 = dc.Decimal(S0)
        self.K = dc.Decimal(K)
        self.n = int(n)
        self.T = dc.Decimal(T)
        self.sigma = dc.Decimal(sigma)
        self.r = dc.Decimal(r)
        self.Upper=Upper
        '''
        self.h = np.float64(h)
        self.S0 = np.float64(S0)
        self.K = np.float64(K)
        self.n = int(n)
        self.T = np.float64(T)
        self.sigma = np.float64(sigma)
        self.r = np.float64(r)
        

    def build(self, AM=True, Upper=True):
        if AM == True:
            self.type = 'American Asian Option'
        if AM == False:
            self.type = 'European Asian Option'
        starttime = dt.datetime.now()
        ##########################################################################################
        # 对原始数据做一定的处理
        self.Upper = Upper
        h = self.h
        S0 = self.S0
        K = self.K
        n = self.n
        T = self.T
        sigma = self.sigma
        r = self.r
        deltat = T / n  # 每两个step之间的时间间隔
        u = np.exp(sigma * np.sqrt(deltat))
        d = 1 / u
        p = (np.exp(r * deltat) - d) / (u - d)
        self.u = u
        self.p = p
        ##########################################################################################
        '''
        生成Nodes，计算S股票价格
        Nodes[i]是第i步所有的Node，从下往上，每个元素是list
        Nodes[i][j]是第i步的第j个Node，是一个字典，存储了S：股票价格，数；SP：所有的singular point集合，已排序，list；Aint：区间，2个元素的list。
        '''
        Nodes = []
        for i in range(0, n + 1):
            nn = []
            for j in range(0, i + 1):
                nn.append(dict(S=S0 * (u ** (2 * j - i)), SP=[], Aint=[]))
            Nodes.append(nn)
        self.Nodes = Nodes

        ##########################################################################################
        # 计算叶节点的singular point
        for j in range(0, n + 1):
            Nodes[n][j]['Aint'] = [
                S0 / (n + 1) * ((1 - d ** (n - j + 1)) / (1 - d) + d ** (n - j) * ((1 - u ** (j + 1)) / (1 - u) - 1)),
                S0 / (n + 1) * ((1 - u ** (j + 1)) / (1 - u) + u ** (j) * (
                        (1 - d ** (n - j + 1)) / (1 - d) - 1))]  # Aint放A的上下界
            if K < Nodes[n][j]['Aint'][0]:  # K<Amin
                Nodes[n][j]['SP'].append([Nodes[n][j]['Aint'][0],
                                          Nodes[n][j]['Aint'][0] - K])
                # 由于j=0和j=i的时候只有一个singular point,防止因为区间端点有两个算两次，下同
                if j != 0 and j != n: Nodes[n][j]['SP'].append([
                    Nodes[n][j]['Aint'][1],
                    Nodes[n][j]['Aint'][1] - K])

            elif K > Nodes[n][j]['Aint'][1]:  # K>Amax
                Nodes[n][j]['SP'].append([Nodes[n][j]['Aint'][0], 0])
                if j != 0 and j != n:
                    Nodes[n][j]['SP'].append([
                        Nodes[n][j]['Aint'][1]
                        , 0])
            else:  # 否则有3个singular point
                Nodes[n][j]['SP'].append([Nodes[n][j]['Aint'][0], 0])
                Nodes[n][j]['SP'].append([K, 0])
                Nodes[n][j]['SP'].append([Nodes[n][j]['Aint'][1],
                                          Nodes[n][j]['Aint'][1] - K])

        ##########################################################################################
        # 向前更新Aint区间
        for i in range(n - 1, -1, -1):
            for j in range(0, i + 1):
                # 更新Aint
                Nodes[i][j]['Aint'] = [((i + 2) * Nodes[i + 1][j + 1]['Aint'][0] - Nodes[i + 1][j + 1]['S']) / (i + 1),
                                       ((i + 2) * Nodes[i + 1][j]['Aint'][1] - Nodes[i + 1][j]['S']) / (i + 1)]

        # 算SP
        for i in range(n - 1, -1, -1):
            #print("i is {}".format(i))
            for j in range(0, i + 1):
                # 如果是最上面j==i或者最下面j==0的点，只有一个路径，直接算
                N = Nodes[i][j]  # 共享地址
                Nu = Nodes[i + 1][j + 1]  # 上涨的点
                Nd = Nodes[i + 1][j]  # 下跌的点
                if j == 0:
                    p1 = Nu['SP'][0][1]  # 上涨价
                    p2 = Nd['SP'][0][1]  # 下跌价
                    N['SP'].append([N['Aint'][0],
                                    np.exp(-r * deltat) * (p * p1 + (1 - p) * p2)])
                elif j == i:
                    p1 = Nu['SP'][0][1]
                    p2 = Nd['SP'][-1][1]
                    N['SP'].append([N['Aint'][0],
                                    np.exp(-r * deltat) * (p * p1 + (1 - p) * p2)])
                else:
                    # 不是最上面或者最下面的点
                    # B
                    BNodes = list()
                    for [a, Pa] in Nd['SP']:
                        # B=Decimal((((i+2)*a-S0*u**(2*j-i-1))/(i+1)))
                        B = (((i + 2) * a - S0 * u ** (2 * j - i - 1)) / (i + 1))
                        if B < N['Aint'][0] or B > N['Aint'][1]:  continue
                        Bup = ((i + 1) * B + Nu['S']) / (i + 2)
                        v1 = np.array(Nu['SP'], dtype=np.float64)  # 上涨端点的singular point集合
                        wh = sum(Bup > v1[:, 0]) - 1
                        '''
                        if v1[wh + 1, 0] - v1[wh, 0] == 0:
                            print('aaa')
                            while 1:
                                1
                        '''
                        p1 = (Bup - v1[wh, 0]) / (v1[wh + 1, 0] - v1[wh, 0]) * (v1[wh + 1, 1] - v1[wh, 1]) + v1[wh, 1]
                        # v1=Nodes[i+1][j+1]['SP']
                        # p1=_LPredict(Bup,v1)
                        p2 = Pa
                        P = np.exp(-r * deltat) * (p * p1 + (1 - p) * p2)
                        '''
                        if P < 0:
                            print(i, j, B, P)
                            '''
                        BNodes.append([B, P])
                    # C
                    CNodes = list()
                    for [a, Pa] in Nu['SP']:
                        # C=Decimal((((i+2)*a-S0*u**(2*j-i+1))/(i+1)))
                        C = (((i + 2) * a - S0 * u ** (2 * j - i + 1)) / (i + 1))
                        if C < N['Aint'][0] or C > N['Aint'][1]:  continue
                        Cdown = ((i + 1) * C + Nd['S']) / (i + 2)
                        v2 = np.array(Nd['SP'], dtype=np.float64)
                        wh = sum(Cdown > v2[:, 0]) - 1
                        p1 = Pa
                        p2 = (Cdown - v2[wh, 0]) / (v2[wh + 1, 0] - v2[wh, 0]) * (v2[wh + 1, 1] - v2[wh, 1]) + v2[wh, 1]
                        P = np.exp(-r * deltat) * (p * p1 + (1 - p) * p2)
                        CNodes.append([C, P])

                    # 合并B和C，排序

                    Nodeij = BNodes + CNodes
                    Nodeij.sort()

                    # 美式
                    if (AM == True):
                        Amax = Nodes[i][j]['Aint'][1]
                        Amin = Nodes[i][j]['Aint'][0]
                        if Amax - K > Nodeij[-1][1]:
                            if Amin - K >= Nodeij[0][1]:
                                Nodeij = []
                                Nodeij.append([Amin, Amin - K])
                                Nodeij.append([Amax, Amax - K])
                            else:

                                for ii, [A, P] in enumerate(Nodeij):
                                    if P < A - K:
                                        [Am, Pm] = Nodeij[ii - 1]
                                        Abar = (K * A - K * Am + A * Pm - Am * P) / (A - Am - P + Pm)
                                        break
                                del Nodeij[ii:]
                                # if Nodeij[ii-2][0]!=Am:print("111111111")
                                Nodeij.append([Abar, Abar - K])
                                Nodeij.append([Amax, Amax - K])

                    # 删除相同的值
                    Nodeijtemp = []
                    npNodeij = np.array(Nodeij)
                    Nodeijtemp = np.array(Nodeij[0]).reshape(1, 2)
                    for iter in range(len(Nodeij)):
                        if npNodeij[iter, 0] not in Nodeijtemp[:, 0]:
                            # reshape因为形状一样才能拼接
                            Nodeijtemp = np.concatenate((Nodeijtemp, npNodeij[iter, :].reshape(1, 2)), axis=0)
                        '''
                        if npNodeij[iter, 0] == npNodeij[iter - 1, 0] and npNodeij[iter, 1] != npNodeij[iter - 1, 1]:
                            Nodeijtemp[-1, 1] = npNodeij[iter, 1]
                            print(npNodeij[iter, 1], npNodeij[iter - 1, 1])
                            print("1111111111111")
                            '''
                    Nodeij = Nodeijtemp.tolist()
                    npNodeij = np.array(Nodeij)
                    '''
                    for item in range(1, len(Nodeij)):
                        if (npNodeij[item, 0] == npNodeij[item - 1, 0]):
                            print(item)
                    '''
                    # 删减近似
                    if self.Upper:
                        # 上近似
                        iSP = 0
                        while True:
                            NN = Nodeij.copy()
                            iSP += 1
                            if iSP >= len(NN) - 1:
                                break
                            [A0, P0] = NN[iSP - 1]
                            [A1, P1] = NN[iSP]
                            [A2, P2] = NN[iSP + 1]
                            PP1 = (P2 - P0) / (A2 - A0) * (A1 - A0) + P0
                            epsilon = PP1 - P1
                            if epsilon < h:
                                del Nodeij[iSP]
                                continue
                    else:
                        # 下近似
                        NN = Nodeij.copy()
                        iSP = 1
                        while True:
                            iSP += 1
                            if iSP >= len(NN) - 1:  # 如果点不够了
                                break
                            [A0, P0] = NN[iSP - 2]
                            [A1, P1] = NN[iSP - 1]
                            [A2, P2] = NN[iSP]
                            [A3, P3] = NN[iSP + 1]
                            if ((P1 - P0) * (A3 - A2) - (A1 - A0) * (P3 - P2)) == 0:
                                del NN[iSP - 1]
                                del NN[iSP]
                                iSP -= 1
                                continue
                            xbar = ((P2 - P0) * (A3 - A2) * (A1 - A0) + A0 * (P1 - P0) * (A3 - A2) - A2 * (P3 - P2) * (
                                    A1 - A0)) / ((P1 - P0) * (A3 - A2) - (A1 - A0) * (P3 - P2))
                            if xbar == A0 or xbar == A3:
                                del NN[iSP - 1]
                                del NN[iSP]
                                iSP -= 1
                                continue
                            '''
                            if A1 - A0 == 0:
                                print(i, j, 'level function')
                            '''
                            ybar = P0 + (xbar - A0) / (A1 - A0) * (P1 - P0)
                            NodeAppend = [xbar, ybar]
                            yybar = (P2 - P1) / (A2 - A1) * xbar + P1 - (P2 - P1) / (A2 - A1) * A1
                            epsilon = abs(yybar - ybar)
                            if epsilon < h:
                                NN[iSP] = NodeAppend
                                del NN[iSP - 1]
                                continue

                    for Node in NN:
                        Nodes[i][j]['SP'].append(Node)

        self.V = Nodes[0][0]['SP'][0][1]
        ##########################################################################################
        # 计算每个Node的F都是哪些
        endtime = dt.datetime.now()
        self.comtime = endtime - starttime
        
        
"""
以上为类定义
"""

SP1 = SBPModel(h=10**(-5),S0=100,K=90,n=30,T=1,sigma=0.2,r=0.1-0.03)
SP1.build(AM=True,Upper=False)

print(SP1.V)

        
