#!/usr/bin/env python3 zfz
# -*- coding: utf-8 -*-
"""
Hull-White CRR Method
"""
import numpy as np
import datetime as dt

class HWModel:
    def __init__(self,h,S0,K,n,T,sigma,r): #h是HW模型的参数，K strike price，n步数，T存续期
        self.h = h
        self.S0 = S0
        self.K = K
        self.n = n
        self.T = T
        self.sigma = sigma
        self.r = r
        self.model = 'Hull-White Binomial Method'
    def build(self,AM = True):
        if AM == True:
            self.type = 'American Asian Option'
        if AM == False:
            self.type = 'European Asian Option'
        starttime = dt.datetime.now() 
        ##########################################################################################
        #对原始数据做一定的处理
        h = self.h
        S0 = self.S0
        K = self.K 
        n = self.n
        T = self.T
        sigma = self.sigma
        r = self.r
        deltat = T/n #每两个step之间的时间间隔
        u = np.e**(sigma*np.sqrt(deltat))
        d = 1/u
        p = (np.e**(r*deltat)-d)/(u-d)
        self.u = u
        self.p = p
        ##########################################################################################
        #计算股价S
        trace = [] #trace存放每一步之前有多少个节点，
        for i in range(1,n+3): #trace[i]表示第i步前有多少个Node，如trace[1]=1，trace[2]=3；trace[-1]是所有Node的个数
            trace.append(sum(range(0,i))) #i从1取到n+2，所以这里range(0,i)当i=n+2时取到0到n+1
        
        S = [0 for i in range(0,trace[-1])] #S表示股价，Node(i,j)就可以表示为S[trace[i]+j]
        MM = np.matrix([[0 for i in range(0,n+1)],[0 for i in range(0,n+1)]],dtype = np.float32) #MM储存每个step的A的最大值和最小值，第一行最大，第二行最小
        F = {} #F是一个字典，储存插值的端点值
        for i in range(0,n+1):
            for j in range(0,i+1):
                S[trace[i]+j] = S0*(u**(2*j-i)) #计算每个Node的股价
        self.S = S
        ##########################################################################################
        #计算每个Node的F都是哪些
        MM[0,0] = MM[1,0] = S0
        F[0] = np.matrix([[S0],[0]],dtype = np.float32)
        for i in range(1,n+1): #遍历每一个step
            MM[0,i] = (i*F[i-1][0,0]+S[trace[i]+0])/(i+1) #计算step i 时的minA
            MM[1,i] = (i*F[i-1][0,-1]+S[trace[i]+i])/(i+1) #计算step i 时的maxA
        
            
            k = 0
            semh = [S0] #semh即S0*np.e**(m*h)
            m = [0]
            while 1: #计算m最大达到多少的时候能够覆盖
                k += 1
                m.append(k)
                semh.append(S0*np.exp(k*h))
                if semh[-1] > MM[1,i]: #如果超过该Node的最大A值了
                    break
            k = 0
            while 1: #计算m最小达到多少的时候能够覆盖
                k -= 1
                m.insert(0,k)
                semh.insert(0,S0*np.exp(k*h))
                if semh[0] < MM[0,i]: #如果小于该Node的最小A值了
                    break
            F[i] = np.matrix([semh,m])

        ##########################################################################################
        V = {} #对应着某些F的option价值，索引为 i,j
        V[n,0] = [max(j-K,0) for j in np.array(F[n][0,:])[0]] #先定义最后一个step的期权价值
        for i in range(1,n+1):
            V[n,i] = V[n,0]
        #开始计算每个Node的价值
        for i in range(n-1,-1,-1): #遍历所有step（backward）
            for j in range(0,i+1): #遍历该step的所有Node
                UF = [] #储存该点的F如果upward的价值
                DF = [] #储存该点的F如果downward的价值
                V[i,j] = []
                FF1 = np.array(F[i][0,:])[0] #把这个step的F都拎出来
                FF2 = np.array(F[i+1][0,:])[0] #把下一个step的F都拎出来
                #若在下一个step是upward movement
                FF1U = ((i+1)*FF1+S[trace[i+1]+j+1])/(i+2)
                for k in FF1U: #FF1U即FF1中所有值upward之后相应的值
                    kk = [k for q in FF2] #把k重复FF2的元素个数那么多次
                    ind = sum(kk > FF2)-1 #找到k这个元素在FF2中的位置
                    UF.append((V[i+1,j+1][ind+1]-V[i+1,j+1][ind])*(k-FF2[ind])/(FF2[ind+1]-FF2[ind])+V[i+1,j+1][ind]) #插值
                #若在下一个step是downward movement
                FF1D = ((i+1)*FF1+S[trace[i+1]+j])/(i+2)
                for k in FF1D: #FF1D即FF1中所有值downward之后相应的值
                    kk = [k for q in FF2] #把k重复FF2的元素个数那么多次
                    ind = sum(kk > FF2)-1 #找到k这个元素在FF2中的位置
                    DF.append((V[i+1,j][ind+1]-V[i+1,j][ind])*(k-FF2[ind])/(FF2[ind+1]-FF2[ind])+V[i+1,j][ind]) #插值
                #如果计算欧式期权
                if AM == False:
                    V[i,j] = np.exp(-deltat*r)*(p*np.array(UF) + (1-p)*np.array(DF))
                #如果计算美式期权
                if AM == True:
                    EUvalue = np.exp(-deltat*r)*(p*np.array(UF) + (1-p)*np.array(DF)) #欧式期权的价值
                    AMvalue = [max(j-K,0) for j in FF1]#现在立即行权的价值
                    for k in range(0,len(FF1)): #对于每个F，都比较一下是等待的价值高，还是立即行权的价值高
                        V[i,j].append(max(EUvalue[k],AMvalue[k]))
        endtime = dt.datetime.now() 
        print((endtime-starttime))
        self.V = V[0,0]
        self.comtime = (endtime-starttime)
"""
##########################################################################################
以上为类定义
##########################################################################################
"""

x = HWModel(h=0.005,S0=100,K=90,n=25,T=1,sigma=0.2,r=0.1-0.03)
x.build(AM = True)










        
