#!/usr/bin/env python3 zfz
# -*- coding: utf-8 -*-
'''
CRR Model
'''
import numpy as np
import datetime as dt
import json
import os
import sys
pa='/Users/apple/Documents/文稿/Models_programs/亚式期权/' # 文件存储路径
if os.path.exists(pa) == False: # 如果不存在这个路径就退出程序
    print('####################')
    print('请更改路径到一个存在的本地文件夹！！以斜杠\'/\'结尾！！')
    print('####################')
    sys.exit()
NN=5
class CRRModel:
    def __init__(self,S0,K,n,T,sigma,r): #K strike price，n步数，T存续期
        self.S0 = S0
        self.K = K
        self.n = n
        self.T = T
        self.sigma = sigma
        self.r = r
        self.model = 'exact CRR Binomial Method'
    def build(self,AM = True):
        if AM == True:
            self.type = 'American Asian Option'
        if AM == False:
            self.type = 'European Asian Option'
        starttime = dt.datetime.now() 
        ##########################################################################################
        #对原始数据做一定的处理
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
        trace = [] #trace存放每一步之前有多少个节点，
        for i in range(1,n+3): #trace[i]表示第i步前有多少个Node，如trace[1]=1，trace[2]=3；trace[-1]是所有Node的个数
            trace.append(sum(range(0,i))) #i从1取到n+2，所以这里range(0,i)当i=n+2时取到0到n+1
        
        S = [0 for i in range(0,trace[-1])] #S表示股价，Node(i,j)就可以表示为S[trace[i]+j]

        for i in range(0,n+1):
            for j in range(0,i+1):
                S[trace[i]+j] = S0*(u**(2*j-i)) #计算每个Node的股价
        ##########################################################################################
        #对每条路径计算其average
        A = {} #A储存每条路径的average
        Sp = {} #S储存每条路径经过哪些价格
        path = {} #记录每一步中都有哪些路径
        for i in range(1,n+1):
            if i % NN==1 and i != 1: #如果i是11，21，31...不包括1
                with open(''.join([pa,str(int(i/NN)),'.json']),'w') as fobject:
                    json.dump(path,fobject) #1.json中保存的为0-10的path，2.json中保存的为11-20的path，以此类推
                ppath=path[str(i-1)] #清除掉path
                path={}
                path[str(i-1)]=ppath
            path[str(i)] = [] #初始化该step的path
            if i == 1:
                    Sp['0'] = [S0,S[1]]
                    Sp['1'] = [S0,S[2]]
                    A['0'] = np.average(Sp['0'])
                    A['1'] = np.average(Sp['1'])
                    path['1'] = ['0','1']
            else:
                for j in path[str(i-1)]:
                    ustr = ''.join([j,'1']) #j这个路径下一步是up后的路径，如'101'的ustr就是'1011'
                    dstr = ''.join([j,'0']) #j这个路径下一步是up后的路径，如'101'的ustr就是'1010'
                    Sp[ustr] = Sp[j]+[Sp[j][-1]*u] #整个路径经过的价格
                    Sp[dstr] = Sp[j]+[Sp[j][-1]*d]
                    A[ustr] = np.average(Sp[ustr]) #这个路径的average
                    A[dstr] = np.average(Sp[dstr])
                    path[str(i)].append(ustr) #把这个路径计入path中
                    path[str(i)].append(dstr)
                    
        ##########################################################################################
        #对每条路径计算其option value
        V = {} #计算每个路径的价值
        for i in range(0,n+1):
            V[str(i)]={} #让V的每个元素也是一个dict
        #此时path字典包括最后的step，所以直接用
        for j in path[str(n)]: #先算到期时候的价值
            V[str(n)][j] = max(A[j]-K,0) 
        
        for i in range(n-1,0,-1):
            if i != n-1:
                V[str(i+2)]={}
            if i%NN==0: #如果i是10的倍数，则应该读取下一个json文件了，比如i=20，则读取2.json
                with open(''.join([pa,str(int(i/NN)),'.json'])) as fobject:
                    path=json.load(fobject)
            for j in path[str(i)]:
                ustr = ''.join([j,'1']) #j这个路径下一步是up后的路径，如'101'的ustr就是'1011'
                dstr = ''.join([j,'0']) #j这个路径下一步是up后的路径，如'101'的ustr就是'1010'
                if AM == True: #如果这是美式期权
                    EUvalue = np.e**(-deltat*r)*(p*V[str(i+1)][ustr]+(1-p)*V[str(i+1)][dstr]) #等到以后行权的价值
                    AMvalue = max(A[j]-K,0) #立即行权的价值
                    V[str(i)][j] = max(EUvalue,AMvalue) #该路径的价值
                if AM == False: #如果这是欧式期权
                    V[str(i)][j] = np.e**(-deltat*r)*(p*V[ustr]+(1-p)*V[dstr])
        V['0']['']=np.e**(-deltat*r)*(p*V['1']['1']+(1-p)*V['1']['0'])
        self.V = V['0']['']
        endtime = dt.datetime.now()
        print(endtime-starttime) 
        self.comtime = endtime-starttime
'''
##########################################################################################
以上为类定义
##########################################################################################
'''

z = CRRModel(S0=100,K=90,n=15,T=1,sigma=0.2,r=0.1-0.03)
z.build()









