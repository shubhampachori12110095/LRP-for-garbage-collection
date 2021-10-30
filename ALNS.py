# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 20:46:36 2021

@author: hyx
"""


from net import simpleGraph
import osmnx as nx
import networkx as nx
import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from gurobipy import *


def nodeGeneration():
    city = simpleGraph()
    city.addCollectPoi(1,1.72,2.28,360,369,554,403)
    city.addCollectPoi(2,2.77,3.49,581,379,319,470)
    city.addCollectPoi(3,4.5,4.72,575,442,520,321)
    city.addCollectPoi(4,5.55,2.44,370,536,498,412)
    city.addCollectPoi(5,6.43,3.52,459,270,509,277)
    city.addCollectPoi(6,6.46,5.69,252,321,574,421)
    city.addCollectPoi(7,2.51,6.52,244,367,383,354)
    city.addCollectPoi(8,3.69,7.52,429,279,357,368)
    city.addCollectPoi(9,5.44,8.47,301,227,270,526)
    city.addCollectPoi(10,2.33,9.38,514,574,272,445)
    city.addCollectPoi(11,7.32,7.27,344,270,357,274)
    city.addCollectPoi(12,9.58,7.6,318,229,207,531)
    city.addCollectPoi(13,8.6,5.51,298,271,215,497)
    city.addCollectPoi(14,8.7,2.86,317,571,403,555)
    city.addCollectPoi(15,7.19,1.75,269,571,533,418)
    city.addCollectPoi(16,3.4,1.51,286,462,570,345)
    city.addTransPoi(101,2.17,4.31)
    city.addTransPoi(102,7.49,3.72)
    city.addTransPoi(103,5.01,7.14)
    city.addTransPoi(104,2.0,8.0)
    city.addTransPoi(105,5.38,1.23)
    city.addGarPoi(1,1.62,1.26)
    city.addGarPoi(2,1.89,7.39)
    city.addGarPoi(3,8.6,4.49)
    city.addGarPoi(4,8.7,1.21)
    city.shortestRoute()
    return city

def initSolution(city,tCap,vCap):#upper truck capcity:6000?
    #allocate customer to its colsest transfer station
    dist1 = city.dist1.copy()
    dist2 = city.dist2.copy()
    total_trash = 0
    for c in city.colPoi:
        total_trash = total_trash+sum(city.demand[c].values())
    cap = dict(sorted(tCap.items(),key=lambda x:x[1],reverse=True))
    key = list(cap.keys())
    transPoi = []
    while 1:
        t = random.randint(0,len(key)-1)
        if total_trash >= cap[key[t]]:
            total_trash = total_trash-cap[key[t]]
            transPoi.append(key[t])
            del key[t]
        else:
            transPoi.append(key[t])
            break
    assign = {}
    w = {}#total weight of transfer station
    for t in city.transPoi:
        assign[t] = []
        w[t] = 0
    for c in city.colPoi:
        temp = 100000
        for t in transPoi:
            if dist2[c,t] < temp:
                temp = dist2[c,t]
                trans = t
        transPoi1 = transPoi.copy()
        while 1:
            if w[trans] + sum(city.demand[c].values()) <= tCap[trans]:
                assign[trans].append(c)
                w[trans] = w[trans] + sum(city.demand[c].values())
                break
            else:
                for i in range(len(transPoi1)):
                    if transPoi1[i] == trans:
                        del transPoi1[i]
                        break
                temp = 100000
                for t in transPoi1:
                    if dist2[c,t] < temp:
                        temp = dist2[c,t]
                        trans = t
    de = {}
    for t in city.transPoi:
        de[t] = {}
        de[t]['recycle'] = 0
        de[t]['others'] = 0
        de[t]['hazardous'] = 0
        de[t]['kitchen'] = 0
        for c in assign[t]:
            de[t]['recycle'] = de[t]['recycle']+city.demand[c]['recycle']
            de[t]['others'] = de[t]['others']+city.demand[c]['others']
            de[t]['hazardous'] = de[t]['hazardous']+city.demand[c]['hazardous']
            de[t]['kitchen'] = de[t]['kitchen']+city.demand[c]['kitchen']
    treat = {1:'recycle',2:'others',3:'hazardous',4:'kitchen'}
    
    save = {}
    for g in city.garPoi:
        save[g] = {}
        for i in transPoi:
            for j in transPoi:
                if i == j:
                    pass
                else:
                    if dist1[g,i]+dist1[g,j] >= dist1[i,j]:
                        save[g][i,j] = dist1[g,i]+dist1[g,j]-dist1[i,j]
                    else:
                        save[g][i,j] = 0
    route = {}
    belong = {}
    for g in city.garPoi:
        route[g] = {}
        temp = 1
        for i in transPoi:
            route[g][temp] = [i]
            temp = temp + 1
        belong[g] = dict((v[0],k) for k,v in route[g].items())    
    weight = {}
    for g in city.garPoi:
        weight[g] = {}
        for car in route[g]:
            node = route[g][car][0]
            weight[g][car] = de[node][treat[g]]
        for arc in save[g]:
            if belong[g][arc[0]] != belong[g][arc[1]]:
                k = belong[g][arc[1]]
                if len(route[g][k]) == 1:
                    pos1 = belong[g][arc[0]]
                    pos2 = belong[g][arc[1]]
                    if weight[g][pos1]+de[arc[1]][treat[g]] <= truck_cap:#truck capacity 6000?
                        weight[g][pos1] = weight[g][pos1]+de[arc[1]][treat[g]]
                        route[g][pos1].append(arc[1])
                        belong[g][arc[1]] = pos1
                        del route[g][pos2]
                        del weight[g][pos2]
                k = belong[g][arc[0]]
                if len(route[g][k]) == 1:
                    pos1 = belong[g][arc[0]]
                    pos2 = belong[g][arc[1]]
                    if weight[g][pos2]+de[arc[0]][treat[g]] <= truck_cap:
                        weight[g][pos2] = weight[g][pos2]+de[arc[1]][treat[g]]
                        route[g][pos2].insert(0,arc[0])
                        belong[g][arc[0]] = pos2
                        del route[g][pos1]
                        del weight[g][pos1]  
    temp = []
    for p in route.keys():
        temp.append(p)
        for r in route[p]:
            temp.extend(route[p][r])
            temp.append(0)
    route = temp
    
    #saving algorithm
    save_dist = {}
    for t in transPoi:
        save_dist[t] = {}
        node = assign[t]
        for i in node:
            for j in node:
                if i == j:
                    pass
                else:
                    if dist2[t,i]+dist2[t,j] >= dist2[i,j]:
                        save_dist[t][i,j] = dist2[t,i]+dist2[t,j]-dist2[i,j]
                    else:
                        save_dist[t][i,j] = 0
    for t in transPoi:
        save_dist[t] = sorted(save_dist[t].items(),key=lambda x:x[1],reverse=1)
        save_dist[t] = dict(save_dist[t])
    
    def init_route():
        #create initial route(each collection point corresponds to one route)
        route = {}
        belong = {}
        for t in city.transPoi:
            route[t] = {}
            temp = 1
            for i in assign[t]:
                route[t][temp] = [i]
                temp = temp + 1
            belong[t] = dict((v[0],k) for k,v in route[t].items())
        return route,belong
    
    
    def merge_route(route,belong,trash1,trash2):
        #merge route
        weight = {}
        for t in transPoi:
            weight[t] = {}#total weigh of the route
            for car in route[t]:
                weight[t][car] = []
                node = route[t][car][0]
                weight[t][car].append(city.demand[node][trash1])
                weight[t][car].append(city.demand[node][trash2])
            for arc in save_dist[t]:
                if belong[t][arc[0]] != belong[t][arc[1]]:
                    k = belong[t][arc[1]]
                    if len(route[t][k]) == 1:
                        pos1 = belong[t][arc[0]]#第一个节点属于的路线
                        pos2 = belong[t][arc[1]]#第二个节点属于的路线
                        #when new node is added, the capacity limit is not exceeded 
                        if weight[t][pos1][0]+city.demand[arc[1]][trash1] <= vCap/2 and  \
                            weight[t][pos1][1]+city.demand[arc[1]][trash2] <= vCap/2:
                            weight[t][pos1][0] = weight[t][pos1][0]+city.demand[arc[1]][trash1]
                            weight[t][pos1][1] = weight[t][pos1][1]+city.demand[arc[1]][trash2]
                            route[t][pos1].append(arc[1])
                            belong[t][arc[1]] = pos1
                            del route[t][pos2]
                            del weight[t][pos2]
                    k = belong[t][arc[0]]
                    if len(route[t][k]) == 1:
                        pos1 = belong[t][arc[0]]
                        pos2 = belong[t][arc[1]]
                        if weight[t][pos2][0]+city.demand[arc[0]][trash1] <= vCap/2 and  \
                            weight[t][pos2][1]+city.demand[arc[0]][trash2] <= vCap/2:
                            weight[t][pos2][0] = weight[t][pos2][0]+city.demand[arc[0]][trash1]
                            weight[t][pos2][1] = weight[t][pos2][1]+city.demand[arc[0]][trash2]
                            route[t][pos2].insert(0,arc[0])
                            belong[t][arc[0]] = pos2
                            del route[t][pos1]
                            del weight[t][pos1]
        return route
    route1,belong1 = init_route()
    route2,belong2 = init_route()
    route1 = merge_route(route1,belong1,'recycle','others')
    route2 = merge_route(route2,belong2,'hazardous','kitchen')
    
    temp = []
    for t in route1:
        temp.append(t)
        for r in route1[t]:
            temp.extend(route1[t][r])
            temp.append(0)
    route1 = temp
    
    temp = []
    for t in route2:
        temp.append(t)
        for r in route2[t]:
            temp.extend(route2[t][r])
            temp.append(0)  
    route2 = temp
    route = [route,route1,route2]
    return assign,route

def objective(route,upper_penality=1000,lower_penality=1000,trans_penality=1000):
    obj = 0
    de = {}
    for i in range(len(route[1])):
        if route[1][i] in city.transPoi:
            de[route[1][i]] = {}
            de[route[1][i]]['recycle'] = 0
            de[route[1][i]]['others'] = 0
            de[route[1][i]]['hazardous'] = 0
            de[route[1][i]]['kitchen'] = 0
            if i == len(route[1])-1:#该转运站一定不开放
                continue
            for j in range(i+1,len(route[1])):
                if route[1][j] in city.colPoi:
                    de[route[1][i]]['recycle'] = de[route[1][i]]['recycle']+city.demand[route[1][j]]['recycle']
                    de[route[1][i]]['others'] = de[route[1][i]]['others']+city.demand[route[1][j]]['others']
                    de[route[1][i]]['hazardous'] = de[route[1][i]]['hazardous']+city.demand[route[1][j]]['hazardous']
                    de[route[1][i]]['kitchen'] = de[route[1][i]]['kitchen']+city.demand[route[1][j]]['kitchen']
                elif route[1][j] == 0:#最后一个站开放
                    if j == len(route[1])-1:
                        obj = obj + oc_trans
                    if sum(de[route[1][i]].values()) > tCap[route[1][i]]:#违背容量约束
                        obj = obj+trans_penality*(sum(de[route[1][i]].values())-tCap[route[1][i]])
                    continue
                else:
                    if j != i+1:#i开放
                        obj = obj+oc_trans
                    if sum(de[route[1][i]].values()) > tCap[route[1][i]]:#违背容量约束
                        obj = obj+trans_penality*(sum(de[route[1][i]].values())-tCap[route[1][i]])
                    break
    for i in range(3):#车辆固定成本
        for j in route[i]:
            if j == 0:
                obj = obj + oc_vehicle
    treat = {1:'recycle',2:'others',3:'hazardous',4:'kitchen'}
    for i in range(len(route[0])):
        if route[0][i] in city.garPoi:
            j = i
            w = 0
            while 1:
                if route[0][j+1] != 0:
                    obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                       *city.dist1[route[0][j],route[0][j+1]])+ \
                        unit_cost*const_value*beta*gamma*city.dist1[route[0][j],route[0][j+1]]*w
                else:
                    obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                       *city.dist1[route[0][j],route[0][i]])+ \
                        unit_cost*const_value*beta*gamma*city.dist1[route[0][j],route[0][i]]*w
                j = j+1
                if route[0][j] != 0:
                    w = w + de[route[0][j]][treat[route[0][i]]]
                    if w > truck_cap:#truck capacity
                        obj = obj + upper_penality*(w-truck_cap)
                else:
                    if j != len(route[0])-1:
                        if route[0][j+1] in city.garPoi:#当下一个点是处理厂
                            break
                        else:
                            j = j+1#新的一辆车
                            obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                           *city.dist1[route[0][i],route[0][j]])
                            w = de[route[0][j]][treat[route[0][i]]]
                            if w > truck_cap:#truck capacity
                                obj = obj + upper_penality*(w-truck_cap)
                    else:
                        break             
    for i in range(len(route[1])):
        if route[1][i] in city.transPoi:
            if i == len(route[1])-1:
                continue
            if route[1][i+1] in city.transPoi:#该点不开放
                continue
            j = i
            w1 = 0
            w2 = 0
            while 1:
                if route[1][j+1] != 0:
                    obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                       *city.dist2[route[1][j],route[1][j+1]])+ \
                        unit_cost*const_value*beta*gamma*city.dist2[route[1][j],route[1][j+1]]*(w1+w2)
                else:
                    obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                       *city.dist2[route[1][j],route[1][i]])+ \
                        unit_cost*const_value*beta*gamma*city.dist2[route[1][j],route[1][i]]*(w1+w2)
                j = j+1
                if route[1][j] != 0:
                    w1 = w1 + city.demand[route[1][j]]['recycle']
                    w2 = w2 + city.demand[route[1][j]]['others']
                    if w1 > vCap/2:
                        obj = obj + lower_penality*(w1-vCap/2)
                    if w2 > vCap/2:
                        obj = obj + lower_penality*(w2-vCap/2)
                else:
                    if j != len(route[1])-1:
                        if route[1][j+1] in city.transPoi:#当下一个点是转运站
                            break
                        else:
                            j = j+1#新的一辆车
                            obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                           *city.dist2[route[1][i],route[1][j]])
                            w1 = city.demand[route[1][j]]['recycle']
                            w2 = city.demand[route[1][j]]['others']
                            if w1 > vCap/2:
                                obj = obj + lower_penality*(w1-vCap/2)
                            if w2 > vCap/2:
                                obj = obj + lower_penality*(w2-vCap/2)
                    else:
                        break                
    for i in range(len(route[2])):
        if route[2][i] in city.transPoi:
            if i == len(route[2])-1:
                continue
            if route[2][i+1] in city.transPoi:
                continue
            j = i
            w1 = 0
            w2 = 0
            while 1:
                if route[2][j+1] != 0:#到达下一个收集点
                    obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                       *city.dist2[route[2][j],route[2][j+1]])+ \
                        unit_cost*const_value*beta*gamma*city.dist2[route[2][j],route[2][j+1]]*(w1+w2)
                else:#返回转运站
                    obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                       *city.dist2[route[2][j],route[2][i]])+ \
                        unit_cost*const_value*beta*gamma*city.dist2[route[2][j],route[2][i]]*(w1+w2)
                j = j+1
                if route[2][j] != 0:
                    w1 = w1 + city.demand[route[2][j]]['hazardous']
                    w2 = w2 + city.demand[route[2][j]]['kitchen']
                    if w1 > vCap/2:
                        obj = obj + lower_penality*(w1-vCap/2)
                    if w2 > vCap/2:
                        obj = obj + lower_penality*(w2-vCap/2)
                else:
                    if j != len(route[2])-1:#j不是最后一个点
                        if route[2][j+1] in city.transPoi:#当下一个点是转运站
                            break
                        else:
                            j = j+1#新的一辆车
                            obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                           *city.dist2[route[2][i],route[2][j]])
                            w1 = city.demand[route[2][j]]['hazardous']
                            w2 = city.demand[route[2][j]]['kitchen']
                            if w1 > vCap/2:
                                obj = obj + lower_penality*(w1-vCap/2)
                            if w2 > vCap/2:
                                obj = obj + lower_penality*(w2-vCap/2)
                    else:
                        break
    return obj

class ALNS:
    def __init__(self,route):
        self.route1 = route[0].copy()#upper path
        self.route2 = route[1].copy()
        self.route3 = route[2].copy()
        self.q = math.ceil(0.15*len(city.colPoi))#destroy numbers
        self.open = []
        pos1 = self.route1.index(1)
        pos2 = self.route1.index(2)
        for i in range(pos1+1,pos2):
            if self.route1[i] != 0:
                self.open.append(self.route1[i])
        print(self.open)
        self.currentSolu = copy.deepcopy(route)
        self.bestSolu = copy.deepcopy(route)
        self.temperature = 0.1*objective(route)#初始温度
        self.cooling_ratio = 0.99#衰减率
        self.max_iteration = 2000#最大迭代次数
        self.max_unimproved_iteration = 50
        self.process = []
        

        self.points_destroy_score = {'rand_remove':[0,0],'greedy_remove':[0,0],'related_remove':[0,0]}
        self.repair_score = {'greedy_insert':[0,0],'greedy_insert_perturbation':[0,0],'regret_k_insert':[0,0]}
        self.station_destroy_weight = {'station_remove':1,'station_swap':2,'station_open':1,
                                       'greedy_station_swap':2}
        self.points_destroy_weight = {'rand_remove':1,'greedy_remove':1,'related_remove':1}
        self.repair_weight = {'greedy_insert':1,'greedy_insert_perturbation':1,'regret_k_insert':1}
        self.weight_update_segment = 50#每50次迭代更新算子权重
        
        
    def cost(self,route,trashtype,lower_penality=1000,trans_penality=1000):#trashtype=1,为可回收垃圾和其他垃圾
        obj = 0
        de = {}
        for i in range(len(route)):
            if route[i] in city.transPoi:
                de[route[i]] = {}
                de[route[i]]['recycle'] = 0
                de[route[i]]['others'] = 0
                de[route[i]]['hazardous'] = 0
                de[route[i]]['kitchen'] = 0
                if i == len(route)-1:#该转运站一定不开放
                    continue
                for j in range(i+1,len(route)):
                    if route[j] not in city.transPoi and route[j] != 0:
                        de[route[i]]['recycle'] = de[route[i]]['recycle']+city.demand[route[j]]['recycle']
                        de[route[i]]['others'] = de[route[i]]['others']+city.demand[route[j]]['others']
                        de[route[i]]['hazardous'] = de[route[i]]['hazardous']+city.demand[route[j]]['hazardous']
                        de[route[i]]['kitchen'] = de[route[i]]['kitchen']+city.demand[route[j]]['kitchen']
                    elif route[j] == 0:#最后一个站开放
                        if j == len(route)-1:
                            obj = obj + oc_trans
                            if sum(de[route[i]].values()) > tCap[route[i]]:#违背容量约束
                                obj = obj+trans_penality*(sum(de[route[i]].values())-tCap[route[i]])
                        continue
                    else:
                        if j != i+1:#i开放
                            obj = obj+oc_trans
                        if sum(de[route[i]].values()) > tCap[route[i]]:#违背容量约束
                            obj = obj+trans_penality*(sum(de[route[i]].values())-tCap[route[i]])                
                        break
        obj = obj+route.count(0)*oc_vehicle#fixed cost of vehicle
        for i in range(len(route)):
            if route[i] in city.transPoi:
                if i == len(route)-1:#最后一个点是转运站，一定不开放
                    continue
                if route[i+1] in city.transPoi:#该点不开放
                    continue
                j = i
                w1 = 0
                w2 = 0
                while 1:
                    if route[j+1] != 0:
                        obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                           *city.dist2[route[j],route[j+1]])+ \
                            unit_cost*const_value*beta*gamma*city.dist2[route[j],route[j+1]]*(w1+w2)
                    else:
                        obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                           *city.dist2[route[j],route[i]])+ \
                            unit_cost*const_value*beta*gamma*city.dist2[route[j],route[i]]*(w1+w2)
                    j = j+1
                    if route[j] != 0:
                        if trashtype == 1:
                            w1 = w1 + city.demand[route[j]]['recycle']
                            w2 = w2 + city.demand[route[j]]['others']
                        else:
                            w1 = w1 + city.demand[route[j]]['hazardous']
                            w2 = w2 + city.demand[route[j]]['kitchen']
                        if w1 > vCap/2:
                            obj = obj + lower_penality*(w1-vCap/2)#penalty coefficent
                        if w2 > vCap/2:
                            obj = obj + lower_penality*(w2-vCap/2)
                    else:
                        if j != len(route)-1:
                            if route[j+1] in city.transPoi:#当下一个点是转运站
                                break
                            else:
                                j = j+1#新的一辆车
                                obj = obj + unit_cost*const_value*((delta+alpha*gamma+curb_weigh*beta*gamma)#fixed cost
                                                               *city.dist2[route[i],route[j]])
                                if trashtype == 1:
                                    w1 = city.demand[route[j]]['recycle']
                                    w2 = city.demand[route[j]]['others']
                                else:
                                    w1 = city.demand[route[j]]['hazardous']
                                    w2 = city.demand[route[j]]['kitchen']
                        else:
                            break
        return obj
        
    def route_normalization(self):
        #[101, 28, 0, 3, 24, 0, 102, 27, 2, 0, 10, 11, 0, 23, 6, 0, 103, 18, 20, 0, 22, 7, 0, 16, 0, 31, 0, 104, 14, 0, 
        #33, 0, 8, 0, 1, 0, 105, 26, 17, 0, 30, 0, 32, 19, 0, 25, 0, 106, 107, 108, 29, 0, 13, 0, 15, 0,0]

        for i in range(len(self.route2)-1):#
            #考虑整条路径被删除，出现连续两个0或空转运站后面跟着0
            if self.route2[i]==0 or (self.route2[i] in city.transPoi and i!=len(self.route2)-1):
                j = i+1
                while 1:
                    if j == len(self.route2):
                        break
                    if self.route2[j] == 0:
                        del self.route2[j]
                        if j == len(self.route2):
                            break
                    else:
                        break
            if i == len(self.route2)-1:
                break
        for i in range(len(self.route3)-1):
            if self.route3[i]==0 or (self.route3[i] in city.transPoi and i!= len(self.route3)-1):
                j = i+1
                while 1:
                    if j == len(self.route3):
                        break
                    if self.route3[j] == 0:
                        del self.route3[j]      
                        if j == len(self.route3):
                            break
                    else:
                        break
            if i == len(self.route3)-1:
                break
        
    def rand_remove(self):
        length = len(self.route2)
        p = 0
        pool = []
        while 1:
            t = random.randint(0,length-1)
            if self.route2[t] in city.colPoi:
                poi = self.route2[t]#删除点
                del self.route2[t]
                length = length-1
                self.route3.remove(poi)
                p = p+1
                pool.append(poi)
            if p == self.q:
                break
        self.route_normalization()
        return pool

    def upper_optimize(self):
        final_route = []
        trans = len(self.open)#开放转运站的个数
        group = []
        for i in range(2**trans):
            group.append(bin(i).replace('0b','').rjust(trans,'0'))
        de = {}
        for t in self.open:
            de[t] = {1:0,2:0,3:0,4:0}
            start = self.route2.index(t)
            if t != list(city.transPoi.keys())[-1]:
                end = self.route2.index(t+1)
            else:
                end = len(self.route2)
            for i in range(start+1,end):
                if self.route2[i] != 0:
                    de[t][1] = de[t][1]+city.demand[self.route2[i]]['recycle']
                    de[t][2] = de[t][2]+city.demand[self.route2[i]]['others']
                    de[t][3] = de[t][3]+city.demand[self.route2[i]]['hazardous']
                    de[t][4] = de[t][4]+city.demand[self.route2[i]]['kitchen']#第t个转运站需要处理厂运输的垃圾
        for g in range(1,5):
            start = self.route1.index(g)
            if g < 4:
                end = self.route1.index(g+1)
            else:
                end = len(self.route1)
            a = [0]*trans
            w = 0
            cost = 0
            columns = []
            routes = 0
            costs = []
            sequences = []
            sequence = []
            for i in range(start,end):#初始解为当前的route1
                if i != len(self.route1)-1:
                    if self.route1[i+1] in self.open:
                        a[self.open.index(self.route1[i+1])] = 1
                        sequence.append(self.route1[i+1])
                        if self.route1[i] == 0:
                            cost = cost+unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)* \
                                city.dist1[g,self.route1[i+1]]+ \
                                unit_cost*const_value*beta*gamma*city.dist1[g,self.route1[i+1]]*w
                        else:
                            cost = cost+unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)* \
                                city.dist1[self.route1[i],self.route1[i+1]]+ \
                                unit_cost*const_value*beta*gamma*city.dist1[self.route1[i],self.route1[i+1]]*w
                        w = w+de[self.route1[i+1]][g]
                    else:#一辆车完成行程
                        if i != end-1:
                            cost = cost+unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)* \
                                city.dist1[self.route1[i],g]+ \
                                unit_cost*const_value*beta*gamma*city.dist1[self.route1[i],g]*w
                            w = 0#清空重量
                            costs.append(cost)
                            cost = 0#成本清零
                            columns.append(a)
                            sequences.append(sequence)
                            sequence = []
                            routes = routes+1
                            a = [0]*trans
                        else:
                            pass
                else:
                    pass
            a = {}
            route = []
            for i in range(routes):
               route.append(i) 
            for i in range(len(route)):#第几条路
                for j in range(len(self.open)):#哪个转运站
                    a[self.open[j],route[i]] = columns[i][j]
            mp = Model()
            mp.update()
            y = mp.addVars(route,lb=0,ub=1,obj=costs,name='y')
            mp.addConstrs(quicksum(a[i,l]*y[l] for l in route)==1 for i in self.open)
            mp.setParam(GRB.Param.LogToConsole,0)
            mp.optimize()
            '''
            origin_route = []
            for ro in range(len(columns)):
                temp = []
                for t in range(len(columns[ro])):
                    if columns[ro][t] == 1:
                        temp.append(self.open[t])
                origin_route.append(temp)
            '''
            temp = []
            for v in mp.getVars():
                temp.append(v)
            r = dict(list(zip(temp,sequences)))
            dual = mp.getAttr(GRB.Attr.Pi,mp.getConstrs())
            pi = {}
            pi = dict(list(zip(self.open,dual)))
            pi[0] = 0 
            times=0
            min_rc = -10
            while min_rc < -1 and times<=20:
                flow = {}
                seq = {}
                for i in self.open:
                    flow[i,()] = unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)*city.dist1[g,i]#直接到达每个转运站
                    seq[i,()] = [i]
                c = copy.deepcopy(flow)
                for i in range(1,trans):#经过城市个数
                    for t in self.open:
                        visited = self.open.copy()
                        visited.remove(t)#可能经过的点
                        pos = self.open.index(t)
                        choose = []
                        for j in group:
                            if j[pos] == '1':#到达转运站t
                                j = j[0:pos]+j[pos+1:trans]
                                if j.count('1') == i:#开放个数为i个的子集
                                    choose.append(j)
                        temp2 = []
                        for j in choose:
                            temp1 = []
                            for p in range(len(j)):
                                if j[p] == '1':
                                    temp1.append(visited[p])#经过城市组合
                            temp2.append(temp1)
                        for temp in temp2:#对于每一个组合
                            mini = np.Infinity
                            w = 0
                            for p in temp:
                                w = w+de[p][g]#经过城市总重量
                            if w+de[t][g] >= truck_cap:
                                break
                            for p in temp:#从p到t
                                temp1 = temp.copy()
                                temp1.remove(p)
                                temp1 = tuple(temp1)
                                if flow[p,temp1]+unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)*city.dist1[p,t]-pi[p]+ \
                                    unit_cost*const_value*beta*gamma*city.dist1[p,t]*w<= mini:#拥有最小检验数的路线
                                    #找到最小的经过p到达t的路
                                    mini = flow[p,temp1]+unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)*city.dist1[p,t]-pi[p]+ \
                                        unit_cost*const_value*beta*gamma*city.dist1[p,t]*w#拥有最小cost的路
                                    cost=c[p,temp1]+unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)*city.dist1[p,t]+ \
                                        unit_cost*const_value*beta*gamma*city.dist1[p,t]*w
                                    sequence = copy.deepcopy(seq[p,temp1])
                                    sequence.append(t)
                                    seq[t,tuple(temp)] = sequence#访问顺序
                            temp = tuple(temp)
                            flow[t,temp] = mini
                            c[t,temp] = cost
                for i in flow:
                    w = de[i[0]][g]
                    for p in i[1]:
                        w = w+de[p][g]
                    flow[i] = flow[i]+unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)*city.dist1[i[0],g]-pi[i[0]]+ \
                                unit_cost*const_value*beta*gamma*city.dist1[i[0],g]*w
                    c[i] = c[i]+unit_cost*const_value*(delta+alpha*gamma+curb_weigh*beta*gamma)*city.dist1[i[0],g]+ \
                                unit_cost*const_value*beta*gamma*city.dist1[i[0],g]*w
                min_rc = sorted(flow.items(),key=lambda x:x[1])[0][1]#最小reduced cost
                if min_rc > -1:
                    break
                min_route = sorted(flow.items(),key=lambda x:x[1])[0][0]#最小reduced cost对应的key
                cost = c[min_route]#该路径的成本
                poi = seq[min_route]
                b = []
                for i in self.open:
                    if i in poi:
                        b.append(1)
                    else:
                        b.append(0)
                column=Column(b,mp.getConstrs())
                new_route = mp.addVar(obj=cost,column=column,name='r%g'%(times))
                mp.optimize()
                r[new_route] = seq[min_route]#变量对应的路径
                #print('%g:%g'%(g,mp.objVal))
                dual = mp.getAttr(GRB.Attr.Pi,mp.getConstrs())#得到对偶变量
                pi = {}
                pi = dict(list(zip(self.open,dual)))
                pi[0] = 0
                times = times+1#遇到最优解有0.5的情况
            for v in mp.getVars():
                v.vtype = GRB.BINARY
            mp.optimize()
            final_route.append(g)
            for v in mp.getVars():
                if v.x>0:
                    for p in r[v]:
                        final_route.append(p)
                    #print(v.varName,':',v.x)
                    final_route.append(0)
        self.route1 = copy.deepcopy(final_route)
        #print(self.route1)

    def greedy_remove(self):
        pool = []
        obj = self.cost(self.route2,1)+self.cost(self.route3,2)#下层路径的总成本
        delta_obj = {}
        for i in range(len(self.route2)):
            route2 = self.route2.copy()
            route3 = self.route3.copy()
            if self.route2[i] in city.colPoi:
                if (route2[i-1] == 0 or route2[i-1] in city.transPoi) and route2[i+1] == 0:#一条路一辆车
                    del route2[i+1]
                temp = route3.index(self.route2[i])
                route2.remove(self.route2[i])
                if (route3[temp-1] == 0 or route3[temp-1] in city.transPoi) and route3[temp+1] == 0:
                    del route3[temp+1]
                route3.remove(self.route2[i])#删除该点
                delta_obj[self.route2[i]] = obj-self.cost(route2,1)-self.cost(route3,2)#下层减少成本            
        delta_obj = dict(sorted(delta_obj.items(),key=lambda x:x[1],reverse=True))        
        p = 0
        for key in delta_obj:
            if p == self.q:
                break
            else:
                self.route2.remove(key)            
                self.route3.remove(key)
                pool.append(key)
                p = p+1
        self.route_normalization()
        return pool

    def related_remove(self):
        pool = []
        while 1:
            pos = random.randint(0,len(self.route2)-1)#seed pos
            if self.route2[pos] in city.colPoi:
                break
        dist = {}
        for key in city.dist2:
            if key[0] == self.route2[pos] and key[1] in city.colPoi:
                dist[key] = city.dist2[key]
        dist = dict(sorted(dist.items(),key=lambda x:x[1],reverse=False))
        dist = list(dist.keys())
        dist = [p[1] for p in dist]
        for i in range(self.q):
            pool.append(dist[i])
            self.route2.remove(dist[i])
            self.route3.remove(dist[i])
        self.route_normalization()
        return pool

    def station_remove(self):
        total_trash = 0
        for c in city.colPoi:
            total_trash = total_trash+sum(city.demand[c].values())
        cap = dict(sorted(tCap.items(),key=lambda x:x[1],reverse=True))
        mini = 0#minium needed transfer staions
        for t in cap:
            if total_trash >= cap[t]:
                mini = mini+1
                total_trash = total_trash-cap[t]
            else:
                mini = mini+1
                break
        if len(self.open)-1 >= mini:
            pos = random.randint(0, len(self.open)-1)#randomly remove a station
            if self.open[pos] != list(city.transPoi.keys())[-1]:
                start1 = self.route2.index(self.open[pos])
                end1 = self.route2.index(self.open[pos]+1)
                start2 = self.route3.index(self.open[pos])
                end2 = self.route3.index(self.open[pos]+1)
            else:
                start1 = self.route2.index(self.open[pos])
                end1 = len(self.route2)
                start2 = self.route3.index(self.open[pos])
                end2 = len(self.route3)
            pool = self.route2[start1+1:end1]
            pool = [i for i in pool if i!=0]
            del self.route2[start1+1:end1]
            del self.route3[start2+1:end2]
            for i in range(4):
                pos1 = self.route1.index(self.open[pos])
                if (self.route1[pos1-1] == 0 or self.route1[pos1-1] in city.garPoi) and self.route1[pos1+1] == 0:#
                    del self.route1[pos1+1]
                del self.route1[pos1]
            del self.open[pos]
        else:
            pool = self.station_swap()
        return pool

    def greedy_station_swap(self):#关闭平均服务成本最高的转运站，以距离为权重选择另一家转运站
        if len(self.open) < len(city.transPoi):
            close = list(city.transPoi.keys())
            for i in self.open:
                if i in close:
                    close.remove(i)
            average_cost = {}
            for t in self.open:
                route1 = self.route1.copy()
                route2 = self.route2.copy()
                route3 = self.route3.copy()
                start2 = route2.index(t)
                start3 = route3.index(t)
                if t != list(city.transPoi.keys())[-1]:
                    end2 = route2.index(t+1)                    
                    end3 = route3.index(t+1)
                else:
                    end2 = len(route2)
                    end3 = len(route3)
                count = 0
                for i in range(start2+1,end2):
                    if route2[i] != 0:
                        count+=1#计算有多少服务点
                del route2[start2+1:end2]
                del route3[start3+1:end3]
                for i in range(4):
                    pos = route1.index(t)
                    if (route1[pos-1] in city.garPoi or route1[pos-1]==0) and route1[pos+1]==0:#一个站一条路
                        del route1[pos+1]
                    del route1[pos]
                delta_obj = objective(self.currentSolu)-objective([route1,route2,route3])
                average_cost[t] = delta_obj/count
            choose = sorted(average_cost.items(),key=lambda x:x[1],reverse=True)[0][0]
            dist = {}
            for i in close:
                dist[choose,i] = city.dist2[choose,i]#根据距离选择权重
            total_dist = sum(dist.values())
            rand = random.random()*total_dist
            for key in dist:
                if rand <= dist[key]:
                    chooseOpen = key[1]#选中开放的转运站
                    break
                else:
                    rand = rand-dist[key]
            start2 = self.route2.index(choose)
            start3 = self.route3.index(choose)
            if choose != list(city.transPoi.keys())[-1]:
                end2 = self.route2.index(choose+1)
                end3 = self.route3.index(choose+1)
            else:
                end2 = len(self.route2)
                end3 = len(self.route3)
            pool = self.route2[start2+1:end2]
            pool = [i for i in pool if i!=0]
            del self.route2[start2+1:end2]
            del self.route3[start3+1:end3]
            for i in range(len(self.route1)):
                if self.route1[i] == choose:
                    self.route1[i] = chooseOpen
            self.open.remove(choose)
            self.open.append(chooseOpen)
        else:
            pool = self.station_remove()
        return pool
                    
    def station_swap(self):
        if len(self.open) < len(city.transPoi):
            pos1 = random.randint(0, len(self.open)-1)#closed station
            close = list(city.transPoi.keys())
            for i in self.open:
                if i in close:
                    close.remove(i)
            dist = {}
            for i in close:
                dist[self.open[pos1],i] = city.dist2[self.open[pos1],i]
            total_dist = sum(dist.values())
            rand = random.random()*total_dist#roulette
            for key in dist:
                if rand <= dist[key]:
                    pos2 = key[1]
                else:
                    rand = rand-dist[key]
            if self.open[pos1] != list(city.transPoi.keys())[-1]:
                start1 = self.route2.index(self.open[pos1])
                end1 = self.route2.index(self.open[pos1]+1)#next station
                start2 = self.route3.index(self.open[pos1])
                end2 = self.route3.index(self.open[pos1]+1)
            else:
                start1 = self.route2.index(self.open[pos1])
                end1 = len(self.route2)
                start2 = self.route3.index(self.open[pos1])
                end2 = len(self.route3)
            pool = self.route2[start1+1:end1]
            pool = [i for i in pool if i!=0]
            del self.route2[start1+1:end1]
            del self.route3[start2+1:end2]
            for i in range(len(self.route1)):
                if self.route1[i] == self.open[pos1]:
                    self.route1[i] = pos2
            del self.open[pos1]
            self.open.append(pos2)
        else:
            pool = self.station_remove()
        return pool
         
    def station_open(self):
        if len(self.open) < len(city.transPoi):
            pool = []
            if len(self.open) == len(city.transPoi):
                self.station_swap()
            else:
                close = list(city.transPoi.keys())
                for i in self.open:
                    if i in close:
                        close.remove(i)
                pos = random.randint(0, len(close)-1)#randomly open a station
                dist = {}
                for key in city.dist2:
                    if key[0] == close[pos] and key[1] in city.colPoi:
                        dist[key] = city.dist2[key]
                dist = dict(sorted(dist.items(),key=lambda x:x[1],reverse=False))
                dist = [key[1] for key in list(dist.keys())]
                for i in range(self.q):
                    self.route2.remove(dist[i])
                    self.route3.remove(dist[i])
                    pool.append(dist[i])
                self.open.append(close[pos])
            self.route_normalization()
            pos = self.route1.index(2)
            self.route1.insert(pos,0)
            self.route1.insert(pos,self.open[-1])
            pos = self.route1.index(3)
            self.route1.insert(pos,0)
            self.route1.insert(pos,self.open[-1])
            pos = self.route1.index(4)
            self.route1.insert(pos,0)
            self.route1.insert(pos,self.open[-1])
            self.route1.append(self.open[-1])
            self.route1.append(0)
        else:
            pool = self.station_remove()
        return pool
            
    def greedy_insert(self,pool):
        #random.shuffle(pool)#打乱插入点的顺序
        for c in pool:
            minCost = {}
            best_insert = {}
            for pos in self.open:
                if pos != list(city.transPoi.keys())[-1]:#不是最后一个站
                    start1 = self.route2.index(pos)
                    end1 = self.route2.index(pos+1)#next station
                    start2 = self.route3.index(pos)
                    end2 = self.route3.index(pos+1)
                else:
                    start1 = self.route2.index(pos)
                    end1 = len(self.route2)#next station
                    start2 = self.route3.index(pos)
                    end2 = len(self.route3)
                minCost1 = np.Infinity
                for i in range(start1+1,end1+1):
                    temp = self.route2.copy()
                    if i == end1:
                        temp.insert(i,0)#create a new road
                        temp.insert(i,c)
                    else:
                        temp.insert(i,c)
                    delta_cost1 = self.cost(temp,1)-self.cost(self.route2,1)#考虑全部成本
                    if delta_cost1 < minCost1:
                        best_insert1 = temp
                        minCost1 = delta_cost1
                minCost2 = np.Infinity
                for i in range(start2+1,end2+1):
                    temp = self.route3.copy()
                    if i == end2:
                        temp.insert(i,0)
                        temp.insert(i,c)#create a new road
                    else:
                        temp.insert(i,c)
                    delta_cost2 = self.cost(temp,2)-self.cost(self.route3,2)
                    if delta_cost2 < minCost2:
                        best_insert2 = temp
                        minCost2 = delta_cost2
                minCost[pos] = delta_cost1+delta_cost2#每个转运站的最小增加成本
                best_insert[pos] = [best_insert1,best_insert2]#每个转运站的最佳插入位置
            minCost = sorted(minCost.items(),key=lambda x:x[1],reverse=False)[0][0]
            self.route2 = copy.deepcopy(best_insert[minCost][0])
            self.route3 = copy.deepcopy(best_insert[minCost][1])

    def greedy_insert_perturbation(self,pool):
        #random.shuffle(pool)
        for c in pool:
            minCost = {}
            best_insert = {}
            for pos in self.open:
                if pos != list(city.transPoi.keys())[-1]:#不是最后一个站
                    start1 = self.route2.index(pos)
                    end1 = self.route2.index(pos+1)#next station
                    start2 = self.route3.index(pos)
                    end2 = self.route3.index(pos+1)
                else:
                    start1 = self.route2.index(pos)
                    end1 = len(self.route2)#next station
                    start2 = self.route3.index(pos)
                    end2 = len(self.route3)
                minCost1 = np.Infinity
                for i in range(start1+1,end1+1):
                    temp = self.route2.copy()
                    if i == end1:
                        temp.insert(i,0)#create a new road
                        temp.insert(i,c)
                    else:
                        temp.insert(i,c)
                    perturbation = 0.4*random.random()+0.8
                    delta_cost1 = (self.cost(temp,1)-self.cost(self.route2,1))*perturbation#考虑全部成本
                    if delta_cost1 < minCost1:
                        best_insert1 = temp
                        minCost1 = delta_cost1
                minCost2 = np.Infinity
                for i in range(start2+1,end2+1):
                    temp = self.route3.copy()
                    if i == end2:
                        temp.insert(i,0)
                        temp.insert(i,c)#create a new road
                    else:
                        temp.insert(i,c)
                    perturbation = 0.4*random.random()+0.8
                    delta_cost2 = (self.cost(temp,2)-self.cost(self.route3,2))*perturbation
                    if delta_cost2 < minCost2:
                        best_insert2 = temp
                        minCost2 = delta_cost2
                minCost[pos] = delta_cost1+delta_cost2#每个转运站的最小增加成本
                best_insert[pos] = [best_insert1,best_insert2]#每个转运站的最佳插入位置
            minCost = sorted(minCost.items(),key=lambda x:x[1],reverse=False)[0][0]
            self.route2 = copy.deepcopy(best_insert[minCost][0])
            self.route3 = copy.deepcopy(best_insert[minCost][1])
    
    def regret_k_insert(self,pool):
        random.shuffle(pool)
        for c in pool:
            minCost = {}
            best_insert = {}
            for pos in self.open:
                if pos != list(city.transPoi.keys())[-1]:#不是最后一个站
                    start1 = self.route2.index(pos)
                    end1 = self.route2.index(pos+1)#next station
                    start2 = self.route3.index(pos)
                    end2 = self.route3.index(pos+1)
                else:
                    start1 = self.route2.index(pos)
                    end1 = len(self.route2)#next station
                    start2 = self.route3.index(pos)
                    end2 = len(self.route3)
                minCost1 = np.Infinity
                delta_cost1 = {}
                insert1 = {}
                for i in range(start1+1,end1+1):
                    temp = self.route2.copy()
                    if i == end1:
                        temp.insert(i,0)#create a new road
                        temp.insert(i,c)
                    else:
                        temp.insert(i,c)
                    delta_cost1[i] = self.cost(temp,1)-self.cost(self.route2,1)#插入每个点使目标值增加的值
                    insert1[i] = temp
                #k = random.randint(0,1)#随机选择最小2个中的一个
                bestPos = sorted(delta_cost1.items(),key=lambda x:x[1],reverse=False)[0][0]
                delta_cost1 = delta_cost1[i]
                Kbest_insert1 = insert1[i]

                delta_cost2 = {}
                minCost2 = np.Infinity
                insert2 = {}
                for i in range(start2+1,end2+1):
                    temp = self.route3.copy()
                    if i == end2:
                        temp.insert(i,0)
                        temp.insert(i,c)#create a new road
                    else:
                        temp.insert(i,c)
                    delta_cost2[i] = self.cost(temp,2)-self.cost(self.route3,2)#插入每个点使目标值增加的值
                    insert2[i] = temp
                #k = random.randint(0,1)#随机选择最小2个中的一个
                bestPos = sorted(delta_cost2.items(),key=lambda x:x[1],reverse=False)[0][0]
                delta_cost2 = delta_cost2[i]
                Kbest_insert2 = insert2[i]
                minCost[pos] = delta_cost1+delta_cost2#每个转运站的最小增加成本
                best_insert[pos] = [Kbest_insert1,Kbest_insert2]#每个转运站的最佳插入位置
            minCost = sorted(minCost.items(),key=lambda x:x[1],reverse=False)
            if minCost[1][1] <= 1.5*minCost[0][1]:#如果第二个增加成本是第一个两倍以下，选择第二好的
                minCost = minCost[1][0]
            else:
                minCost = minCost[0][0]
            self.route2 = best_insert[minCost][0]
            self.route3 = best_insert[minCost][1]
    
    def upper_adjust(self):
        delta_cost = {}
        for t in self.open:
            route = self.route1.copy()
            for i in range(4):
                pos = route.index(t)
                if (route[pos-1] == 0 or route[pos-1] in city.garPoi) and route[pos+1] == 0:#
                    del route[pos+1]
                del route[pos]
            delta_cost[t] = objective([self.route1,self.route2,self.route3])-objective([route,self.route2,self.route3])
        removal = sorted(delta_cost.items(),key=lambda x:x[1],reverse=True)[0][0]#greedy remove
        for i in range(4):
            pos = self.route1.index(removal)
            if (self.route1[pos-1] == 0 or self.route1[pos-1] in city.garPoi) and self.route1[pos+1] == 0:#
                del self.route1[pos+1]#error
            del self.route1[pos]
        for g in city.garPoi:
            delta_cost = {}
            if g != list(city.garPoi)[-1]:#last station
                start = self.route1.index(g)
                end = self.route1.index(g+1)
            else:
                start = self.route1.index(g)
                end = len(self.route1)
            for i in range(start+1,end+1):
                temp = self.route1.copy()
                if i == end:
                    temp.insert(i,0)
                    temp.insert(i,removal)
                else:
                    temp.insert(i,removal)
                delta_cost[i] = objective([temp,self.route2,self.route3])-objective([self.route1,self.route2,self.route3])
            insertion = sorted(delta_cost.items(),key=lambda x:x[1],reverse=False)[0][0]#best insert position
            if insertion == end:
                self.route1.insert(insertion,0)
                self.route1.insert(insertion,removal)
            else:
                self.route1.insert(insertion,removal)
        
    def select_operators(self,weight):
        total_weight = sum(weight.values())
        rand = random.random()*total_weight
        for key in weight.keys():
            if rand <= weight[key]:
                select = key
                break
            else:
                rand = rand-weight[key]
        return select
    
    def main(self):
        p = 0
        #更换转运站后，如果接受新解,p=0，未接受p=30,减少小规模破坏的迭代次数（50回合没有更新,不需继续)
        for t in range(self.max_iteration):#5000
            #print(t,' ',end=' ')
            if p == self.max_unimproved_iteration:#50
                select1 = self.select_operators(self.station_destroy_weight)#self.open发生改变
                open = copy.deepcopy(self.open)
                if select1 == 'station_remove':
                    pool = self.station_remove()
                elif select1 == 'station_swap':
                    pool = self.station_swap()
                elif select1 == 'greedy_station_swap':
                    pool = self.greedy_station_swap()
                else:
                    pool = self.station_open()
                select2 = self.select_operators(self.repair_weight)
                if select2 == 'greedy_insert':
                    self.greedy_insert(pool)
                elif select2 == 'greedy_insert_perturbation':
                    self.greedy_insert_perturbation(pool)
                else:
                    self.regret_k_insert(pool)
                #self.upper_adjust()    #上层优化,exact solution?
                self.upper_optimize()

                if objective([self.route1,self.route2,self.route3]) < objective(self.currentSolu):
                    print(t,' better solution found,transfer station changed,new station',self.open)
                    self.currentSolu = copy.deepcopy([self.route1,self.route2,self.route3])#概率接受该解
                    p = 0#p清零
                    if objective([self.route1,self.route2,self.route3]) <= objective(self.bestSolu):
                        self.station_destroy_weight[select1] = self.station_destroy_weight[select1]+2#比最优解好加2分
                        self.bestSolu = copy.deepcopy([self.route1,self.route2,self.route3])#更新最优解
                        print(t,' best solution updated')
                    else:
                        self.station_destroy_weight[select1] = self.station_destroy_weight[select1]+1#比当前解好加1分 
                else:
                    temp = np.exp((objective(self.currentSolu)-objective([self.route1,self.route2,self.route3]))/10000)
                    #print('accepted possibility:',temp)
                    rand = random.random()
                    if rand <= temp:
                        self.currentSolu = copy.deepcopy([self.route1,self.route2,self.route3])
                        print(t,' possibility is',temp,' and transfer station changed,new station:',self.open)
                        p = 0
                        #temperature up
                        #self.temperature = self.temperature + 3000
                    else:
                        self.route1 = copy.deepcopy(self.currentSolu[0])
                        self.route2 = copy.deepcopy(self.currentSolu[1])
                        self.route3 = copy.deepcopy(self.currentSolu[2])#unaccepted operator 
                        print(t,self.open,' possibility is',temp,' transfer station not changed')
                        self.open = copy.deepcopy(open)
                        p = 30
                '''self.currentSolu = copy.deepcopy([self.route1,self.route2,self.route3])#一定接受该解
                if objective([self.route1,self.route2,self.route3]) <= objective(self.bestSolu):
                    self.station_destroy_weight[select1] = self.station_destroy_weight[select1]+2#比最优解好加2分
                    self.bestSolu = copy.deepcopy([self.route1,self.route2,self.route3])#更新最优解
                print(t,': transfer station changed',self.open)'''
                #p = 0
            else:
                select1 = self.select_operators(self.points_destroy_weight)#destroy operatiors
                if select1 == 'rand_remove':
                    pool = self.rand_remove()
                elif select1 == 'greedy_remove':
                    pool = self.greedy_remove()
                else:
                    pool = self.related_remove()
                self.points_destroy_score[select1][1] = self.points_destroy_score[select1][1]+1#使用过一次
                select2 = self.select_operators(self.repair_weight)#repair operators
                if select2 == 'greedy_insert':
                    self.greedy_insert(pool)
                elif select2 == 'greedy_insert_perturbation':
                    self.greedy_insert_perturbation(pool)
                else:
                    self.regret_k_insert(pool)
                self.repair_score[select2][1] = self.repair_score[select2][1]+1
                #self.upper_adjust()
                self.upper_optimize()
        
                if objective([self.route1,self.route2,self.route3]) < objective(self.currentSolu):
                    #print(t,': better solution found')
                    self.currentSolu = copy.deepcopy([self.route1,self.route2,self.route3])#update current solution
                    p = 0
                    if objective(self.currentSolu) < objective(self.bestSolu):
                        self.bestSolu = copy.deepcopy(self.currentSolu)#update best solution
                        print(t,'best solution updated')
                        self.points_destroy_score[select1][0] = self.points_destroy_score[select1][0]+20#最优解加20分
                        self.repair_score[select2][0] = self.repair_score[select2][0]+20
                    else:
                        self.points_destroy_score[select1][0] = self.points_destroy_score[select1][0]+10#比当前解优加10分
                        self.repair_score[select2][0] = self.repair_score[select2][0]+10        
                else:
                    temp = np.exp((objective(self.currentSolu)-objective([self.route1,self.route2,self.route3]))/self.temperature)
                    #print('accepted possibility:',temp)
                    rand = random.random()
                    if rand <= temp:
                        self.currentSolu = copy.deepcopy([self.route1,self.route2,self.route3])
                        self.points_destroy_score[select1][0] = self.points_destroy_score[select1][0]+2#比当前解差但是被接受加2分
                        self.repair_score[select2][0] = self.repair_score[select2][0]+2
                    else:
                        self.route1 = copy.deepcopy(self.currentSolu[0])
                        self.route2 = copy.deepcopy(self.currentSolu[1])
                        self.route3 = copy.deepcopy(self.currentSolu[2])#unaccepted operator
                    p = p+1#没有改善解迭代次数加1
            if t%self.weight_update_segment == 0 and t>0:#update operators weights
                for key in self.points_destroy_weight:
                    if self.points_destroy_score[key][1] > 0:#used operators
                        self.points_destroy_weight[key] = 0.5*self.points_destroy_weight[key]+ \
                                                    0.5*self.points_destroy_score[key][0]/self.points_destroy_score[key][1]
                        self.points_destroy_score[key] = [0,0]
                for key in self.repair_weight:
                    if self.repair_score[key][1] > 0:
                        self.repair_weight[key] = 0.5*self.repair_weight[key]+ \
                                                0.5*self.repair_score[key][0]/self.repair_score[key][1]
                        self.repair_score[key] = [0,0]
            self.temperature = self.temperature*self.cooling_ratio
            self.process.append(objective(self.bestSolu))
            self.q = math.ceil((0.1*random.random()+0.1)*len(city.colPoi))
            '''if t %100 == 0:
                print(objective(self.bestSolu))'''

city = nodeGeneration() 
city.nodePlot()
#paramater set
e_friction = 0.2
e_speed = 36.67
e_displacement = 6.9
vehicle_train_efficiency = 0.45
disel_e_efficiency = 0.45
curb_weigh = 5000
acceleration = 0
speed = 15 
gravation_const = 9.8 
road_angel = 0 
frontal_suraface = 8 
air_density = 1.2 
aero_drag_coeff = 0.7 
rolling_resistence_coeff = 0.01
const_value = 0.025 
unit_cost = 6.5
oc_trans = 5000
oc_vehicle = 200#车辆成本
truck_cap = 5000#卡车容量
vCap = 3000 
tCap = {}
for t in city.transPoi:#转运站容量
    tCap[t] = 10000

alpha = 0.5*aero_drag_coeff*frontal_suraface*air_density*speed*speed
beta = acceleration+gravation_const*np.sin(road_angel)+ \
    gravation_const*rolling_resistence_coeff*np.cos(road_angel)
gamma = 1/(1000*vehicle_train_efficiency*disel_e_efficiency)
delta = e_friction*e_speed*e_displacement/speed

assign,route = initSolution(city,tCap,vCap)#'road' object,capacity of transfer station, capacity of vehicle            
obj = objective(route)
print(obj)
solution = ALNS(route)
solution.main()
obj = objective(solution.bestSolu)
print('\n final objective is:',obj)
print('bestsolution is:',solution.bestSolu)
print('station_destroy_weight:',solution.station_destroy_weight)
print('repair_weight:',solution.repair_score,solution.repair_weight)
print('points_destroy weight:',solution.points_destroy_score,solution.points_destroy_weight)
plt.plot(solution.process)
plt.show()
