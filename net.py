import matplotlib.pyplot as plt
import numpy as np

class simpleGraph():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.colPoi = {}
        self.transPoi = {}
        self.garPoi = {}
        self.demand = {}
        self.dist1 = {}
        self.dist2 = {}

    def addCollectPoi(self,name,lat,lng,recycle,others,hazardous,kitchen):
        self.colPoi[name] = [lng,lat]
        self.demand[name] = {}
        self.demand[name]['recycle'] = recycle
        self.demand[name]['others'] = others
        self.demand[name]['hazardous'] = hazardous
        self.demand[name]['kitchen'] = kitchen

    def addTransPoi(self,name,lat,lng):#添加转运站
        self.transPoi[name] = [lat,lng]
        
    def addGarPoi(self,name,lat,lng):#添加处理厂
        self.garPoi[name] = [lat,lng]

    def shortestRoute(self):
        echelon1 = self.garPoi.copy()
        echelon1.update(self.transPoi)
        echelon2 = self.colPoi.copy()
        echelon2.update(self.transPoi)
        for i in echelon1.keys():
            for j in echelon1.keys():
                self.dist1[i,j] = 1000*np.sqrt((echelon1[i][0]-echelon1[j][0])**2+ \
                                  (echelon1[i][1]-echelon1[j][1])**2)
        for i in echelon2.keys():
            for j in echelon2.keys():
                self.dist2[i,j] = 1000*np.sqrt((echelon2[i][0]-echelon2[j][0])**2+ \
                                    (echelon2[i][1]-echelon2[j][1])**2)

    def nodePlot(self):
        for i in self.colPoi:
            self.ax.scatter(self.colPoi[i][1],self.colPoi[i][0],c='red')
            self.ax.annotate(i,xy=(self.colPoi[i][1],self.colPoi[i][0]),
                    xytext=(self.colPoi[i][1]+0.1,self.colPoi[i][0]+0.1))
        for i in self.transPoi:
            self.ax.scatter(self.transPoi[i][1],self.transPoi[i][0],c='blue')
            self.ax.annotate(i,xy=(self.transPoi[i][1],self.transPoi[i][0]),
                    xytext=(self.transPoi[i][1]+0.1,self.transPoi[i][0]+0.1))
        for i in self.garPoi:
            self.ax.scatter(self.garPoi[i][1],self.garPoi[i][0],c='g')
            self.ax.annotate(i,xy=(self.garPoi[i][1],self.garPoi[i][0]),
                    xytext=(self.garPoi[i][1]+0.1,self.garPoi[i][0]+0.1))
        plt.show()


if __name__ == '__main__':
    city = simpleGraph()

    city.addCollectPoi(1,2.51,2.22,429,279,357,368)
    city.addCollectPoi(2,1.49,4.46,301,227,270,526)
    city.addCollectPoi(3,1.64,3.34,344,270,357,274)
    city.addCollectPoi(4,1.52,.51,318,229,207,531)
    city.addCollectPoi(5,1.25,4.0,298,271,215,497)
    city.addCollectPoi(6,1.76,4.99,317,571,403,555)
    city.addCollectPoi(7,3.6,1.97,269,571,533,418)
    city.addCollectPoi(8,2.88,0.71,286,462,570,345)
    city.addTransPoi(101,2.4,0.31)
    city.addTransPoi(102,4.15,1.57)
    city.addTransPoi(103,0.35,3.65)
    city.addGarPoi(1,1.47,1.75)
    city.addGarPoi(2,4.78,4.02)
    city.addGarPoi(3,0.71,1.96)
    city.addGarPoi(4,2.05,0.33)
    city.shortestRoute()
    city.nodePlot()