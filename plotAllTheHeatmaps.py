import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sklearn.linear_model import LinearRegression
%matplotlib inline

def get_temperature(lat, lon):
    nc_obj = Dataset('HadISST_sst.nc')
    '''
    print(nc_obj)
    print(nc_obj.variables.keys())
    for i in nc_obj.variables.keys():
        print('___________________________________________')
        print(i)
        print(nc_obj.variables[i])
    '''
    latitude = int(89.5 - lat)
    longitude = int(179.5 + lon)
    # print(nc_obj.variables['latitude'][latitude])
    # print(nc_obj.variables['longitude'][longitude])
    temperature = []  # 储存所有的温度
    for t in np.arange(1800):
        temperature.append(nc_obj.variables['sst'][t][latitude][longitude])
    return temperature

list_i=[]
list_j=[]
iii=62
jjj=-10
while iii>48:
    list_i.append(iii)
    iii-=0.4
while jjj<4.1:
    list_j.append(jjj)
    jjj+=0.4

print(len(list_i))
print(len(list_j))
kk=np.zeros((len(list_i),len(list_j)))
print(kk.shape)
for i in range(len(list_i)):
    for j in range(len(list_j)):
        lis = get_temperature(list_i[i],list_j[j])
        y=[]
        for k in range(1800):
            if k%12==6 :
                y.append(lis[k])
        if not y[0]==y[0]:
            print("LAND")
            continue
        result=[]
        for ip in range(30):
            t = (y[ip*5]+y[ip*5+1]+y[ip*5+2]+y[ip*5+3]+y[ip*5+4])/5.0
            result.append(t)
        x = np.arange(1870,2020,5)
        x = np.array(x).reshape(-1,1)
        result=np.array(result).reshape(-1,1)
        model = LinearRegression(fit_intercept=True, normalize=False) 
        model.fit(x, result)
        value=model.predict([[2070]]).reshape(-1)
        pred=value[0]
        kk[i][j]=pred
        print(pred)

import seaborn as sns
import pandas as pd

fig, ax = plt.subplots(figsize = (5,5))
List_i = ['%.1f'%oi for oi in list_i]
List_j = ['%.1f'%oj for oj in list_j]
sns.heatmap(pd.DataFrame(kk, columns=List_j, index =List_i), vmin=9, vmax=18, xticklabels= True, yticklabels= True, square=True, cmap="Spectral_r")
ax.set_title('SST in 2019', fontsize = 9)
ax.set_ylabel('latitude', fontsize = 9)
ax.set_xlabel('longitude', fontsize = 9) 
plt.xticks(rotation=68,fontsize=6)
plt.yticks(fontsize=6)

plt.savefig('2019.eps', bbox_inches='tight')

list_i=[]
list_j=[]
iii=62
jjj=-10
while iii>48:
    list_i.append(iii)
    iii-=0.4
while jjj<4.1:
    list_j.append(jjj)
    jjj+=0.4

cur = np.zeros((len(list_i),len(list_j)))
print(cur.shape)
for i in range(len(list_i)):
    for j in range(len(list_j)):
        lis = get_temperature(list_i[i],list_j[j])
        y=[]
        for k in range(1800):
            if k%12==6 :
                y.append(lis[k])
        if not y[0]==y[0]:
            print("LAND")
            continue
        result=[]
        for ip in range(30):
            t = (y[ip*5]+y[ip*5+1]+y[ip*5+2]+y[ip*5+3]+y[ip*5+4])/5.0
            result.append(t)
        pred=result[len(result)-1]
        cur[i][j]=pred
        print(pred)

fig, ax = plt.subplots(figsize = (5,5))
Listc_i = ['%.1f'%oi for oi in list_i]
Listc_j = ['%.1f'%oj for oj in list_j]
sns.heatmap(pd.DataFrame(cur, columns=Listc_j, index =Listc_i), vmin=9, vmax=18, xticklabels= True, yticklabels= True, square=True, cmap="Spectral_r")
ax.set_title('SST by linear regression in 2070', fontsize = 9)
ax.set_ylabel('latitude', fontsize = 9)
ax.set_xlabel('longitude', fontsize = 9) 
plt.xticks(rotation=68,fontsize=6)
plt.yticks(fontsize=6)
plt.savefig('l2070.eps', bbox_inches='tight')

fig, ax = plt.subplots(figsize = (5,5))
delta = kk-cur
cop = delta
cop1 = np.where(cop==0,-0.9,cop)
sns.heatmap(pd.DataFrame(cop1, columns=Listc_j, index =Listc_i), vmax = 2.5 ,vmin= - 1.0, xticklabels= True, yticklabels= True, square=True, cmap="Spectral_r")
ax.set_title('variation of SST by linear regression', fontsize = 9)
ax.set_ylabel('latitude', fontsize = 9)
ax.set_xlabel('longitude', fontsize = 9) 
plt.xticks(rotation=68,fontsize=6)
plt.yticks(fontsize=6)
plt.savefig('vl2070.eps', bbox_inches='tight')

list_i=[]
list_j=[]
iii=62
jjj=-10
while iii>48:
    list_i.append(iii)
    iii-=0.4
while jjj<4.1:
    list_j.append(jjj)
    jjj+=0.4
print(len(list_i))
print(len(list_j))
from sklearn.preprocessing import PolynomialFeatures

qua=np.zeros((len(list_i),len(list_j)))
print(qua.shape)
for i in range(len(list_i)):
    for j in range(len(list_j)):
        lis = get_temperature(list_i[i],list_j[j])
        y=[]
        for k in range(1800):
            if k%12==6 :
                y.append(lis[k])
        if not y[0]==y[0]:
            print("LAND")
            continue
        result=[]
        for ip in range(30):
            t = (y[ip*5]+y[ip*5+1]+y[ip*5+2]+y[ip*5+3]+y[ip*5+4])/5.0
            result.append(t)
        x = np.arange(1870,2020,5)
        x = np.array(x).reshape(-1,1)
        result=np.array(result).reshape(-1,1)
        deg2 = PolynomialFeatures(degree = 2)  ## degree = 2
        model2 = LinearRegression()
        model2.fit(deg2.fit_transform(x),result)
        value=model2.predict(deg2.fit_transform([[2070]])).reshape(-1)
        pred=value[0]
        qua[i][j]=pred
        print(pred)

fig, ax = plt.subplots(figsize = (5,5))
Listq_i = ['%.1f'%oi for oi in list_i]
Listq_j = ['%.1f'%oj for oj in list_j]
sns.heatmap(pd.DataFrame(qua, columns=Listq_j, index =Listq_i), vmin=9, vmax=18, xticklabels= True, yticklabels= True, square=True, cmap="Spectral_r")
ax.set_title('SST by quadratic regression in 2070', fontsize = 9)
ax.set_ylabel('latitude', fontsize = 9)
ax.set_xlabel('longitude', fontsize = 9) 
plt.xticks(rotation=68,fontsize=6)
plt.yticks(fontsize=6)
plt.savefig('q2070.eps', bbox_inches='tight')

%config InlineBackend.figure_format = 'svg'
fig, ax = plt.subplots(figsize = (5,5))
deltaq = qua-cur
ccop = deltaq
ccop1 = np.where(ccop==0,-1,ccop)
sns.heatmap(pd.DataFrame(ccop1, columns=Listq_j, index =Listq_i),vmax=2.5, vmin=-1, xticklabels= True, yticklabels= True, square=True, cmap="Spectral_r")
ax.set_title('Variation of SST by quadratic regression', fontsize = 9)
ax.set_ylabel('latitude', fontsize = 9)
ax.set_xlabel('longitude', fontsize = 9) 
plt.xticks(rotation=68,fontsize=6)
plt.yticks(fontsize=6)
plt.savefig('vq2070.eps', bbox_inches='tight')

list_i=[]
list_j=[]
iii=62
jjj=-10
while iii>48:
    list_i.append(iii)
    iii-=0.4
while jjj<4.1:
    list_j.append(jjj)
    jjj+=0.4
print(len(list_i))
print(len(list_j))
from sklearn.preprocessing import PolynomialFeatures

mid=np.zeros((len(list_i),len(list_j)))
print(mid.shape)
for i in range(len(list_i)):
    for j in range(len(list_j)):
        lis = get_temperature(list_i[i],list_j[j])
        y=[]
        for k in range(1800):
            if k%12==0 :
                y.append(lis[k])
        if not y[0]==y[0]:
            print("LAND")
            continue
        result=[]
        for ip in range(30):
            t = (y[ip*5]+y[ip*5+1]+y[ip*5+2]+y[ip*5+3]+y[ip*5+4])/5.0
            result.append(t)
        x = np.arange(1870,2020,5)
        x = np.array(x).reshape(-1,1)
        result=np.array(result).reshape(-1,1)
        deg21 = PolynomialFeatures(degree = 2)  ## degree = 2
        model21 = LinearRegression()
        model21.fit(deg21.fit_transform(x),result)
        value=model21.predict(deg21.fit_transform([[2070]])).reshape(-1)
        pred=value[0]
        mid[i][j]=pred
        print(pred)