#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:13:40 2020

@author: edwinsolis
"""
## Examen Mineria de Datos
## EDWIN SOLIS
## 28/06/2020

import os
import pandas as pd
import scipy.stats as s
import seaborn as sns
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


# Dataset: check if files exists in directory
def dataset():
    p = os.path.isfile('Wind_Data.xlsx')
    if p == True:
        #Import dataset from excel file
        df = pd.read_excel('Wind_Data.xlsx','Wind') 
    else:
        #Use dataset from repo GitHub
        print('No existen los datos del viento en el directorio')
        print('Importación de dataset desde GitHub repo')
        url = 'https://raw.githubusercontent.com/esoliss/dataminingMERR/master/Datasets/wind.csv'
        df = pd.read_csv(url, error_bad_lines=False)
        df = df.drop(columns=['Unnamed: 0'])
    q = os.path.isfile('Curva_Aerogenerador_1MW.xlsx')
    if q == True:
        #Import dataset from excel file
        dg = pd.read_excel('Curva_Aerogenerador_1MW.xlsx','wind') 
    else:
        #Use dataset from repo GitHub
        print('\nNo existen los datos del aerogenerador en el directorio')
        print('Importación de dataset desde GitHub repo')
        urla = 'https://raw.githubusercontent.com/esoliss/dataminingMERR/master/Datasets/aero.csv'
        dg = pd.read_csv(urla, error_bad_lines=False)
        dg = dg.drop(columns=['Unnamed: 0'])
    return df,dg
# Dataset wind and aerogenerator
wind,aero = dataset()
# Weibull distribution for wind from dataset
def weib(wind):
    data = wind.stack().droplevel(level=0)
    params = s.exponweib.fit(data, floc=0, f0=1)
    r = s.exponweib.rvs(*params,size=1)[0]
    return r 

# Wind turbine power output from curve
def powerW(r,aero):
    rbf = interp1d(aero['Speed (m/s)'], aero['Power (kW)'])
    fi = rbf(r)
    dat = pd.DataFrame(data=np.column_stack((r,fi)),columns=['x','y'])
    p = dat.iloc[0]['y']
    return (p/1000)*15

## Load Forecasting
def load(T):
    mu = 0.05
    sigma = 0.03
    S0 = 10
    dt = 0.01
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt)
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X)
    lo = S[-1]
    return lo

## Conventional generation
def powerC(mu):
    std = 0.04
    ge = np.random.normal(mu, std, 1)[0]
    return ge

#Monte carlo
def mc(year,geco):
    ww = weib(wind)
    pww =powerW(ww,aero)
    ld = load(year)
    pwc=powerC(geco)
    return ww,pww,ld,pwc
def montecarlo(year_study,geco,itera):
    results_mc = np.zeros((itera,4))
    #results_mc[:,0]=int(1)#year_study
    n =0
    for i in results_mc[:,0]:
        results_mc[n,0],results_mc[n,1],results_mc[n,2],results_mc[n,3]=mc(year_study,geco)
        n+=1
    return results_mc
itera = 100
#results = montecarlo(int(10),22,itera)
case = []
study = [(1,15),(5,16),(10,22)]
for it in study:
    results = montecarlo(int(it[0]),it[1],itera)
    case.append(results)

### RESULTS: Plots

def Pltgen(cas,year):
    if year[0]==1:
        results = cas[0]
    if year[0]==5:
        results = cas[1]
    if year[0]==10:
        results = cas[2]
#plt.figure(1)
    plt.axvline(x=np.mean(results[:,3])-2*np.std(results[:,3]),label = '2\u03c3 = '+
                str(round(np.mean(results[:,3])-2*np.std(results[:,3]),2))+'%',
                color = 'r', linestyle = '--')
    plt.legend()
#plt.xlim(11,15)
    plt.ylim(0,0.5)
    sns.distplot(results[:,1],color='black', label ='Potencia granja')
    sns.distplot(results[:,2],color='blue', label ='Carga')
    sns.distplot(results[:,2]-results[:,1],color='orange', label ='Potencia neta')
    sns.distplot(results[:,3],color='green', label ='Potencia firme '+str(year[1])+'MW')
    plt.xlabel('Potencia (MW)')
    plt.ylabel('Densidad de probabilidad')
    plt.suptitle('Resultados año '+str(year[0]))
    plt.legend()
    plt.show()
    return

def Zoomplt(cas,year):
    if year[0]==1:
        results = cas[0]
        plt.xlim(9,16)
    if year[0]==5:
        results = cas[1]
        plt.xlim(11,16)
    if year[0]==10:
        results = cas[2]
        plt.xlim(15,22.5)
#plt.figure(2)
    plt.axvline(x=np.mean(results[:,3])-2*np.std(results[:,3]),label = '2\u03c3 = '+
                str(round(np.mean(results[:,3])-2*np.std(results[:,3]),2))+'%',
                color = 'r', linestyle = '--')
    plt.ylim(0,0.5)
    sns.distplot(results[:,2],color='blue', label ='Carga')
    sns.distplot(results[:,2]-results[:,1],color='orange', label ='Potencia neta')
    sns.distplot(results[:,3],color='green', label ='Potencia firme'+str(year[1])+'MW')
    plt.legend()
    plt.suptitle('Zoom Resultados año '+str(year[0]))
    plt.show()
    return

def Gresults(wind,aero):
# Weibull distribution for wind from dataset
    data = wind.stack().droplevel(level=0)
    params = s.exponweib.fit(data, floc=0, f0=1)
    shape = params[1]
    scale = params[3]
    print ('Parametros Weibull estimados')
    print ('shape:',shape)
    print ('scale:',scale)
    plt.figure(10)
#### Plotting
## Stochastic wind
    values,bins,hist = plt.hist(data,bins=20,range=(0,25),density=True,alpha=0.2,label='Datos del viento')
    center = np.arange(0, 25, 0.005)
    plt.plot(center,s.exponweib.pdf(center,*params),lw=1,label='Weibull PDF',color='r')
    plt.hist(s.exponweib.rvs(*params,size=100),density=True,histtype='step',color='g',label='Viento estocástico')
    plt.legend()
    plt.xlabel('Velocidad del viento (m/s)')
    plt.ylabel('Densidad')
    plt.suptitle('Modelo estocástico de la velocidad del viento')
    plt.show
## Wind Turbine power output
    r = s.exponweib.rvs(*params,size=10)
    plt.figure(20)
    plt.plot(aero['Speed (m/s)'], aero['Power (kW)'],label='Curva potencia vs viento')
    rbf = interp1d(aero['Speed (m/s)'], aero['Power (kW)'])
    fi = rbf(r)
    dat = pd.DataFrame(data=np.column_stack((r,fi)),columns=['x','y'])
    plt.scatter(dat['x'], dat['y'], c='r',label='Valor calculado (Weibull)')
    plt.xlabel('Velocidad del viento (m/s)')
    plt.ylabel('Potencia (kW)')
    plt.suptitle('Cálculo de la potencia de un aerogenerador')
    plt.legend()
    plt.show()
## Load Forecasting
    T = 15
    mu = 0.05
    sigma = 0.03
    S0 = 10
    dt = 0.01
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt)
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X)
    plt.axvline(x=1,color='red',linestyle = '--',label='Año 1')
    plt.axvline(x=5,color='blue',linestyle = '--',label='Año 5')
    plt.axvline(x=10,color='green',linestyle = '--',label='Año 10')
    plt.plot(t, S)
    plt.xlabel('Año')
    plt.ylabel('Demanda (MW)')
    plt.suptitle('Crecimiento de la demanda GBM')
    plt.legend()
    plt.show()
    return
Gresults(wind,aero)
for it in study:
    Pltgen(case,it)
    Zoomplt(case,it)
print('###################RESULTADOS###################')
print('\nAÑO 1*******************************************')
print('Se asume que existe generación firme de 15MW')
print('No se requiere instalar generación firme extra')
print('\nAÑO 5*******************************************')
print('Se asume para comparar generación firme de 16MW')
print('Se requiere incrementar generación convencional')
print('\nAÑO 10******************************************')
print('Se asume para comparar generación firme de 22MW')
print('Se requiere incrementar generación convencional')
    