#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:39:21 2020

@author: edwinsolis
"""


import pandas as pd
import os

## Importación de datos desde dataset excel
wind = pd.read_excel('Wind_Data.xlsx','Wind')
aero = pd.read_excel('Curva_Aerogenerador_1MW.xlsx','wind')
# Dataset: check if file exists in directory
def dataset():
    p = os.path.isfile('Bisagra.xlsx')
    if p == True:
        #Import dataset from Bisagra.xls
        df = pd.read_excel('Bisagra.xlsx','Dimensiones') 
    else:
        #Use dataset from repo GitHub
        print('No existen los datos del viento en el directorio')
        print('Importación de dataset desde GitHub repo')
        url = 'https://raw.githubusercontent.com/esoliss/dataminingMERR/master/Datasets/wind.csv'
        df = pd.read_csv(url, error_bad_lines=False)
        df = df.drop(columns=['Unnamed: 0'])
    q = os.path.isfile('Bisagra2.xlsx')
    if q == True:
        #Import dataset from Bisagra.xls
        dg = pd.read_excel('Bisagra2.xlsx','Dimensiones') 
    else:
        #Use dataset from repo GitHub
        print('\nNo existen los datos del aerogenerador en el directorio')
        print('Importación de dataset desde GitHub repo')
        urla = 'https://raw.githubusercontent.com/esoliss/dataminingMERR/master/Datasets/aero.csv'
        dg = pd.read_csv(urla, error_bad_lines=False)
        dg = dg.drop(columns=['Unnamed: 0'])
    return df,dg

r,m = dataset()
wind.to_csv('wind.csv')
aero.to_csv('aero.csv')
