import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import math


class rm2():
      def __init__(self,y,x):
          self.y=y
          self.x=x
      
      def rm2ns(self,df):
          scaler=MinMaxScaler()
          y=df.iloc[:,0:1]
          x=df.iloc[:,1:2]
          try:   ##https://stackoverflow.com/questions/62561902/valueerror-illegal-value-in-4-th-argument-of-internal-none-when-running-sklearn
             mlr1=LinearRegression(fit_intercept=True,normalize=True).fit(x,y)   ##This modification is made to avoid error appers in repeated operations
          except:
             mlr1=LinearRegression(fit_intercept=True,normalize=True).fit(x,y)
          try:
             mlr2=LinearRegression(fit_intercept=False,normalize=True).fit(x,y)
          except:
             mlr2=LinearRegression(fit_intercept=False,normalize=True).fit(x,y)

          #xsc=scaler.fit_transform(np.array(x).reshape(-1,1))
          #ysc=scaler.fit_transform(np.array(y).reshape(-1,1))
          #mlr1.fit(x,y)
          r2=mlr1.score(x,y)
          #mlr2.fit(x,y)
          r20=mlr2.score(x,y)
          mlr2.fit(y,x)
          r20d=mlr2.score(y,x)
          rm2=(1-math.sqrt(r2-r20))*r2
          rm2d=(1-math.sqrt(r2-r20d))*r2
          rm2f=(rm2+rm2d)/2
          drm2=abs(rm2-rm2d)
          return rm2f,rm2,rm2d,drm2
      
      def fit(self):
          df=pd.concat([self.y,self.x],axis=1)
          df['Min']=df[self.y.columns[0]].min()
          df['range']=df[self.y.columns[0]].max()-df[self.y.columns[0]].min()
          df['scaley']=(df[self.y.columns[0]]-df['Min'])/(df['range'])
          df['scalex']=(df[self.x.columns[0]]-df['Min'])/(df['range'])
          dfn=df.iloc[:,-2:]
          rm2f,rm2,rm2d,drm2=self.rm2ns(dfn)
          return rm2f,drm2

          
