from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression


class loo():
      def __init__(self,X,y,dX,comp):
          self.X=X
          self.y=y
          self.dX=dX
          self.comp=comp
          
          
      def cal(self):
          loo = LeaveOneOut()
          X2=np.array(self.X)
          y=np.array(self.y)
          l=[]
          for train_index, test_index in loo.split(X2):
              X_train, X_test = X2[train_index], X2[test_index]
              y_train, y_test = y[train_index], y[test_index]
              X_train=pd.DataFrame(X_train)
              X_test=pd.DataFrame(X_test) 
              reg=PLSRegression(n_components=self.comp,max_iter=10000)
              reg.fit(X_train,y_train)
              l.append(reg.predict(X_test)[0])
          l=pd.DataFrame(l)
          l.columns=['Pred_loo']
          dataf=pd.concat([self.dX,l], axis=1)
          dataf['Del_Res']=dataf[self.y.columns[0]]-dataf['Pred_loo']
          dataf['aver']=dataf[self.y.columns[0]].mean()
          aver=dataf[self.y.columns[0]].mean()
          dataf['nsum']=dataf[self.y.columns[0]]-dataf['aver']
          Q2Loo=1-((dataf['Del_Res']**2).sum()/(dataf['nsum']**2).sum())
          return Q2Loo,aver,l