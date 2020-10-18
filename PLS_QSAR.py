import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import os
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from loo_pls import loo
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rm2 import rm2
from applicability import apdom
from rm2 import rm2
import math
import numpy as np
from matplotlib import pyplot



form = tk.Tk()
form.title("PLS-QSAR")
form.geometry("650x350")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text="Data preparation")

initialdir=os.getcwd()

def datatr():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
   
    
def datats():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
    
def calstep1(X,y,tr,mc):
    l1,l2,l3=[],[0.001],[]
    for i in range(1,mc):
        pls = PLSRegression(n_components=i,max_iter=10000)
        pls.fit(X, y)
        cv=loo(X,y,tr,i)
        c,m,l=cv.cal()
        #l1.append(r2)
        l2.append(c)
        val=(l2[len(l2)-1]-l2[len(l2)-2])
        val2=val/l2[len(l2)-2]*100   
        l3.append(val2)
        if val2<int(forthEntryTabThree.get()):
           break
        print(val2)
    return i-1

def calstep2(X,y,tr,mc):
    l1,lt,l3=[],[0.001],[]
    for i in range(1,mc):
        pls = PLSRegression(n_components=i,max_iter=10000)
        pls.fit(X, y)
        cv=loo(X,y,tr,i)
        c,m,l=cv.cal()
        c1=mean_absolute_error(y,l)
        lt.append(c1)
        val=abs(lt[len(lt)-1]-lt[len(lt)-2])
        val2=val/lt[len(lt)-2]*100   
        if val2<int(forthEntryTabThree.get()):
           break
        print(val2)
    return i-1
    
    

def writefile1():
    Xtr=file1.iloc[:,1:-1]
    ytr=file1.iloc[:,-1:]
    ntr=file1.iloc[:,0:1]
    tr=file1.iloc[:,1:]
    
    #Xts=file2.iloc[:,1:-1]
    #yts=file2.iloc[:,-1:]
    nts=file2.iloc[:,0:1]
    
    Xts=file2[Xtr.columns]
    yts=file2[ytr.columns]
    
    
    dct=str(OFNEntry.get())
    if not os.path.isdir(dct):
       os.mkdir(dct)
    filex = 'Results.txt'
    file_path = os.path.join(dct, filex)
    filer = open(file_path, "w")    
    ms=int(thirdEntryTabThree.get())
    
    if Criterion.get()=='cv':
       a1=calstep1(Xtr,ytr,tr,ms)
    elif Criterion.get()=='mae':
       a1=calstep2(Xtr,ytr,tr,ms)
       
    
    pls2 = PLSRegression(n_components=a1,max_iter=10000)
    pls2.fit(Xtr,ytr)
    r2=pls2.score(Xtr,ytr)
    cv=loo(Xtr,ytr,tr,a1)
    c,m,l=cv.cal()
    ypr=pd.DataFrame(pls2.predict(Xtr))
    ypr.columns=['Predicted']
    
    rm2tr,drm2tr=rm2(ytr,l).fit()
    l=pd.DataFrame(l)
    l.columns=['Predicted_LOO']
    d=mean_absolute_error(ytr,ypr)
    e=(mean_squared_error(ytr,ypr))**0.5
    adstr=apdom(Xtr,Xtr)
    yadstr=adstr.fit() 
    df=pd.concat([ntr,Xtr,ytr,ypr,l,yadstr],axis=1)
    name1="Train_prediction.csv"
    df.to_csv(os.path.join(dct,name1),index=False)
    
    
        
    filer.write("Sub-training set results "+"\n")
    filer.write("\n")
    filer.write('Number of component selected: '+str(a1)+"\n")
    filer.write("Descriptors :"+str(Xtr.columns.tolist())+"\n")
    filer.write("Coefficients :")
    for i in (pls2.coef_):
        filer.write(str(round(i[0],3))+',')
    filer.write("\n")
    filer.write("R2(Train):"+str(r2)+"\n")
    filer.write("Q2LOO(Train):"+str(c)+"\n")
    filer.write('MAE(Train): '+str(d)+"\n")
    filer.write('RMSE(Train): '+str(e)+"\n")
    filer.write('rm2LOO: '+str(rm2tr)+"\n")
    filer.write('delta rm2LOO: '+str(drm2tr)+"\n")
    filer.write("\n")
    
    if ytr.columns[0] in file2.columns:
       ytspr=pd.DataFrame(pls2.predict(Xts))
       ytspr.columns=['Predicted']
       rm2ts,drm2ts=rm2(yts,ytspr).fit()
       tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
       maets=mean_absolute_error(yts,ytspr)
       tsdf.columns=['Active','Predict']
       tsdf['Aver']=m
       tsdf['Aver2']=tsdf['Predict'].mean()
       tsdf['diff']=tsdf['Active']-tsdf['Predict']
       tsdf['diff2']=tsdf['Active']-tsdf['Aver']
       tsdf['diff3']=tsdf['Active']-tsdf['Aver2']
       r2pr=1-((tsdf['diff']**2).sum()/(tsdf['diff2']**2).sum())
       r2pr2=1-((tsdf['diff']**2).sum()/(tsdf['diff3']**2).sum())
       RMSEP=((tsdf['diff']**2).sum()/tsdf.shape[0])**0.5
       adts=apdom(Xts,Xtr)
       yadts=adts.fit()
       dfts=pd.concat([nts,Xts,yts,ytspr,yadts],axis=1)
       name2="Test_prediction.csv"
       dfts.to_csv(os.path.join(dct,name2),index=False)
       #dfts.to_csv(str(c_)+"_sfslda_tspr.csv",index=False)
       filer.write('Test set results: '+"\n")
       filer.write('Number of observations: '+str(yts.shape[0])+"\n")
       filer.write('Q2F1/R2Pred: '+ str(r2pr)+"\n")
       filer.write('Q2F2: '+ str(r2pr2)+"\n")
       filer.write('rm2test: '+str(rm2ts)+"\n")
       filer.write('delta rm2test: '+str(drm2ts)+"\n")
       filer.write('RMSEP: '+str(RMSEP)+"\n")
       filer.write('MAE(Test): '+str(maets)+"\n")
       filer.write("\n")
       plt1=pyplot.figure(figsize=(15,10))
       pyplot.scatter(ytr,ypr, label='Train', color='blue')
       pyplot.plot([ytr.min(), ytr.max()], [ytr.min(), ytr.max()], 'k--', lw=4)
       pyplot.scatter(yts,ytspr, label='Test', color='red')
       pyplot.ylabel('Predicted values',fontsize=28)
       pyplot.xlabel('Observed values',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       #rocn='obs_vspred.png'
       name3="obs_vspred.png"
       plt1.savefig(os.path.join(dct,name3),dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
       #plt1.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      #format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
       plt2=pyplot.figure(figsize=(15,10))
       pyplot.scatter(ytr,l, label='Train(LOO)', color='blue')
       pyplot.plot([ytr.min(), ytr.max()], [ytr.min(), ytr.max()], 'k--', lw=4)
       pyplot.scatter(yts,ytspr, label='Test', color='red')
       pyplot.ylabel('Predicted values',fontsize=28)
       pyplot.xlabel('Observed values',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       name4="obsloo_vspred.png"
       plt2.savefig(os.path.join(dct,name4),dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
    else:
        Xts=file2.iloc[:,1:]
        nts=file2.iloc[:,0:1]
        ytspr=pd.DataFrame(reg.predict(Xts))
        ytspr.columns=['Pred']
        adts=apdom(Xts,Xtr)
        yadts=adts.fit()
        dfts=pd.concat([nts,Xts,ytspr,yadts],axis=1)
        name2="Test_prediction.csv"
        dfts.to_csv(os.path.join(dct,name2),index=False)
        #dfts.to_csv(str(c_)+"_sfslda_scpr.csv",index=False)
    filer.close()
    pyplot.close() 
    

   
firstLabelTabThree = tk.Label(tab1, text="Select training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=95,y=10)
firstEntryTabThree = tk.Entry(tab1, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab1,text='Browse', command=datatr,font=("Helvetica", 10))
b3.place(x=480,y=10)

secondLabelTabThree = tk.Label(tab1, text="Select test/screening set",font=("Helvetica", 12))
secondLabelTabThree.place(x=45,y=40)
secondEntryTabThree = tk.Entry(tab1,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab1,text='Browse', command=datats,font=("Helvetica", 10))
b4.place(x=480,y=40)

OFN=Label(tab1, text='Type output folder name',font=("Helvetica", 12))
OFN.place(x=125,y=75)
OFNEntry=Entry(tab1)
OFNEntry.place(x=300,y=78)

thirdLabelTabThree=Label(tab1, text='Maximum number of components',font=("Helvetica", 12))
thirdLabelTabThree.place(x=60,y=105)
thirdEntryTabThree=Entry(tab1)
thirdEntryTabThree.place(x=300,y=108)

Criterion_Label = ttk.Label(tab1, text="Condition:",font=("Helvetica", 12))
Criterion = StringVar()
Criterion.set(False)
Criterion_Gini = ttk.Radiobutton(tab1, text='CVLOO', variable=Criterion, value="cv")
Criterion_Entropy = ttk.Radiobutton(tab1, text='MAELOO', variable=Criterion, value="mae")
Criterion_Label.place(x=210,y=135)
Criterion_Gini.place(x=300,y=135)
Criterion_Entropy.place(x=370,y=135)

forthLabelTabThree=Label(tab1, text='incremenet',font=("Helvetica", 12))
forthLabelTabThree.place(x=170,y=165)
forthEntryTabThree=Entry(tab1)
forthEntryTabThree.place(x=300,y=168)

b2=Button(tab1, text='Generate model', command=writefile1,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=310,y=200)


tab_parent.pack(expand=1, fill='both')

form.mainloop()