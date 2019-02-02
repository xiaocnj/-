#导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn import metrics
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from pylab import *
from matplotlib import font_manager
import scipy.stats as stats
from sklearn.externals import joblib
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

##读取数据
data1=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\2013-2018总表.xlsx',index_col=0,)
data1=data1.dropna()
#AQI5年来每日的折线图 生成AQI与其他污染物的回归方程
AQIday=data1.loc[:,'AQI']
AQIday.plot(title='2013/12-2018/11 每日AQI指数 ',LineWidth=2,marker='o',grid=True,use_index=True)
plt.xlabel('日期',fontsize=12)
plt.ylabel('AQI',fontsize=12)
plt.show()
X1=data1.iloc[:,2:8].astype(float)
y1=data1.iloc[:,0].astype(float)
X1_train,X1_test,y1_train,y1_test=model_selection.train_test_split(X1,y1,test_size=0.3,random_state=1)
linregTr1=LinearRegression()
linregTr1.fit(X1_train,y1_train)
print(linregTr1.intercept_,linregTr1.coef_)
y1_train_pred=linregTr1.predict(X1_train)
y1_test_pred=linregTr1.predict(X1_test)
train_err1=metrics.mean_squared_error(y1_train,y1_train_pred)
test_err1=metrics.mean_squared_error(y1_test,y1_test_pred)
print('the mean squar error of train and test are:{:.2f},{:.2f}'.format(train_err1,test_err1))
predict_score1=linregTr1.score(X1_test,y1_test)
print('The decision coeficient is:{:.2f}'.format(predict_score1))

#饼图
piedata=data1.loc[:,'质量等级']
datasum=piedata.value_counts()
datasum.plot(kind='pie',figsize=(6,6),title='五年来空气质量等级分布',fontsize=14,autopct='%1.1f%%')
plt.show()
piedata2014=data1.iloc[32:396,1]
datasum2014=piedata2014.value_counts()
datasum2014.plot(kind='pie',figsize=(6,6),title='2014空气质量等级分布',fontsize=14,autopct='%1.1f%%')
plt.show()
piedata2015=data1.iloc[396:761,1]
datasum2015=piedata2015.value_counts()
datasum2015.plot(kind='pie',figsize=(6,6),title='2015空气质量等级分布',fontsize=14,autopct='%1.1f%%')
plt.show()
piedata2016=data1.iloc[762:1127,1]
datasum2016=piedata2016.value_counts()
datasum2016.plot(kind='pie',figsize=(6,6),title='2016空气质量等级分布',fontsize=14,autopct='%1.1f%%')
plt.show()
piedata2017=data1.iloc[1128:1492,1]
datasum2017=piedata2017.value_counts()
datasum2017.plot(kind='pie',figsize=(6,6),title='2017空气质量等级分布',fontsize=14,autopct='%1.1f%%')
plt.show()
piedata2018=data1.iloc[1493:1826,1]
datasum2018=piedata2018.value_counts()
datasum2018.plot(kind='pie',figsize=(6,6),title='2018空气质量等级分布',fontsize=14,autopct='%1.1f%%')
plt.show()

#AQI 每月
data2=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\2013-2018上海空气质量指数月统计历史数据.xlsx',index_col=0,skiprows=1)
AQImonth=data2.loc[:,'AQI']
AQImonth.plot(title='2013/12-2018/11 每月AQI指数曲线 ',LineWidth=2,marker='o',linestyle='dashed',grid=True,use_index=True)
plt.xlabel('Year',fontsize=12)
plt.ylabel('AQI',fontsize=12)
plt.show()

#冬季总表
data3=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\2013-2018冬季总表.xlsx',index_col=0)
#数据处理 清洗
data3.describe()
data3drop=data3.dropna()
data3drop.describe()
#风向
data3drop.loc[data3drop['风向']=='北风','风向']=1
data3drop.loc[data3drop['风向']=='东北风','风向']=2
data3drop.loc[data3drop['风向']=='东风','风向']=3
data3drop.loc[data3drop['风向']=='东南风','风向']=4
data3drop.loc[data3drop['风向']=='南风','风向']=5
data3drop.loc[data3drop['风向']=='西南风','风向']=6
data3drop.loc[data3drop['风向']=='西风','风向']=7
data3drop.loc[data3drop['风向']=='西北风','风向']=8
#质量等级
data3drop.loc[data3drop['质量等级']=='优','质量等级']=1
data3drop.loc[data3drop['质量等级']=='良','质量等级']=2
data3drop.loc[data3drop['质量等级']=='轻度污染','质量等级']=3
data3drop.loc[data3drop['质量等级']=='中度污染','质量等级']=4
data3drop.loc[data3drop['质量等级']=='重度污染','质量等级']=5
data3drop.loc[data3drop['质量等级']=='严重污染','质量等级']=5 #因为严重污染只有一天 无法做分类所以归类合并重度污染

#只用温度 风力 风向拟合
X3=data3drop.iloc[:,7:11].astype(float)
y3=data3drop.iloc[:,0].astype(float)
X3_train,X3_test,y3_train,y3_test=model_selection.train_test_split(X3,y3,test_size=0.3,random_state=1)
linregTr3=LinearRegression()
linregTr3.fit(X3_train,y3_train)
print(linregTr3.intercept_,linregTr3.coef_)
y3_train_pred=linregTr3.predict(X3_train)
y3_test_pred=linregTr3.predict(X3_test)
train_err3=metrics.mean_squared_error(y3_train,y3_train_pred)
test_err3=metrics.mean_squared_error(y3_test,y3_test_pred)
print('the mean squar error of train and test are:{:.2f},{:.2f}'.format(train_err3,test_err3))
predict_score3=linregTr3.score(X3_test,y3_test)
print('The decision coeficient is:{:.2f}'.format(predict_score3))


#10个X（所有）拟合最终AQI函数
X2=data3drop.iloc[:,1:11].astype(float)
y2=data3drop.iloc[:,0].astype(float)
X2_train,X2_test,y2_train,y2_test=model_selection.train_test_split(X2,y2,test_size=0.3,random_state=1)
linregTr2=LinearRegression()
linregTr2.fit(X2_train,y2_train)
print(linregTr2.intercept_,linregTr2.coef_)
y2_train_pred=linregTr2.predict(X2_train)
y2_test_pred=linregTr2.predict(X2_test)
train_err2=metrics.mean_squared_error(y2_train,y2_train_pred)
test_err2=metrics.mean_squared_error(y2_test,y2_test_pred)
print('the mean squar error of train and test are:{:.2f},{:.2f}'.format(train_err2,test_err2))
predict_score2=linregTr2.score(X2_test,y2_test)
print('The decision coeficient is:{:.2f}'.format(predict_score2))


#神经网络确定质量等级
X4=data3drop.iloc[:,1:11].astype(float)
y4=data3drop.iloc[:,13].astype(float)
X4_train,X4_test,y4_train,y4_test=model_selection.train_test_split(X4,y4,test_size=0.3,random_state=1)
mlp = MLPClassifier(solver='lbfgs',alpha=10,hidden_layer_sizes=(11,11,11,11,11,11,11),random_state=1)
mlp.fit(X4_train,y4_train)
print(mlp.score(X4_train,y4_train))
y4_predicted4 = mlp.predict(X4_test)
print("Classification report for %s" % mlp)
print (metrics.classification_report(y4_test, y4_predicted4) )
print( "Confusion matrix:\n", metrics.confusion_matrix(y4_test, y4_predicted4))

#决策树确定质量等级
from sklearn import tree
X5=data3drop.iloc[:,1:11].astype(float)
y5=data3drop.iloc[:,13].astype(float)
X5_train,X5_test,y5_train,y5_test=model_selection.train_test_split(X5,y5,test_size=0.3,random_state=1)
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X5_train,y5_train)
print(clf.score(X5_train,y5_train))
y5_predicted5=clf.predict(X5_test)
print(metrics.classification_report(y5_predicted5,y5_test))
print('confusion matrix:')
print(metrics.confusion_matrix(y5_predicted5,y5_test))

#用最高，最低气温加污染物拟合函数 (能推广)
X6=data3drop.iloc[:,1:9].astype(float)
y6=data3drop.iloc[:,0].astype(float)
X6_train,X6_test,y6_train,y6_test=model_selection.train_test_split(X6,y6,test_size=0.3,random_state=1)
linregTr6=LinearRegression()
linregTr6.fit(X6_train,y6_train)
print(linregTr6.intercept_,linregTr6.coef_)
y6_train_pred=linregTr6.predict(X6_train)
y6_test_pred=linregTr6.predict(X6_test)
train_err6=metrics.mean_squared_error(y6_train,y6_train_pred)
test_err6=metrics.mean_squared_error(y6_test,y6_test_pred)
print('the mean squar error of train and test are:{:.2f},{:.2f}'.format(train_err6,test_err6))
predict_score6=linregTr6.score(X6_test,y6_test)
print('The decision coeficient is:{:.2f}'.format(predict_score6))

#额外随机选取上海某个月检验正确率
data4=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\2015年678月假设检验.xlsx',index_col=0)
X7=data4.iloc[:,2:10]
y7=data4.iloc[:,0]
predict_y7=linregTr6.predict(X7)
print(metrics.mean_squared_error(predict_y7,y7))
predict_new_Y_value=y7.astype(float)
print(stats.pearsonr(predict_new_Y_value, predict_y7))#相关性

#画出图 发现波峰波谷有明显误差
x7 = range(1,93)
plt.figure(figsize=(20,8),dpi=80)
plt.plot(x7,predict_new_Y_value,label="预测曲线",color="#F08080")
plt.plot(x7,predict_y7,label="实际曲线",color="#DB7093",linestyle="--")
_xtick_labels = [format(i) for i in x7]
plt.xticks(x7,_xtick_labels)
plt.grid(alpha=0.4,linestyle=':')
plt.legend(loc="upper left")
plt.show()
#查找原因发现污染物受月份影响非常明显 有一定周期性所以冬天与夏天在污染物系数有明显不同
O3month=data2.loc[:,'O3']
O3month.plot(title='2013/12-2018/11 每月O3指数曲线 ',LineWidth=2,marker='o',linestyle='dashed',grid=True,use_index=True)
plt.xlabel('Year',fontsize=12)
plt.ylabel('O3',fontsize=12)
plt.show()

PM25month=data2.loc[:,'PM2.5']
PM25month.plot(title='2013/12-2018/11 每月PM2.5指数曲线 ',LineWidth=2,marker='o',linestyle='dashed',grid=True,use_index=True)
plt.xlabel('Year',fontsize=12)
plt.ylabel('PM2.5',fontsize=12)
plt.show()

PM10month=data2.loc[:,'PM10']
PM10month.plot(title='2013/12-2018/11 每月PM10指数曲线 ',LineWidth=2,marker='o',linestyle='dashed',grid=True,use_index=True)
plt.xlabel('Year',fontsize=12)
plt.ylabel('PM10',fontsize=12)
plt.show()

SO2month=data2.loc[:,'SO2']
SO2month.plot(title='2013/12-2018/11 每月SO2指数曲线 ',LineWidth=2,marker='o',linestyle='dashed',grid=True,use_index=True)
plt.xlabel('Year',fontsize=12)
plt.ylabel('SO2',fontsize=12)
plt.show()

COmonth=data2.loc[:,'CO']
COmonth.plot(title='2013/12-2018/11 每月CO指数曲线 ',LineWidth=2,marker='o',linestyle='dashed',grid=True,use_index=True)
plt.xlabel('Year',fontsize=12)
plt.ylabel('CO',fontsize=12)
plt.show()

NO2month=data2.loc[:,'NO2']
NO2month.plot(title='2013/12-2018/11 每月NO2指数曲线 ',LineWidth=2,marker='o',linestyle='dashed',grid=True,use_index=True)
plt.xlabel('Year',fontsize=12)
plt.ylabel('NO2',fontsize=12)
plt.show()

#于是决定重新再用6 7 8月+12 1 2月的进行再次拟合（最后模型）

data5=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\2013-2018夏冬两季总表.xlsx',index_col=0)
X8=data5.iloc[:,1:9].astype(float)
y8=data5.iloc[:,0].astype(float)
X8_train,X8_test,y8_train,y8_test=model_selection.train_test_split(X8,y8,test_size=0.3,random_state=1)
linregTr8=LinearRegression()
linregTr8.fit(X8_train,y8_train)
print(linregTr8.intercept_,linregTr8.coef_)
y8_train_pred=linregTr8.predict(X8_train)
y8_test_pred=linregTr8.predict(X8_test)
train_err8=metrics.mean_squared_error(y8_train,y8_train_pred)
test_err8=metrics.mean_squared_error(y8_test,y8_test_pred)
print('the mean squar error of train and test are:{:.2f},{:.2f}'.format(train_err8,test_err8))
predict_score8=linregTr8.score(X8_test,y8_test)
print('The decision coeficient is:{:.2f}'.format(predict_score8))
joblib.dump(linregTr8,'回归线性模型.pkl')

#额外选取假设检验
data6=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\2014年10月2016年3月2018年5月假设检验.xlsx',index_col=0)
X9=data6.iloc[:,1:9]
y9=data6.iloc[:,0]
predict_y9=linregTr8.predict(X9)
print(metrics.mean_squared_error(predict_y9,y9))
predict_new_Y_value2=y9.astype(float)
print(stats.pearsonr(predict_new_Y_value2, predict_y9)) #相关性

x9 = range(1,94)
plt.figure(figsize=(20,8),dpi=80)
plt.plot(x9,predict_new_Y_value2,label="预测曲线",color="#F08080")
plt.plot(x9,predict_y9,label="实际曲线",color="#DB7093",linestyle="--")
_xtick_labels = [format(i) for i in x9]
plt.xticks(x7,_xtick_labels)
plt.grid(alpha=0.4,linestyle=':')
plt.legend(loc="upper left")
plt.show()

#至此上海的全部研究完 接下来进行全国的推广

#推广到苏州用上海的最后模型进行检验 选用了2018年苏州的数据
data7=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\2018年苏州空气质量指数日历史数据.xlsx',index_col=0)
data7drop=data7.dropna()
X10=data7drop.iloc[:,1:9]
y10=data7drop.iloc[:,0]
predict_y10=linregTr8.predict(X10)
print(metrics.mean_squared_error(predict_y10,y10))
predict_new_Y_value3=y10.astype(float)
print(stats.pearsonr(predict_new_Y_value3, predict_y10))

x10 = range(1,298)
plt.figure(figsize=(20,8),dpi=80)
plt.plot(x10,predict_new_Y_value3,label="预测曲线",color="#F08080")
plt.plot(x10,predict_y10,label="实际曲线",color="#DB7093",linestyle="--")
_xtick_labels = [format(i) for i in x10]
plt.xticks(x10,_xtick_labels)
plt.grid(alpha=0.4,linestyle=':')
plt.legend(loc="upper left")
plt.show()

#推广到北京
data8=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\2018年北京空气质量指数日历史数据.xlsx',index_col=0)
data8drop=data8.dropna()
X11=data8drop.iloc[:,1:9]
y11=data8drop.iloc[:,0]
predict_y11=linregTr8.predict(X11)
print(metrics.mean_squared_error(predict_y11,y11))
predict_new_Y_value4=y11.astype(float)
print(stats.pearsonr(predict_new_Y_value4, predict_y11))

x11 = range(1,271)
plt.figure(figsize=(20,8),dpi=80)
plt.plot(x11,predict_new_Y_value4,label="预测曲线",color="#F08080")
plt.plot(x11,predict_y11,label="实际曲线",color="#DB7093",linestyle="--")
_xtick_labels = [format(i) for i in x11]
plt.xticks(x11,_xtick_labels)
plt.grid(alpha=0.4,linestyle=':')
plt.legend(loc="upper left")
plt.show()

#说明此线性模型具有一定推广性

#至此开始做每个污染物和气温的时序分析
# O3
data10=data8=pd.read_excel('C:\\Users\\11384\\Desktop\\python大作业\\预测\\上海\\shanghai.xlsx',index_col=0)

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data10.loc[:,'O3_8h'])
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data10.loc[:,'O3_8h'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data10.loc[:,'O3_8h'])) 
O3last2month=data10.iloc[-60:,7]

D_O3 = O3last2month.diff().dropna() #对原数据进行1阶差分，删除非法值
print('差分序列－ADF－检验结果为：', ADF(D_O3)) #平稳性检测

from statsmodels.tsa.arima_model import ARIMA
O3last2month= O3last2month.astype(float)
pmax = int(len(D_O3)/10) #一般阶数不超过length/10
qmax = int(len(D_O3)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       try: #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(O3last2month, (p,1,q)).fit().aic)
       except:
          tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))
model = ARIMA(O3last2month, (p,1,q)).fit() 
model.summary2() #给出模型报告
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。
preO3=model.forecast(1)[0]
#PM2.5
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data10.loc[:,'PM2.5'])
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data1.loc[:,'PM2.5'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data10.loc[:,'PM2.5'])) 

PM25last2month=data10.iloc[-60:,2]
from statsmodels.tsa.arima_model import ARIMA
PM25last2month= PM25last2month.astype(float)
pmax = int(len(PM25last2month)/10) #一般阶数不超过length/10
qmax = int(len(PM25last2month)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       try: #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(PM25last2month, (p,0,q)).fit().aic)
       except:
          tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))
model = ARIMA(PM25last2month, (p,0,q)).fit() 
model.summary2() #给出模型报告
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。
prePM25=model.forecast(1)[0]
#Co
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data10.loc[:,'CO'])
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data1.loc[:,'CO'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data10.loc[:,'CO'])) 

COlast2month=data10.iloc[-60:,5]
from statsmodels.tsa.arima_model import ARIMA
COlast2month= COlast2month.astype(float)
pmax = int(len(COlast2month)/10) #一般阶数不超过length/10
qmax = int(len(COlast2month)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       try: #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(COlast2month, (p,0,q)).fit().aic)
       except:
          tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))
model = ARIMA(COlast2month, (p,0,q)).fit() 
model.summary2() #给出模型报告
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。
preCO=model.forecast(1)[0]

#PM10
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data10.loc[:,'PM10'])
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data1.loc[:,'PM10'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data10.loc[:,'PM10'])) 

PM10last2month=data10.iloc[-60:,3]
from statsmodels.tsa.arima_model import ARIMA
PM10last2month= PM10last2month.astype(float)
pmax = int(len(PM10last2month)/10) #一般阶数不超过length/10
qmax = int(len(PM10last2month)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       try: #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(PM10last2month, (p,0,q)).fit().aic)
       except:
          tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))
model = ARIMA(PM10last2month, (p,0,q)).fit() 
model.summary2() #给出模型报告
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。
prePM10=model.forecast(1)[0]

#NO2
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data10.loc[:,'NO2'])
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data10.loc[:,'NO2'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data1.loc[:,'NO2'])) 

NO2last2month=data10.iloc[-60:,6]
from statsmodels.tsa.arima_model import ARIMA
NO2last2month= NO2last2month.astype(float)
pmax = int(len(NO2last2month)/10) #一般阶数不超过length/10
qmax = int(len(NO2last2month)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       try: #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(NO2last2month, (p,0,q)).fit().aic)
       except:
          tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))
model = ARIMA(NO2last2month, (p,0,q)).fit()
model.summary2() #给出模型报告
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。
preNO2=model.forecast(1)[0]

#SO2
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data10.loc[:,'SO2'])
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data10.loc[:,'SO2'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data10.loc[:,'SO2'])) 

SO2last2month=data10.iloc[-60:,4]
from statsmodels.tsa.arima_model import ARIMA
SO2last2month= SO2last2month.astype(float)
pmax = int(len(SO2last2month)/10) #一般阶数不超过length/10
qmax = int(len(SO2last2month)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       try: #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(SO2last2month, (p,0,q)).fit().aic)
       except:
          tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))
model = ARIMA(SO2last2month, (p,0,q)).fit() 
model.summary2() #给出模型报告
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。
preSO2=model.forecast(1)[0]

#最低气温
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data10.loc[:,u'最低气温'])
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data10.loc[:,u'最低气温'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data10.loc[:,u'最低气温'])) 

D_TMIN = O3last2month.diff().dropna() #对原数据进行1阶差分，删除非法值
print('差分序列－ADF－检验结果为：', ADF(D_TMIN)) #平稳性检测

TMINlast2month=data10.iloc[-60:,8]
from statsmodels.tsa.arima_model import ARIMA
TMINlast2month= TMINlast2month.astype(float)
pmax = int(len(D_TMIN)/10) #一般阶数不超过length/10
qmax = int(len(D_TMIN)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       try: #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(TMINlast2month, (p,1,q)).fit().aic)
       except:
          tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))
model = ARIMA(TMINlast2month, (p,1,q)).fit() 
model.summary2() #给出模型报告
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。
preTMIN=model.forecast(1)[0]

#最高气温
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data10.loc[:,u'最高气温'])
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声-检验结果：', acorr_ljungbox(data10.loc[:,u'最高气温'], lags=1))
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF-检验结果：', ADF(data10.loc[:,u'最高气温'])) 

D_TMAX = O3last2month.diff().dropna() #对原数据进行1阶差分，删除非法值
print('差分序列－ADF－检验结果为：', ADF(D_TMAX)) #平稳性检测

TMAXlast2month=data10.iloc[-60:,9]
from statsmodels.tsa.arima_model import ARIMA
TMAXlast2month= TMAXlast2month.astype(float)
pmax = int(len(D_TMAX)/10) #一般阶数不超过length/10
qmax = int(len(D_TMAX)/10) #一般阶数不超过length/10
e_matrix = [] #评价矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       try: #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(TMAXlast2month, (p,2,q)).fit().aic)
       except:
          tmp.append(None)
    e_matrix.append(tmp)
e_matrix = pd.DataFrame(e_matrix) #从中可以找出最小值
p,q = e_matrix.stack().idxmin() #先用stack展平，然后用找出最小值位置。
print('AIC最小的p值和q值为：%s、%s' %(p,q))
model = ARIMA(TMAXlast2month, (p,2,q)).fit() #这个用d=1没有值
model.summary2() #给出模型报告
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。
preTMAX=model.forecast(1)[0]
 


predictX=np.array([prePM25,prePM10,preSO2,preCO,preNO2,preO3,preTMIN,preTMAX]).T
print(predictX)
preAQI=linregTr8.predict(predictX)
print(preAQI)
