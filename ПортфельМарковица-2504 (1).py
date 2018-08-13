
# coding: utf-8

# In[78]:


#author Damir Pavlin, 2018 Moscow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# In[79]:


stockNames = ['EBAY', 'F', 'GE', 'INTC', 'JNJ', 'MSFT', 'NKE', 'XOM' ] #лист с именами эмитентов
closeVector = {} #пустой словарь для векторов котировок эмитентов
logYield = {} #пустой словарь для лог.доходностей эмитентов
weightStocks= {} #пустой словарь для весов


# In[82]:


def MarkowitzPavlin(year, weightPreviousStocks): #для первой итерации предыдущую доходость надо ставить 0

    dt = 252 #число торговых дней в году
    stockNames = ['EBAY', 'F', 'GE', 'INTC', 'JNJ', 'MSFT', 'NKE', 'XOM' ] #лист с именами эмитентов
    closeVector = {} #пустой словарь для векторов котировок эмитентов
    logYield = {} #пустой словарь для лог.доходностей эмитентов
    weightStocks= {} #пустой словарь для весов



    for i in range(len(stockNames)): #цикл по проверке Nand и составлению векторов котировок
        frame = pd.read_csv(stockNames[i] + ".csv", header = 0, sep = ',')
        close = frame['Close']
        logProf = []
        k = 0
        for j in range (len(close)): #проверка на наличие нанов
            if type(close[j]) == np.float64: #TODO сделать очистку данных в случае Nanov
                k += 1
            else:
                print ("Начальник у нас Nan, возможно строка! Эмитент ", stockNames[i], " строка ", j, ", значени: ", close[j])

        weightStocks[stockNames[i]]=0 #задание пустых весов

        logProf.append(0.1)
        for m in range (len(close)-1): #заполнение вектора логдоходности
            logProf.append(np.log(close[m+1]/close[m]))

        closeVector[stockNames[i]] = close #составление векторов котировок
        logYield[stockNames[i]] = logProf #составление векторов логдоходностей

        #print ("Эмитент: ", stockNames[i], " всего строк: ", k, "cтрок в лог.дох: ", len(logProf) )

    #print (closeVector)

    covDoubleArray = [] #задание пустой матрицы ковариаций 
    covDoubleArray.append([])
    covDoubleArray[0].append(0)
    for i in range(len(stockNames)):
        covDoubleArray[0].append(stockNames[i])

    for i in range(1, len(stockNames)+1):
        covDoubleArray.append([])
        covDoubleArray[i].append(stockNames[i-1])
        for j in range(1, len(stockNames)+1):
            covDoubleArray[i].append(0)

    corDoubleArray = [] #задание пустой матрицы корреляций
    corDoubleArray.append([])
    corDoubleArray[0].append(0)
    for i in range(len(stockNames)):
        corDoubleArray[0].append(stockNames[i])

    for i in range(1, len(stockNames)+1):
        corDoubleArray.append([])
        corDoubleArray[i].append(stockNames[i-1])
        for j in range(1, len(stockNames)+1):
            corDoubleArray[i].append(0)




    #print(covDoubleArray)    


    #year = 2003 #год портфеля
    n = year-2003

    middleYield = {} #пустой словарь для средних год.лог.доходностей
    stDev = {} #пустой словарь для стандартного отклонения

    #print ("Средние годовые доходности и риск за ", year, " год:\n")
    for i in range(len(stockNames)):
        middleYield[stockNames[i]] = np.average(logYield[stockNames[i]][dt*n : dt + dt*n]) #средняя год.лог.доходность
        stDev[stockNames[i]] = np.std(logYield[stockNames[i]][dt*n : dt + dt*n]) #станд.откл., он же риск

        #TODO отформатировать вывод
        #print(stockNames[i], " доходность : ", middleYield[stockNames[i]], " ; риск : ",  stDev[stockNames[i]])

    maxCov = [0, '', ''] #КОВАРИАЦИИ
    for i in range(1, len(stockNames) + 1):
        for j in range(1, len(stockNames) + 1):
            covDoubleArray[i][j] = round(np.cov(logYield[stockNames[i-1]][dt*n : dt + dt*n], logYield[stockNames[j-1]][dt*n : dt + dt*n])[0][1],7)
            if stockNames[i-1] == stockNames[j-1]:
                covDoubleArray[i][j] = '' #вышибаем диагональ
            else:
                if covDoubleArray[i][j] > maxCov[0]: #вычисляем наиболее зависимые инструменты
                    maxCov[0] = covDoubleArray[i][j]
                    maxCov[1] = stockNames[j-1]
                    maxCov[2] = stockNames[i-1]

    dfCov = pd.DataFrame(covDoubleArray) #таблица пандас с ковариациями
    print ("\n\nГод: ", year)
    print ("Наиболее зависимые бумаги ", maxCov[1], " и ", maxCov[2], ", ковариация: ", maxCov[0])
    print ("Матрица ковариаций")
    display (dfCov)

    maxCor = [0, '', ''] #КОРРЕЛЯЦИИ
    for i in range(1, len(stockNames) + 1):
        for j in range(1, len(stockNames) + 1):
            corDoubleArray[i][j] = round(np.corrcoef(logYield[stockNames[i-1]][dt*n : dt + dt*n], logYield[stockNames[j-1]][dt*n : dt + dt*n])[0][1],6)
            #if stockNames[i-1] == stockNames[j-1]:
                #corDoubleArray[i][j] = 1 #вышибаем диагональ
            #else:
                #if corDoubleArray[i][j] > maxCor[0]: #вычисляем наиболее зависимые инструменты
                   # maxCor[0] = corDoubleArray[i][j]
                   # maxCor[1] = stockNames[j-1]
                    #maxCor[2] = stockNames[i-1]

    dfCor = pd.DataFrame(corDoubleArray) #таблица пандас с ковариациями
    #print ("Наиболее коррелируемые бумаги ", maxCor[1], " и ", maxCor[2], ", коэффициент Пирсона: ", maxCor[0])
    #print ("Матрица корреляций")
    #dfCor

    x0 = [0,0,0,0,0,1,0,0] #начальный массив долей эмитентов
    b =(0, 1) #условия для долей, от нуля до единицы
    bnds = (b, b, b, b, b, b, b, b)

    def cond(x0): #условие суммы всех долей равной единице
        flag = 1 
        sum = 0
        for i in range(8):
            sum += x0[i]
        flag = flag - sum
        return (flag)

    def riskPortf(x0):
        riskPort = np.sqrt((sum((x0[i]**2)*(stDev[stockNames[i]]**2) for i in range(8)) +         2*sum(sum(x0[i]*x0[j]*corDoubleArray[i+1][j+1]*stDev[stockNames[i]]*                  stDev[stockNames[j]] for j in range(i, 8)) for i in range(7)))) #риск инвестиционного портфеля
        return(riskPort)     

    con = {'type': 'eq', 'fun': cond} #формулирование условия

    sol = minimize(riskPortf, x0, method = 'SLSQP', bounds = bnds, constraints = con ) #Задача линейного программирования


    #передача долей эмитентов
    for i in range(len(stockNames)):
        weightStocks[stockNames[i]] = sol.x[i]

    yieldPortf = sum(weightStocks[stockNames[i]]*middleYield[stockNames[i]] for i in range(8))
    #print ("Год: ", year)
    print ("Доли эмитентов: ", weightStocks)
    print ("\nРиск портфеля: ", sol.fun)
    print ("Ожидаемая доходность портфеля: ", yieldPortf )


    # составление диаграммы
    labels = stockNames
    sizes = []
    for i in range(8):
        sizes.append(weightStocks[stockNames[i]])

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  
    plt.show()
    
    #доходность предыдущего портфеля
    if weightPreviousStocks != 0:
        yieldPreviousPortf = sum(weightPreviousStocks[stockNames[i]]*middleYield[stockNames[i]] for i in range(8))
        print("Доходность прошлогоднего портфеля: ", round(yieldPreviousPortf*100,2), " %" )
        
    
    
    return(weightStocks, yieldPortf )


# In[87]:


names = ['Ebay', 'Ford Company', 'General Electrics', 'Intel', 'Johnson & Johnson', 'Microsoft', 'Nike', 'Exxxon Mobil']
stockNames = ['EBAY', 'F', 'GE', 'INTC', 'JNJ', 'MSFT', 'NKE', 'XOM' ]
allNames = pd.DataFrame(names, stockNames)
display(allNames)

grabber = [] #грязный грэббер
histYields = [] #пустой массив для исторических доходностей
weightPreviousStocks = {} #пустой словарь для предыдущих лет

for i in range(15):
    histYields.append(0)

grabber = MarkowitzPavlin(2003, 0)    #первый запуск
histYields[0] = grabber[1]
weightPreviousStocks = grabber[0]

for i in range(2004, 2018):
    grabber = []
    grabber = MarkowitzPavlin(i, weightPreviousStocks)
    histYields[i-2003] = grabber[1]
    weightPreviousStocks = {}
    weightPreviousStocks = grabber[0]


y = histYields #построение графика доходностей
x = []
for i in range (2003, 2018):
    x.append(i)
dfHistYields = pd.DataFrame(x, y)
print ("Таблица доходностей портфелей по годам: ")
display(dfHistYields)
plt.plot(x, y, color="green")
print ("График доходностей портфелей минимального риска")
plt.show()
print (" Аналитический комментарий. На протяжении всего исследования можно было наблюдать сильную зависимостькомпаний Microsoft и Intel, это связано с тем, что они принадлежат одному сектору экономики. Стоит отметитьуникальное положение компаний Exxxon Mobil и Johnson & Johnson, в следствии того, что  у них нет конкурентовв данной выборке (а она является топом SP500 на 2003 год), это позволяло им, многие года занимать значительный объёмпортфеля. Примечательно также падение рынка 2008 года, что также ярко отразилось даже на нашем, консервативном портфеле\n Дамир Павлин. 2018, г. Москва.")
    
    


# In[ ]:





# In[ ]:



                    

