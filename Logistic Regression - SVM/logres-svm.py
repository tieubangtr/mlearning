from tkinter import *
from tkinter.ttk import Combobox
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def getValue(list):
    result = []
    dictionary = {}
    for i in (list):
        if(i in dictionary):
            dictionary[i] += 1
        else:
            dictionary[i] = 1
    for key in dictionary.keys():
        result.append(key)
    return result

class TkWindow:
    data = pd.read_csv('data2.csv')
    data = data[["acousticness","danceability","energy","instrumentalness","liveness","loudness","speechiness","valence", "target"]]
    x = np.array(data.drop(["target"], 1))
    y = np.array(data["target"])
    def __init__(self, win):
        #Label init
        self.lbl1=Label(win, text='acousticness')
        self.lbl2=Label(win, text='danceability')
        self.lbl3=Label(win, text='energy')
        self.lbl4=Label(win, text='instrumentalness')
        self.lbl5=Label(win, text='liveness')
        self.lbl6=Label(win, text='loudness')
        self.lbl7=Label(win, text='speechiness')
        self.lbl8=Label(win, text='valence')
        #Combobox init
        self.cb1=Combobox(win, values=getValue(self.data["acousticness"]))
        self.cb1.current(0)
        self.cb2=Combobox(win, values=getValue(self.data["danceability"]))
        self.cb2.current(0)
        self.cb3=Combobox(win, values=getValue(self.data["energy"]))
        self.cb3.current(0)
        self.cb4=Combobox(win, values=getValue(self.data["instrumentalness"]))
        self.cb4.current(0)
        self.cb5=Combobox(win, values=getValue(self.data["liveness"]))
        self.cb5.current(0)
        self.cb6=Combobox(win, values=getValue(self.data["loudness"]))
        self.cb6.current(0)
        self.cb7=Combobox(win, values=getValue(self.data["speechiness"]))
        self.cb7.current(0)
        self.cb8=Combobox(win, values=getValue(self.data["valence"]))
        self.cb8.current(0)
        #Button init
        self.b1=Button(win, text='Dự đoán Logistic Regression', command=self.getLogReg)
        self.b2=Button(win, text='Dự đoán SVM', command=self.getSvm)
        #Result input box init
        self.e1=Entry()
        self.e2=Entry()
        #Accuracy
        self.lbl9=Label(win, text='Độ chính xác: ')
        self.lbl10=Label(win, text='Độ chính xác: ')
        #Placing elements
        #Comboboxes
        self.lbl1.place(x=100, y=50)
        self.cb1.place(x=200, y=50)
        self.lbl2.place(x=100, y=100)
        self.cb2.place(x=200, y=100)
        self.lbl3.place(x=100, y=150)
        self.cb3.place(x=200, y=150)
        self.lbl4.place(x=100, y=200)
        self.cb4.place(x=200, y=200)
        self.lbl5.place(x=100, y=250)
        self.cb5.place(x=200, y=250)
        self.lbl6.place(x=100, y=300)
        self.cb6.place(x=200, y=300)
        self.lbl7.place(x=100, y=350)
        self.cb7.place(x=200, y=350)
        self.lbl8.place(x=100, y=400)
        self.cb8.place(x=200, y=400)
        #Buttons
        self.b1.place(x=100, y=450)
        self.b2.place(x=300, y=450)
        #Inputs
        self.e1.place(x=100, y=500)
        self.e2.place(x=300, y=500)
        #Accuracy
        self.lbl9.place(x=100, y=550)
        self.lbl10.place(x=300, y=550)
    def getLogReg(self):
        # Logistic Regression
        data = np.array([self.cb1.get(), self.cb2.get(), self.cb3.get(), self.cb4.get(), self.cb5.get(), self.cb6.get(), self.cb7.get(), self.cb8.get()]).reshape(1, -1)
        xTrain, xTest, yTrain, yTest = train_test_split(self.x, self.y, test_size=0.3, shuffle = True)
        xTest1 = np.append(xTest, data, axis = 0)

        encoder = OrdinalEncoder()
        xTrain = encoder.fit_transform(xTrain)
        xTest = encoder.transform(xTest)
        xTest1 = encoder.transform(xTest1)

        LogReg = LogisticRegression()
        LogReg.fit(xTrain, yTrain)
        yPredictLogReg = LogReg.predict(xTest)

        predictOutput = LogReg.predict(np.array([xTest1[len(xTest1) - 1]]))

        accuracy = "Độ chính xác: " + str(round(accuracy_score(yTest, yPredictLogReg) * 100, 3)) + " %"
        # Clear data
        self.e1.delete(0, 'end')
        self.e1.insert(END, str(predictOutput[0]))
        print(predictOutput[0])
        self.lbl9.config(text=accuracy)
    def getSvm(self):
        #SVM
        data = np.array([self.cb1.get(), self.cb2.get(), self.cb3.get(), self.cb4.get(), self.cb5.get(), self.cb6.get(), self.cb7.get(), self.cb8.get()]).reshape(1, -1)
        xTrain, xTest, yTrain, yTest = train_test_split(self.x, self.y, test_size=0.3, shuffle = True)
        xTest1 = np.append(xTest, data, axis = 0)
        encoder = OrdinalEncoder()
        xTrain = encoder.fit_transform(xTrain)
        xTest = encoder.transform(xTest)
        xTest1 = encoder.transform(xTest1)

        svmachines = svm.SVC()
        svmachines.fit(xTrain, yTrain)

        predictOutput = svmachines.predict(np.array([xTest1[len(xTest1) - 1]]))
        yPredictSVM = svmachines.predict(xTest)
        accuracy = "Độ chính xác: " + str(round(accuracy_score(yTest, yPredictSVM) * 100, 3)) + " %"
        # Clear data
        self.e2.delete(0, 'end')
        self.e2.insert(END, str(predictOutput[0]))
        print(predictOutput[0])
        self.lbl10.config(text=accuracy)


root = Tk()
newWindow = TkWindow(root)
root.title('Logistic Regression and SVM')
root.geometry("500x650")
root.mainloop()
