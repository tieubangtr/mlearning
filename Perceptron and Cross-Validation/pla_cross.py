from tkinter import *
from tkinter.ttk import Combobox
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
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
    data = pd.read_csv('Employee.csv')
    data = data[["Education","City","Gender","EverBenched","PaymentTier","ExperienceInCurrentDomain","LeaveOrNot"]]
    X = np.array(data.drop(["LeaveOrNot"], 1))
    y = np.array(data["LeaveOrNot"])

    X1 = X

    encoder = OrdinalEncoder()
    X = encoder.fit_transform(X)

    perceptron = Perceptron()

    count = 1
    best_set = 0;
    best_accuracy_score = 0

    selected_xTrain = []
    selected_xTest = []
    selected_yTrain = []
    selected_yTest = []

    # K Fold for cross validation
    kf = KFold(n_splits=4, shuffle=True)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "Length:", len(train_index), "TEST:", test_index, "Length:", len(test_index))
        xTrain, xTest = X[train_index], X[test_index]
        yTrain, yTest = y[train_index], y[test_index]
        perceptron.fit(xTrain, yTrain)
        train_error = perceptron.score(xTrain, yTrain)
        test_error = perceptron.score(xTest, yTest)
        # print("Train error: ", train_error)
        # print("Test error: ", test_error)
        print("Sum: ", train_error + test_error)
        yPredict = perceptron.predict(xTest)
        accuracy_index = accuracy_score(yTest, yPredict)
        accuracy = "Độ chính xác của K-Fold lần thứ " + str(count) + " : " + str(round(accuracy_score(yTest, yPredict) * 100, 3)) + " %"
        print(accuracy)
        # Choosing best model
        if(accuracy_index > best_accuracy_score):
            selected_xTrain = xTrain
            selected_xTest = xTest
            selected_yTrain = yTrain
            selected_yTest = yTest
            best_accuracy_score = accuracy_index
            best_set = count
        count = count + 1

    print("Mô hình được lựa chọn: " + str(best_set) + " - Độ chính xác: " + str(round(best_accuracy_score * 100, 3)) + " %")

    def __init__(self, win):
        #Label init
        self.lbl1=Label(win, text='Education')
        self.lbl2=Label(win, text='City')
        self.lbl3=Label(win, text='Gender')
        self.lbl4=Label(win, text='Ever Benched')
        self.lbl5=Label(win, text='Payment Tier')
        self.lbl6=Label(win, text='Experience In Current Domain')
        #Combobox init
        self.cb1=Combobox(win, values=getValue(self.data["Education"]))
        self.cb1.current(0)
        self.cb2=Combobox(win, values=getValue(self.data["City"]))
        self.cb2.current(0)
        self.cb3=Combobox(win, values=getValue(self.data["Gender"]))
        self.cb3.current(0)
        self.cb4=Combobox(win, values=getValue(self.data["EverBenched"]))
        self.cb4.current(0)

        #Button init
        self.b1=Button(win, text='Dự đoán', command=self.getPct)
        #Result input box init
        self.e1=Entry()
        self.e2=Entry()
        self.e3=Entry()
        #Accuracy
        self.lbl9=Label(win, text='Độ chính xác: ')
        #Placing elements
        #Comboboxes
        self.lbl1.place(x=50, y=50)
        self.cb1.place(x=250, y=50)
        self.lbl2.place(x=50, y=100)
        self.cb2.place(x=250, y=100)
        self.lbl3.place(x=50, y=150)
        self.cb3.place(x=250, y=150)
        self.lbl4.place(x=50, y=200)
        self.cb4.place(x=250, y=200)
        #Buttons
        self.b1.place(x=100, y=450)
        #Inputs
        self.lbl5.place(x=50, y=250)
        self.e2.place(x=250, y=250)
        self.lbl6.place(x=50, y=300)
        self.e3.place(x=250, y=300)
        self.e1.place(x=100, y=500)
        #Accuracy
        self.lbl9.place(x=100, y=550)
    def getPct(self):
        # Get data and sync
        paymentTier = int(self.e2.get())
        expInCurrentDomain = int(self.e3.get())
        data = (self.cb1.get(), self.cb2.get(), self.cb3.get(), self.cb4.get(), paymentTier, expInCurrentDomain)
        data = np.array(data,dtype='object')
        data = np.array(data).reshape(1, -1)
        X1 = np.append(self.X1, data, axis = 0)
        X1 = self.encoder.fit_transform(X1)
        data = np.array([X1[len(X1) - 1]])

        self.perceptron.fit(self.selected_xTrain, self.selected_yTrain)

        y_predict = self.perceptron.predict(self.selected_xTest)
        predict_output = self.perceptron.predict(data)

        output_accuracy = "Độ chính xác: " + str(round(self.best_accuracy_score * 100, 3)) + " %"
        print(output_accuracy)
        print("Recall Score của mô hình: " + str(round(recall_score(y_predict, self.selected_yTest) * 100, 3)) + " %")
        print("Precision Score của mô hình: " + str(round(precision_score(y_predict, self.selected_yTest) * 100, 3)) + " %")
        print("F1 Score của mô hình: " + str(round(f1_score(y_predict, self.selected_yTest) * 100, 3)) + " %")
        # Clear data
        self.e1.delete(0, 'end')
        self.e1.insert(END, "No" if (int(predict_output[0]) == 0) else "Yes")
        self.lbl9.config(text=output_accuracy)


root = Tk()
newWindow = TkWindow(root)
root.title('Perceptron')
root.geometry("500x650")
root.mainloop()
