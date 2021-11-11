from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn

data = pd.read_csv('history_data2.csv')
date = np.array(data["Date time"])
print(date)
data = data[["Temperature", "Minimum Temperature", "Relative Humidity"]]

x = np.array(data.drop(["Temperature"], 1))
y = np.array(data["Temperature"])

#Split data 70/30
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.3)

#Training
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)

print("Hệ số hồi quy: ", model.coef_)
print("Intercept: ", model.intercept_)

accuracy = model.score(xTest, yTest)
print("Độ chính xác của mô hình: ", round(accuracy*100, 3), "%")

#Evaluation
error = []
testVals = model.predict(xTest)
dateIndex = []
i = 0
for test in testVals:
    error.append(yTest[i] - test)
    dateIndex.append(date[i][0:5]) #Get the month value only
    print("Thực tế: " + str(yTest[i]) + " Dự đoán: " + str(round(test, 2)) + " Sai số: " + str(round(error[i], 2)))
    i = i + 1

#Plot
f, axes = plt.subplots(2)
#Plot 1
axes[0].plot(dateIndex, testVals, label="Dự đoán")
axes[0].plot(dateIndex, yTest, label="Thực tế")
axes[0].legend(["Dự đoán", "Thực tế"], loc="lower right")
axes[0].set_xticks(dateIndex[::5])
axes[0].margins(0.01)
#Plot 2
axes[1].plot(dateIndex, yTest, label="Thực tế")
axes[1].legend(["Dự đoán"], loc="lower right")
axes[1].set_xticks(dateIndex[::5])
axes[1].margins(0.01)
#Init plot
plt.xlabel("Ngày trong năm")
plt.ylabel("Nhiệt độ")
plt.show()