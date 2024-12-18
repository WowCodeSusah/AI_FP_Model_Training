import numpy as np
import pandas as pd

def prepData(csvName):
    df = pd.read_csv(csvName + ".csv")

    df = df.drop(df[df['folderName'] == 'folderName'].index)
    df = df.dropna()

    # Remove columns
    df = df.drop(columns=["folderName", "fileName"])

    # Split the data
    df1 = df.iloc[:, :99]
    df2 = df.iloc[:, 99:]

    return df1, df2

def createSplit(dataframe, kAmount):
    # Make Disticted Groups by K Amount
    foldArray = []
    start = 0
    increaseAmount = int(len(dataframe) / kAmount) 
    for x in range(0, kAmount - 1):
        folds = dataframe.iloc[start: start + increaseAmount, :]
        foldArray.append(folds)
        start = start + increaseAmount

    foldArray.append(dataframe.iloc[start:len(dataframe), :])

    return foldArray

def createTestSplit(df1, df2, folds):
    initialX = createSplit(df1, folds)
    initialY = createSplit(df2, folds)

    splitX = []
    splitY = []

    for data in range(0, len(initialX)):
        dataX = []
        dataY = []

        testFrameX = []
        testFrameY = []
        for data2 in range(0, len(initialX)):
            if data != data2:
                testFrameX.append(initialX[data2])
                testFrameY.append(initialY[data2])

        dataX.append(pd.concat(testFrameX))
        dataY.append(pd.concat(testFrameY))

        dataX.append(initialX[data]) 
        dataY.append(initialY[data])

        splitX.append(dataX)
        splitY.append(dataY)

    return splitX, splitY


# Preps the dataframe
df1, df2 = prepData('final')

# Different Folds For testing and validation
testX, testY = createTestSplit(df1, df2, 5)

import prediction
import plotLoss
from sklearn import metrics

accuracyList = []
precisionList = [] 
recallList = [] 
f1List = []

for x in range(0, len(testX)):
    xtest = testX[x][0].apply(pd.to_numeric)
    xvalid = testX[x][1].apply(pd.to_numeric)

    ytest = testY[x][0].replace({"False": 0, "True":1})
    yvalid = testY[x][1].replace({"False": 0, "True":1})

    xtest = xtest.to_numpy() 
    xvalid = xvalid.to_numpy()

    ytest = ytest.to_numpy()
    yvalid = yvalid.to_numpy()

    accuracy, precision, recall, f1, history, y_pred = prediction.VGG(xtest, ytest, xvalid, yvalid)

    confusion_matrix = metrics.confusion_matrix(yvalid, y_pred)

    cm_display =  metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

    cm_display.plot().figure_.savefig('graphs/confusion_matrix_' + str(x) + '.png')

    plotLoss.plot_history(history, str(x))

    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)

print('Accuracy')
print(accuracyList)
print("------------------------------")
print('Precision')
print(precisionList)
print("------------------------------")
print('Recall')
print(recallList)
print("------------------------------")
print('F1')
print(f1List)
print("------------------------------")