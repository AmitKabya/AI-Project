import numpy as np
import pandas as pd
import PCA


class Data:
    def __init__(self, n=0):
        self.mushrooms = []
        file = open("mushrooms_data.txt", 'r')

        for line in file:
            mushroom = line.split(",")
            mushroom[-1] = mushroom[-1][0]
            mushroom.pop(5)
            self.mushrooms.append(mushroom)

        self.dataF = {}
        self.data = self.mushrooms
        self.attributes = ['classes',
                           'cap-shape',
                           'cap-surface',
                           'cap-color',
                           'bruises',
                           'gill-attachment',
                           'gill-spacing',
                           'gill-size',
                           'gill-color',
                           'stalk-shape',
                           'stalk-surface-above-ring',
                           'stalk-surface-below-ring',
                           'stalk-color-above-ring',
                           'stalk-color-below-ring',
                           'veil-type',
                           'veil-color',
                           'ring-number',
                           'ring-type',
                           'spore-print-color',
                           'population',
                           'habitat']

        self.convert()
        self.toPanda()

        if 0 < n < 21:
            self.Dimension_Reduction(n)

        elif n == 0:
            n = 21

        else:
            n = 21
            print("inputError: dimension must be an integer between 1 and 20")
        print(f"running on {n} dimensions")

    def convert(self):
        converted = []

        for i in range(len(self.mushrooms)):
            converted.append([])
            converted[-1].append(1 if self.mushrooms[i][0] == 'e' else 0)

            for j in range(1, len(self.mushrooms[i])):
                converted[-1].append(ord(self.mushrooms[i][j]))

        self.data = converted

    def toPanda(self):
        for attr in self.attributes:
            self.dataF[attr] = []

        for j in range(len(self.data[0])):
            for i in range(len(self.data)):
                self.dataF[self.attributes[j]].append(self.data[i][j])

        self.dataF = pd.DataFrame(self.dataF)

    def Dimension_Reduction(self, n=3):
        npdata = np.array(self.data)
        X = npdata[:, 1:]
        y = npdata[:, 0]
        self.dataF = PCA.pca_predict(X, y, n)


class MissingData:
    def __init__(self, n=0):
        self.mushrooms = []
        file = open("mushrooms_data_missing.txt", 'r')

        for line in file:
            mushroom = line.split(",")
            mushroom[-1] = mushroom[-1][0]
            mushroom.pop(5)
            self.mushrooms.append(mushroom)

        self.dataF = {}
        self.data = self.mushrooms
        self.missingDataF = {}
        self.missingData = []
        self.attributes = ['classes',
                           'cap-shape',
                           'cap-surface',
                           'cap-color',
                           'bruises',
                           'gill-attachment',
                           'gill-spacing',
                           'gill-size',
                           'gill-color',
                           'stalk-shape',
                           'stalk-surface-above-ring',
                           'stalk-surface-below-ring',
                           'stalk-color-above-ring',
                           'stalk-color-below-ring',
                           'veil-type',
                           'veil-color',
                           'ring-number',
                           'ring-type',
                           'spore-print-color',
                           'population',
                           'habitat']

        self.convert()
        self.toPanda()

        if 0 < n < 21:
            self.Dimension_Reduction(n)

        elif n == 0:
            n = 21

        else:
            n = 21
            print("inputError: dimension must be an integer between 1 and 20")
        print(f"running on {n} dimensions")

    def convert(self):
        converted = []
        missingConverted = []

        missingClass = False
        for i in range(len(self.mushrooms)):
            converted.append([])

            if self.mushrooms[i][0] == "-":
                converted[-1].append(-1)
                missingClass = True
            else:
                converted[-1].append(1 if self.mushrooms[i][0] == 'e' else 0)

            for j in range(1, len(self.mushrooms[i])):
                if self.mushrooms[i][j] == "-":
                    converted[-1].append(-1)

                else:
                    converted[-1].append(ord(self.mushrooms[i][j]))

            if missingClass:
                missingConverted.append(converted.pop())
                missingClass = False

        self.data = converted
        self.missingData = missingConverted

    def toPanda(self):
        for attr in self.attributes:
            self.dataF[attr] = []
            self.missingDataF[attr] = []

        for j in range(len(self.data[0])):
            for i in range(len(self.data)):
                self.dataF[self.attributes[j]].append(self.data[i][j])

        for j in range(len(self.missingData[0])):
            for i in range(len(self.missingData)):
                self.missingDataF[self.attributes[j]].append(self.data[i][j])

        self.dataF = pd.DataFrame(self.dataF)
        self.missingDataF = pd.DataFrame(self.missingDataF)

    def Dimension_Reduction(self, n=3):
        npdata = np.array(self.data)
        X = npdata[:, 1:]
        y = npdata[:, 0]
        self.dataF = PCA.pca_predict(X, y, n)
        npmissing = np.array(self.missingData)
        X_missing = npmissing[:, 1:]
        y_missing = npmissing[:, 0]
        self.missingDataF = PCA.pca_predict(X_missing, y_missing, dim=n)
