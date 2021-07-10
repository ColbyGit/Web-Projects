# import
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Open main text file
english_file = open("english.txt")
german_file = open("german.txt")
danish_file = open("danish.txt")

# Open training files
english_training_file = open("english_training.txt", "r+")
german_training_file = open("german_training.txt", "r+")
danish_training_file = open("danish_training.txt", "r+")
english_testing_file = open("english_test.txt", "r+")
german_testing_file = open("german_test.txt", "r+")
danish_testing_file = open("danish_test.txt", "r+")

# Create lines for each file
english_lines = english_file.readlines()
german_lines = german_file.readlines()
danish_lines = danish_file.readlines()
english_training_lines = english_training_file.readlines()
german_training_lines = german_training_file.readlines()
danish_training_lines = danish_training_file.readlines()
english_testing_lines = english_testing_file.readlines()
german_testing_lines = german_testing_file.readlines()
danish_testing_lines = danish_testing_file.readlines()

# # Split the main language files 90% for training and 10% for testing

###WARNING ONLY RUN ONCE OR THIS WILL DOUBLE THE TEXT FILES###
# THE FILES HAVE ALREADY BEEN POPULATED NO NEED TO RUN THIS CODE IT IS JUST HERE FOR SHOW

# for line in english_lines:
#     r = random.random()
#     if r < 0.9:
#         english_training_file.write(line)
#     else:
#         english_testing_file.write(line)

# for line in german_lines:
#     r = random.random()
#     if r < 0.9:
#         german_training_file.write(line)
#     else:
#         german_testing_file.write(line)

# for line in danish_lines:
#     r = random.random()
#     if r < 0.9:
#         danish_training_file.write(line)
#     else:
#         danish_testing_file.write(line)


# Create the datasets
training_dataset = []
training_target_dataset = []

testing_dataset = []
testing_target_dataset = []

# For each 5 letter word in each language: Clean the lines, convert to ord, and append target.
for line in english_training_lines:
    line = line.replace('\n', '')

    if len(line) == 5:
        training_dataset.append([ord(char) for char in line])
        training_target_dataset.append(0)

for line in german_training_lines:
    line = line.replace('\n', '')

    if len(line) == 5:
        training_dataset.append([ord(char) for char in line])
        training_target_dataset.append(1)

for line in danish_training_lines:
    line = line.replace('\n', '')

    if len(line) == 5:
        training_dataset.append([ord(char) for char in line])
        training_target_dataset.append(2)

# For the test file: Clean the lines, convert to ord, append 5 letter words
for line in english_testing_lines:
    line = line.replace('\n', '')

    if len(line) == 5:
        testing_dataset.append([ord(char) for char in line])
        testing_target_dataset.append(0)

for line in german_testing_lines:
    line = line.replace('\n', '')

    if len(line) == 5:
        testing_dataset.append([ord(char) for char in line])
        testing_target_dataset.append(1)

for line in danish_testing_lines:
    line = line.replace('\n', '')

    if len(line) == 5:
        testing_dataset.append([ord(char) for char in line])
        testing_target_dataset.append(2)


# Initialize models
knn_model = KNeighborsClassifier()
svm_model = svm.SVC()
mlp_nn = MLPClassifier()

# fit the models
knn_model.fit(training_dataset, training_target_dataset)
svm_model.fit(training_dataset, training_target_dataset)
mlp_nn.fit(training_dataset, training_target_dataset)

# Define predict result
kpred = knn_model.predict(testing_dataset)
spred = svm_model.predict(testing_dataset)
mpred = mlp_nn.predict(testing_dataset)

# Define counting variables
i = 0
j = 0
k = 0
kNumAccurate = 0
kNumInaccurate = 0
sNumAccurate = 0
sNumInaccurate = 0
mNumAccurate = 0
mNumInaccurate = 0

# Gather number of correct and incorrect predictions
for line in testing_target_dataset:
    if(kpred[i] == line):
        kNumAccurate = kNumAccurate + 1
    else:
        kNumInaccurate = kNumInaccurate + 1

    i = i + 1


for line in testing_target_dataset:
    if(spred[j] == line):
        sNumAccurate = sNumAccurate + 1
    else:
        sNumInaccurate = sNumInaccurate + 1

    j = j + 1

for line in testing_target_dataset:
    if(mpred[k] == line):
        mNumAccurate = mNumAccurate + 1
    else:
        mNumInaccurate = mNumInaccurate + 1

    k = k + 1

# Print results
print(kNumAccurate, kNumInaccurate)
print(sNumAccurate, sNumInaccurate)
print(mNumAccurate, mNumInaccurate)

# Gather the totals
kNumTotal = kNumAccurate + kNumInaccurate
sNumTotal = sNumAccurate + sNumInaccurate
mNumTotal = mNumAccurate + mNumInaccurate

# Calculate the accuracy
knnAccuracy = kNumAccurate/kNumTotal
svmAccuracy = sNumAccurate/sNumTotal
mlpAccuracy = mNumAccurate/mNumTotal

# Label text for each graph
labels = ("KNN", "SVM", "MLP")

# Numbers that you want the bars to represent
value = [knnAccuracy, svmAccuracy, mlpAccuracy]

# Title of the plot
plt.title("Model Accuracy")

# Label for the x values of the bar graph
plt.xlabel("Accuracy")

# Drawing the bar graph
y_pos = np.arange(len(labels))
plt.barh(y_pos, value, align="center", alpha=0.5)
plt.yticks(y_pos, labels)

# Display the graph
plt.show()
