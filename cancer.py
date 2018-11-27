# coding:utf-8
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score  # K折交叉验证模块


# T-test function
def t_test(file_dir, top_num):
    with open(file_dir)as reader:
        index_row = reader.readline().rstrip('\n').split('\t');

    index_row.remove(index_row[0])
    index_row = np.array(index_row)  # 标签

    index_pos = np.argwhere(index_row == "POS")[:, 0];
    index_neg = np.argwhere(index_row == "NEG")[:, 0];

    data = pd.read_csv(file_dir, sep='\t', index_col=0);

    index = np.array(data.index)

    data_frame = np.array(data)

    pos = data_frame[:, index_pos]
    neg = data_frame[:, index_neg]

    test = []
    i = 0
    for _pos in pos:
        t, p = st.ttest_ind(pos[i], neg[i])
        test.append((index[i], p))
        i = i + 1
    test.sort(key=lambda x: x[1])

    if top_num > 0:
        top_n_list = test[0:top_num]
    else:
        top_n_list = test[top_num:]

    top_n_index = []
    i = 0
    for temp in top_n_list:
        top_n_index.append(np.argwhere(index == top_n_list[i][0])[0][0]);
        i = i + 1

    top_n_data = data_frame[top_n_index, :]  # topn的数据
    top_n_data = np.transpose(top_n_data)

    return top_n_data, index_row


# Support Vector Machine
def svm_algorithm(data, index):
    clf = SVC()
    acc = cross_val_score(clf, data, index, cv=5, scoring='accuracy')
    return acc.mean()


# k-NearestNeighbor
def knn_algorithm(data, index):
    knn = neighbors.KNeighborsClassifier()
    acc = cross_val_score(knn, data, index, cv=5, scoring='accuracy')
    return acc.mean()


# Naive Bayes
def n_bayes_algorithm(data, index):
    gnb = GaussianNB()
    acc = cross_val_score(gnb, data, index, cv=5, scoring='accuracy')
    return acc.mean()


def get_acc(file_dir):
    acc_list = []

    for top_num in range(0, 100, 5):
        topndata, label = t_test(file_dir, top_num)

        acc_1 = svm_algorithm(topndata, label)

        acc_2 = n_bayes_algorithm(topndata, label)

        acc_3 = knn_algorithm(topndata, label)

        acc_list.append([acc_1, acc_2, acc_3])

    acc_list = np.array(acc_list)

    return acc_list


# draw line
def draw_line(indicator):
    x = list(range(0, 100, 5))

    plt.plot(x, indicator[:, 0], label='svm', linewidth=2, color='r', marker='o',
             markerfacecolor='green', markersize=4)

    plt.plot(x, indicator[:, 1], label='n-bayes', linewidth=2, color='b', marker='o',
             markerfacecolor='red', markersize=4)

    plt.plot(x, indicator[:, 2], label='knn', linewidth=2, color='g', marker='o',
             markerfacecolor='blue', markersize=4)

    plt.xlabel('Incremental Feature Selection')
    plt.ylabel('mean ACC')
    plt.title('5-Fold CrossValidation ACC chart')
    plt.legend()
    plt.show()

# case
_dir = "/home/lfq/Desktop/ALL3.txt"
acc = get_acc(_dir)
draw_line(acc)
