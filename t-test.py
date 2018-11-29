import numpy as np
import scipy.stats as st
import pandas as pd


def t_test(file_name):
    # get data_matrix
    data_matrix = np.array(pd.DataFrame(pd.read_csv(file_name, sep='\t', index_col=0)))
    
    # get sample name array
    with open(file_name)as source_file:
        sample = source_file.readline().rstrip().split()
        sample.remove(sample[0])
    sample = np.array(sample)

    # get indices of pos or neg
    pos_index = np.argwhere(sample == "POS")[:, 0]
    neg_index = np.argwhere(sample == "NEG")[:, 0]

    # get data of pos or neg
    pos = data_matrix[:, pos_index]
    neg = data_matrix[:, neg_index]

    # get shape of data_matrix
    row, column = data_matrix.shape

    # put the p-value into a dictionary

    pvalue_dict = {}
    for i in range(0, row):
        t, p = st.ttest_ind(pos[i], neg[i], equal_var=True)
        pvalue_dict[i] = p

    return pvalue_dict


def get_top(dic_name, top_num):
    sorted_dic = sorted(dic_name.items(), key=lambda d: d[1])
    top_dict = {}
    for j in range(0, top_num):
        (index_num, pvalues) = sorted_dic.pop(0)
        top_dict[j] = (index_num, pvalues)

    return top_dict


source_name = "ALL3.txt"
print("Start t-test, please waiting...")
pvalue = t_test(source_name)
print("Complete t-test.")
rank = int(input("how much results do you want:"))
print("Fetching results, pleasing waiting...")
rs = get_top(pvalue, rank)
df = pd.read_csv(source_name, sep='\t', index_col=0)
index_name = np.array(df.index)

print("Top " + str(rank) + " of t-test result are following:")
print("No"+'\t'+"ft_name "+'\t'+"p-values")
for k in range(0, 10):
    (num, values) = rs[k]
    print(str(k+1)+"\t"+str(index_name[num])+'\t'+str(values))
