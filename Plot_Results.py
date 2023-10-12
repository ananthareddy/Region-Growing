import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def Plot_Result():
    all_Eval = np.load('all_Eval.npy', allow_pickle=True)
    Terms = ['Accuracy','Sensitivity','Specificity','Precision','FPR','FNR','NPV','FDR','F1-Score','MCC']
    Algorithm = ['TERMS', 'CNN', 'RNN', 'LSTM', 'AutoEncoder']

    Eval_Table1 = all_Eval[3][:, 4:14]
    Table1 = PrettyTable()
    Table1.add_column(Algorithm[0], Terms)
    for j in range(4):
        Table1.add_column(Algorithm[j + 1], Eval_Table1[j])
    print('---------------------------------------- Comparisons ----------------------------------------')
    print(Table1)


    lnn = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    x = [0.35, 0.55, 0.65, 0.75, 0.85]
    for i in range(10):  # 10 evaluation value
        vn = np.zeros((4, 5))
        for j in range(len(x)):  # percentage variation
            for k in range(4):  # 5 methods
                vn[k, j] = all_Eval[j][k, 4 + i]

                if i != 9:
                    vn[k, j] = vn[k, j] * 100

        n_groups = 5
        data = vn
        plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.9
        plt.bar(index, data[0, :], bar_width,
                alpha=opacity,
                color='b',
                label='CNN [16]')

        plt.bar(index + bar_width, data[1, :], bar_width,
                alpha=opacity,
                color='g',
                label='RNN [17]')

        plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                alpha=opacity,
                color='y',
                label='LSTM [18]')

        plt.bar(index + bar_width + bar_width + bar_width, data[3, :], bar_width,
                alpha=opacity,
                color='m',
                label='Autoenoder')


        plt.ylabel(lnn[i])
        plt.xlabel('Learning Percentage')
        plt.xticks(index + bar_width,
                   ('35', '55', '65', '75', '85'))
        plt.legend(loc=4)
        plt.tight_layout()
        path1 = "./Results/alg_%s.png" % (i)
        plt.savefig(path1)
        plt.show()

