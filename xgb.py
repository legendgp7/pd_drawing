import xgboost as xgb

from sklearn.decomposition import PCA
import dataset as ds
import numpy as np
import sys
import aug
import time

bin = np.arange(256)
if not ds.check_folder():
    ds.makeDataset(bin,type="s")

x,y =ds.readDataset(bin,type="s")

#train_images, train_labels,test_images,test_labels =ds.readDataset(bin,type="s")
order = np.random.permutation(y.shape[0])
x = x[order, :, :, :]
y = y[order, :]
div = int(0.85*y.shape[0])

train_images, x_test = x[0:div,:,:,:] / 1.0, x[div:,:,:,:] / 1.0
train_labels, y_test = y[0:div,:], y[div:,:]

x_train, y_train = aug.data_augment(train_images, train_labels,32)


x_train = x_train.reshape(x_train.shape[0],1024)
x_test = x_test.reshape(x_test.shape[0],1024)

y_train = y_train.flatten()
y_test = y_test.flatten()


"""

pca = PCA(n_components=500)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
"""

d_train = xgb.DMatrix(data=x_train, label=y_train)
d_test = xgb.DMatrix(data=x_test,label=y_test)

epoch = 500
ep = epoch

subsample = [0.5]
lr = [0.3]
depth = [6]
gamma = [0.05]
ntree = [20]
cb = [0.7]


best_acc = 0
best_para = []

for ss in subsample:
    for i in lr:
        for j in depth:
            for k in gamma:
                for m in ntree:
                    for n in cb:

                        print("\nCurrent para: %.2f, %d, %.2f, %d, %.2f" % (i, j, k, m, n))
                        watchlist = [(d_train, 'Train'), (d_test, 'Test')]

                        st = time.time()
                        clf = xgb.train({'objective': 'multi:softmax', 'eta': i, 'max_depth': j, 'gamma': k,
                                         'tree_method': 'hist',
                                         'num_parallel_tree': m, 'subsample': ss, 'num_class': 6,
                                         'colsample_bytree': n},
                                        d_train, ep, watchlist, early_stopping_rounds=6)
                        ed = time.time()

                        pred = clf.predict(d_test)
                        y_pred = np.array([round(value) for value in pred])

                        # score
                        n_test = x_test.shape[0]
                        n_correct = np.sum(y_test == y_pred)
                        acc = n_correct / n_test
                        time_used = ed - st
                        print("Test_acc: %.4f" % acc)
                        print("Time usage: %.4f" % time_used)

                        if acc > best_acc:
                            best_acc = acc
                            best_para = [i, j, k, m, n]
                            clf.save_model('xgb_research')

                        record = "ep=%d, lr=%.2f, max_depth=%d, gamma=%.2f, ntree=%d, colsample_bytree=%.2f: acc=%.4f. (Best = %.4f)\n\n" % (
                         ep, i, j, k, m, n, acc, best_acc)
                        with open('history.txt', 'a') as f:
                            f.writelines(record)