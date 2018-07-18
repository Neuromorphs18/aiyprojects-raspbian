from keras.models import Sequential
from keras.layers import Dense

import numpy as np


def vec_from_labels(X, xlab, Ydb):
    Y = np.zeros((X.shape[0], Ydb.shape[1]))

    for i, lab in enumerate(xlab):
        Y[i] = Ydb[lab]

    return Y


def norm_mat(mat):
    return mat/np.linalg.norm(mat, axis=1)[:, None]


def pick_rnd_sample(X):
    rnd_idx = np.random.randint(len(X[0]))
    return X[0][rnd_idx], X[1][rnd_idx], X[2][rnd_idx]


def project_data(X, d=2000):
    proj_mat = np.random.normal(size=(X.shape[1], d))
    return np.dot(X, proj_mat)


def create_train_test(X, xlab, Y, nr_samp):
    return (X[:nr_samp], Y[:nr_samp], xlab[:nr_samp]), (
            X[nr_samp:], Y[nr_samp:], xlab[nr_samp:])


## TODO, this custom metric doesn't work
def closest_vec(y_true, y_pred):
    print(y_true)
    true_label = np.argmin(np.linalg.norm(y_true - X_wrd, axis=1))
    pred_label = np.argmin(np.linalg.norm(y_pred - X_wrd, axis=1))

    return np.int(true_label==pred_label)


path_to_embeddings = './data/cifar10_w2v_embeddings.npz' 
path_to_cifar_fts = './data/output.npz' 

# load w2v vectors and labels
wrd_fts = np.load(path_to_embeddings)['embeddings']
wrd_fts = norm_mat(wrd_fts)
wrd_lab = np.load(path_to_embeddings)['words']

# load visual vectors and labels
vis_fts = np.load(path_to_cifar_fts)['features']
vis_lab = np.argmax(np.load(path_to_cifar_fts)['labels'], axis=1)
vis_fts = norm_mat(vis_fts)


X_vis = vis_fts #project_data(vis_fts, d) # currently: Re-LU
X_wrd = vis_fts # project_data(wrd_fts, d)

# map from labels to vectors
Y_vis = vec_from_labels(X_vis, vis_lab, X_wrd)

n_train = int(0.8*len(X_vis))
train_data, test_data = create_train_test(X_vis, vis_lab, Y_vis, n_train)

dim_vis = X_vis.shape[1]
dim_wrd = Y_vis.shape[1]

model = Sequential()
n_hid = 1000

model.add(Dense(n_hid, activation='relu', input_dim=dim_vis))
model.add(Dense(dim_wrd, activation='linear'))

model.compile(optimizer='adam', # experiment with different stuff
              loss='mean_squared_error', 
              metrics=['accuracy'])  # implement custom metric

model.fit(X_vis, Y_vis, epochs=100, batch_size=24)

#score = model.evaluate(vis_fts, Y_vis)  ##use predict to evaluate
