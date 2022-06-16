import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from wandb.beta.workflows import log_model

import wandb

wandb.init(
    config={"gamma": 0.1, "C": 1.0, "test_size": 0.3, "seed": 0}, project="dagster_test"
)

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=wandb.config.test_size, random_state=wandb.config.seed
)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

svm = SVC(
    kernel="rbf",
    random_state=wandb.config.seed,
    gamma=wandb.config.gamma,
    C=wandb.config.C,
)
svm.fit(X_train_std, y_train)

wandb.log(
    {
        "Train Accuracy": svm.score(X_train_std, y_train),
        "Test Accuracy": svm.score(X_test_std, y_test),
    }
)

log_model(svm, "test", project="dagster_test")
