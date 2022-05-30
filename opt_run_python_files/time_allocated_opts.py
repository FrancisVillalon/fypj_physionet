import autosklearn 
import autosklearn.classification
import autosklearn.metrics 
import sklearn
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd 

# load data
# load data
DATADIR ="./1.0.1/pickled_data/"
def load_pickle_data(pickle_path):
    return pickle.load(open(pickle_path,'rb'))
X_test = load_pickle_data(DATADIR+"X_test_autosklearn.pickle")
X_train = load_pickle_data(DATADIR+"X_train_autosklearn.pickle")
y_test = load_pickle_data(DATADIR+"y_test_autosklearn.pickle")
y_train = load_pickle_data(DATADIR+"y_train_autosklearn.pickle")

X_train[0].shape

# Loop optimization
results_dataframe = pd.DataFrame(data=[],columns=["time_allocated","accuracy","f1_score"])
start_value = 30
for k in range(10):
    print(f"Initiating run {k}")
    start_value = start_value*2
    print(f"Allocating {start_value} seconds to task.")
    automl_classification = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=start_value,
        metric=autosklearn.metrics.f1,
        memory_limit=10240,
        ensemble_size=1,
        resampling_strategy="cv",
        resampling_strategy_arguments={'folds':15}
    )
    automl_classification.fit(X_train,y_train)
    predictions = automl_classification.predict(X_test)
    acc_ = sklearn.metrics.accuracy_score(y_test,predictions)
    f1_score_ = sklearn.metrics.f1_score(y_test,predictions,average="binary",pos_label="Present")
    results_dataframe.loc[results_dataframe.shape[0]] = [
        start_value,
        acc_,
        f1_score_
    ]
    print(f"Resulting accuracy = {acc_} , Resulting f1_score = {f1_score_} ")
    print(results_dataframe)
results_dataframe.to_excel("./time_allocation_opts.xlsx",index=False)