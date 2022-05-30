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
results_dataframe = pd.DataFrame(data=[],columns=["ensemble_size","accuracy","f1_score","time_allocated_for_task"])
for k in range(20):
    print(f"Initiating run {k}")
    start_value = 1+5*k
    print(f"Ensemble size set to {start_value}, time_left = 240")
    automl_classification = autosklearn.classification.AutoSklearnClassifier(
        metric=autosklearn.metrics.f1,
        time_left_for_this_task=240,
        memory_limit=10240,
        ensemble_size=start_value,
        resampling_strategy="cv",
        resampling_strategy_arguments={'folds':15}
    )
    automl_classification.fit(X_train,y_train)
    predictions = automl_classification.predict(X_test)
    acc_ = sklearn.metrics.accuracy_score(y_test,predictions)
    f1_score_ = sklearn.metrics.f1_score(y_test,predictions,average="binary",pos_label="Present")
    results_dataframe.loc[results_dataframe.shape[0]] = [
        automl_classification.ensemble_size,
        acc_,
        f1_score_,
        240
    ]
    print(f"Resulting accuracy = {acc_} , Resulting f1_score = {f1_score_} ")
    print(results_dataframe)
    results_dataframe.to_excel("./ensemble_size_opts.xlsx",index=False)
