import numpy as np
import librosa 
import librosa.display
import pandas as pd
import soundfile as sf
import sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm
from audiomentations import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os 
import autosklearn 
import autosklearn.classification
import autosklearn.metrics 


results_frame = pd.DataFrame(data=[],columns=["n_mfcc","accuracy"])
print(results_frame)
print(f"Initiating optimization run for mfcc...\n")
X_test=np.array([])
n_mfcc_ = 0
opt_count=0
while n_mfcc_ < 260:
    opt_count += 1
    n_mfcc_ += 10
    print(f"Optimization Run {opt_count} : n_mfcc = {n_mfcc_}")
    print(f"Initiating run {opt_count}\n")
    path_opt_run = f"./1.0.1/pickled_data/opt_run_data_augmentation/mfcc_opts/{opt_count}"
    try:
        os.mkdir(path_opt_run)
    except:
        pass
    # augments
    augment = Compose([
        Reverse(p=0.8),
        PolarityInversion(p=1)

    ])

    augment_spec = SpecCompose([
        SpecFrequencyMask(p=1,min_mask_fraction=0.03,max_mask_fraction=(0.10))
    ])

    def augment_signal(S,sr,outputpath):
        augmented_signal = augment(S,sr)
        sf.write(outputpath,augmented_signal,sr)


    # for balancing
    def grab_id(amt,val_srs):
        temp_idx=[]
        temp_count=0
        for i,k in val_srs.iteritems():
            if temp_count >= amt:
                break
            else:
                temp_count+=k
                temp_idx.append(i)
        return temp_idx

    # feature extraction functions
    def mfccs_features_extract(S,sr):
        mfccs_features = librosa.feature.mfcc(y=S, sr=sr, n_mfcc= n_mfcc_)
        mfccs_features = librosa.decompose.nn_filter(mfccs_features)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis= 0)
        return mfccs_scaled_features,mfccs_features

    # paths
    data_path = "/home/francis/fypj_phys/1.0.1/"
    augment_path = "/home/francis/fypj_phys/1.0.1/training_data_balanced/"
    records = data_path+ "RECORDS"

    demographics = pd.read_csv(data_path+"training_data.csv")
    print("Class distribution by patient id")
    print(demographics["Murmur"].value_counts(normalize=True))

    # read all audio recordings and augment
    with open(records,'r') as r:
        filenames = r.readlines()

        # dataframe containing audio recording path with corresponding patient_id
        file_df = pd.DataFrame(filenames,columns=["filenames"])
        file_df["filenames"] = file_df["filenames"].str.strip() + '.wav'
        file_df["patient_id"] = file_df["filenames"].str.split("/").str[1]
        file_df["patient_id"] = file_df["patient_id"].str.split("_").str[0].astype(str)
        # merge with demographics to enrich dataframe
        demographics["Patient ID"] = demographics["Patient ID"].astype(str)
        file_df = file_df.merge(
            demographics[["Murmur","Patient ID"]],
            how='left',
            left_on='patient_id',
            right_on='Patient ID'
        )
        file_df.pop("Patient ID")
    # class distribution by audio recordings
    print("Class distribution by audio recordings"+ "\n" ,file_df["Murmur"].value_counts(normalize=True)) #Heavily biased to absent class

    # grabbing only present and absent class
    file_df_present = file_df.loc[file_df["Murmur"]=="Present"]
    file_df_absent = file_df.loc[file_df["Murmur"]=="Absent"]

    # creating test df
    test_present_ids = grab_id(300,file_df_present["patient_id"].value_counts())
    test_absent_ids = grab_id(300,file_df_absent["patient_id"].value_counts())
    test_ids = test_absent_ids+test_present_ids
    test_df = file_df.loc[file_df["patient_id"].isin(test_ids)]

    # create train df
    train_df = file_df.loc[~(file_df["patient_id"].isin(test_ids)) & (file_df["Murmur"] != "Unknown")]
    print("\nTraining set class distribution (Pre-Augment)\n",train_df["Murmur"].value_counts())
    print("\nAugmenting present class audio data\n")


    # augmenting present class audio data
    filtered_train = train_df.loc[train_df["Murmur"]!="Absent"].copy()
    for i,k in tqdm(filtered_train.iterrows(),total=filtered_train.shape[0]):
        S,sr = librosa.load(data_path+k["filenames"],sr=4000)
        augmented_signal = augment(S,sr)
        augmented_signal_filename = k["filenames"].split(".")[0]+"_a1.wav"
        augment_signal(S,sr,augment_path+augmented_signal_filename)
        train_df = train_df.append(pd.DataFrame(
            data=[[augmented_signal_filename,k["patient_id"],k["Murmur"]]],
            columns=list(train_df.columns.values)
            ))
    print("\nTraining set class distribution (Post-Augment)\n",train_df["Murmur"].value_counts())


    # extracting features and augmenting mfcc spectogram for training set
    train_df["data"] = np.array(None)
    train_df = train_df.reset_index(drop=True)
    print("\nExtracting features and augmenting present class mfcc spectogram\n")
    for i,k in tqdm(train_df.copy().iterrows(),total=train_df.copy().shape[0]):
        S,sr = librosa.load(augment_path+k["filenames"],sr=4000)
        scaled_mfccs , mfccs = mfccs_features_extract(S,sr)
        # print(list(scaled_mfccs))
        train_df.loc[i,"data"] = scaled_mfccs
        if k["Murmur"] != "Absent":
            mfccs_augment = augment_spec(mfccs)
            mfccs_augment = np.mean(mfccs_augment.T,axis=0)
            train_df = train_df.append(pd.DataFrame(
                data=[[
                    k["filenames"].replace(".wav","_augmentspect.wav"),
                    k["patient_id"],
                    k["Murmur"],
                    mfccs_augment
                ]],
                columns=list(train_df.columns.values)
            ))
    print("\nTraining set class distribution (final)\n",train_df["Murmur"].value_counts())

    # class balancing for training set
    present_count = train_df["Murmur"].value_counts()[1]
    absent_ids = grab_id(present_count,train_df.loc[train_df["Murmur"]=="Absent"]["patient_id"].value_counts())
    train_df = pd.concat([
        train_df.loc[train_df["Murmur"]=="Present"],
        train_df.loc[(train_df["Murmur"]=="Absent")&(train_df["patient_id"].isin(absent_ids))]
    ]).sample(frac=1)
    train_df = train_df.reset_index(drop=True)


    test_df["data"] = None
    test_df = test_df.reset_index(drop=True)
    for i,k in tqdm(test_df.copy().iterrows(),total=test_df.copy().shape[0]):
        S,sr = librosa.load(data_path+k["filenames"],sr=4000)
        test_scaled_mfccs, test_melspect = mfccs_features_extract(S,sr)
        test_df.loc[i,"data"] = test_scaled_mfccs
    test_df = test_df.sample(frac=1).reset_index()
    print("\nTesting set class distribution (final)\n",test_df["Murmur"].value_counts())

    # manual splitting and assigning data to variables
    X_test = np.array(test_df["data"].tolist())
    X_train = np.array(train_df["data"].tolist())
    y_test = test_df["Murmur"]
    y_train = train_df["Murmur"]

    print("Exporting prepared data to pickled files.\n")
    # export data 
    import pickle 
    pickle_out_X_train = open(f"{path_opt_run}/X_train_autosklearn.pickle","wb")
    pickle.dump(X_train,pickle_out_X_train)
    pickle_out_X_train.close()

    pickle_out_X_test = open(f"{path_opt_run}/X_test_autosklearn.pickle","wb")
    pickle.dump(X_test,pickle_out_X_test)
    pickle_out_X_test.close()


    pickle_out_y_train = open(f"{path_opt_run}/y_train_autosklearn.pickle","wb")
    pickle.dump(y_train,pickle_out_y_train)
    pickle_out_y_train.close()

    pickle_out_y_test = open(f"{path_opt_run}/y_test_autosklearn.pickle","wb")
    pickle.dump(y_test,pickle_out_y_test)
    pickle_out_y_test.close()
    print("Export complete\n")

    # load data
    DATADIR = path_opt_run
    def load_pickle_data(pickle_path):
        return pickle.load(open(pickle_path,'rb'))
    X_test = load_pickle_data(DATADIR+"/X_test_autosklearn.pickle")
    X_train = load_pickle_data(DATADIR+"/X_train_autosklearn.pickle")
    y_test = load_pickle_data(DATADIR+"/y_test_autosklearn.pickle")
    y_train = load_pickle_data(DATADIR+"/y_train_autosklearn.pickle")

    print("Data preparation complete.\n")
    print("Initiating training of model.\n.")
    automl_classification = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        memory_limit=10240,
        ensemble_size=1,
        metric=autosklearn.metrics.f1_weighted,
        resampling_strategy="cv",
        resampling_strategy_arguments={'folds':15}
    )
    automl_classification.fit(X_train,y_train) 
    print("Training of model complete.\n")
    predictions = automl_classification.predict(X_test)
    acc_ =sklearn.metrics.accuracy_score(y_test,predictions)
    results_frame.loc[results_frame.shape[0]] = [
        n_mfcc_,
        acc_
    ]
    print(results_frame)
    results_frame.to_excel("./opt_run_sheets/mfcc_opts.xlsx",index=False)
print("Optimization run complete")
