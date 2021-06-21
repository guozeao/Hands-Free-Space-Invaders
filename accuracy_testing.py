import numpy as np
import os, time, sys
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import tsfeatures as tf
import pandas as pd
from cache_decorator import Cache
import pickle
# import catch22



def find_shortest_distance(a,b):
    smallest_dist = None
    for i in a:
        for j in b:
            dist = abs(i - j)
            if smallest_dist == None or dist < smallest_dist[0]:
                smallest_dist = [dist, i, j]
    return (smallest_dist[1], smallest_dist[2])

def simple_classifier(data):
    pos_peaks = np.where(data == np.amax(data))
    neg_peaks = np.where(data == np.amin(data))
    pos_peak, neg_peak = find_shortest_distance(pos_peaks[0], neg_peaks[0])
    
    if pos_peak < neg_peak:
        return 0
    elif pos_peak > neg_peak:
        return 1

    return -1

def read_wave(path):
    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}")
        exit()
    else:
        samplerate, raw_data = wavfile.read(path)
        time = np.array(range(0,len(raw_data)))/samplerate
        wave = { 'y' : raw_data, 'time' : time, 'samplerate' : samplerate }
        return wave

def get_actual_labels(path, classifier):
    labels = []
    if not os.path.exists(path):
        print("ERROR: PATH NOT VALID")
        return None
    for root, dirs, files in os.walk(path, topdown=False):
        # Iterating through the directory containing training data
        for name in files:
            if classifier == "simple":
                if name.endswith("S.wav"):
                    wave_file = path + "/" + name
                    name_lst = wave_file.split("_")
                    label = name_lst[len(name_lst)-1]
                    
                    label = label.strip("-S.wav")
                    if (label =="R"):
                        label = "1"
                    else:
                        label = "0"

                    labels.append(label)
            else:
                wave_file = path + "/" + name
                name_lst = wave_file.split("_")
                label = name_lst[len(name_lst)-1]
                
                label = label.strip(".wav")
                if (label =="B"):
                    label = "2"
                elif (label == "R-S"):
                    label = "1"
                else:
                    label = "0"

                labels.append(label)

    return labels


def logistic_read_training_data(path, balance_data = False):
    """
    Reads in the training data .wav files from the specified local directory,
    generates statistics on each wave, and outputs the dataset.

    @param path     : path to local directory containing training data
    @type path      : str
    @return dataset : dataset like { waveId : [0, ...], stat1 : [s11, s12, ...], stat2 : [s21, s22, ...], ... }
    @rtype          : pandas DataFrame
    """
    data = {}
    labels = []
    if not os.path.exists(path):
        print("ERROR: PATH NOT VALID")
        return None
    
    for root, dirs, files in os.walk(path, topdown=False):
        # Iterating through the directory containing training data
        for name in files:
            print(f"READING {name}")
            if name.endswith(".wav"):
                wave_file = path + "/" + name
                # read the wave file
                wave = read_wave(wave_file)
                #print(wave)
                # TODO: Either use find_eye_movements function here to get intervals of movement for each training
                #       data .wav file OR we can use the txt files (still unsure how to interpret them though)

                ###### extract label from the filename ##########
                name_lst = wave_file.split("_")
                label = name_lst[len(name_lst)-1]
                if (label == "B.wav"):
                    label = "2" # blink

                else:
                    label = label.strip("-S.wav")
                    if (label =="R"):
                        label = "1"
                    else:
                        label = "0"

                labels.append(label)
                colnames, wave_stats = generate_statistics(wave)

                # Creating a dictionary in preparation for transformation to a dataframe
                for i, col in enumerate(colnames):
                    if not data.get(col):
                        # For the first iteration of the loop
                        data[col] = [wave_stats[i]]
                    else:
                        data[col].append(wave_stats[i])

            '''if name.endswith(".txt"):
                # TODO: Extract labels from txt files? i.e., left or right [or none?]
                labels_file = path + "/" + name
                labels = open(labels_file).read().split("\n")'''

    data["labels"] = labels
    # print(labels)
    ############ remove blinks for now ###############
    data = pd.DataFrame(data)
    #data = data[data["labels"]!="2"]
    # print(data)
    if (balance_data):
        data = balance_dataset(data)
    return data

def balance_dataset(data):
    len_r = len(data[data["labels"]=="1"])
    len_l = len(data[data["labels"]=="0"])

    smallest = len_l if len_r > len_l else len_r

    data_right = data[data["labels"]=="1"].sample(smallest, random_state=0)
    data_left = data[data["labels"]=="0"].sample(smallest, random_state=0)
    dat = pd.concat([data_right, data_left])
    return dat

def simple_prediction(path):
    labels = []
    if not os.path.exists(path):
        print("ERROR: PATH NOT VALID")
        return None
    
    for root, dirs, files in os.walk(path, topdown=False):
        # Iterating through the directory containing training data
        for name in files:
            if name.endswith("S.wav"):
                
                wave_file = path + "/" + name
                wave = read_wave(wave_file)
                label = simple_classifier(wave['y'])
                labels.append(str(label))
                
    return labels

def logistic_prediction(path, classifier, scaler):
    labels = []
    if not os.path.exists(path):
        print("ERROR: PATH NOT VALID")
        return None
    
    for root, dirs, files in os.walk(path, topdown=False):
        # Iterating through the directory containing training data
        for name in files:
            
            wave_file = path + "/" + name
            print(f"Predicting {wave_file}...")
            wave = read_wave(wave_file)
            label = predict(wave, classifier, scaler)
            print(f"Predicted: {label}")
            labels.append(str(label[0]))
                
    return labels


def build_classifier(training_path, wave_type, classifier = "LR", show_acc = False):
    # REFERENCE: https://in.springboard.com/blog/logistic-regression-explained/
    """
    Builds the specific classifier - default is Logistic Regression - and trains it on the training data.

    @param training_path : local directory containining training data, i.e., .wav files and .txt files
    @type training_path  : str
    @param classifier    : which classifier to use
    @type classifier     : str
    @param show_acc      : flag used to show classifier accuracy
    @type show_acc       : boolean
    @return classifier   : linear regression classifier
    @rtype               : sklearn.linear_model type object
    """
    dataset = logistic_read_training_data(training_path)

    ########## left and right only ################
    #dataset = dataset[dataset["labels"]!="2"]

    ######## specify the features after running lasso feature selection ##########
    #dataset = dataset[['diff2_acf10', 'stdev', 'labels']]
    dataset = dataset[['flat_spots', 'crossing_points', 'diff2_acf10', 'stdev', 'labels']]

    # Take necessary rows and split labels into new dataset, y
    X = dataset.drop(["labels"], axis=1).values
    y = list(dataset["labels"])
    # Divide into training (75%) and testing (25%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Scale the data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    ####### use this to test on the whole dataset #######
    #sc_X = StandardScaler()
    #X = sc_X.fit_transform(X)

    ##### multilinear ######
    #classifier = LogisticRegression(multi_class='multinomial', solver='saga')

    ######## binary classifier, l2 reg ##########
    classifier = LogisticRegression(multi_class= "multinomial", solver='saga', penalty = 'l2')
    #########################
    classifier.fit(X, y) #TODO: change this back to X_train, y_train
    
    if show_acc:
        determine_accuracy(classifier, X_test, y_test)
        plot_accuracy(classifier, X, y)

    # write classifier
    model_filename = f'model_{wave_type}.sav'
    pickle.dump(classifier, open(model_filename, 'wb'))
    
    # write scaler
    scaler_filename = f'scaler_{wave_type}.sav'
    pickle.dump(sc_X, open(scaler_filename, 'wb'))

    classifier.fit(X_train, y_train)

    # predict the class label
    # row = generate_statistics(wave)
    # y_hat = model.predict([row])
    # OR predict a multinomial probability distribution
    # y_hat = model.predict_proba([row])
    #classifier.predict(X_test)
    return classifier, sc_X

def determine_accuracy(model, X, y):
    """
    Runs repeated CV on the model and returns the mean and sd of the accuracy to 3 decimal places.

    @param model : Fitted logisitic regression model
    @type model  : sklearn.linear_model type object
    @param X     : All data (minus labels)
    @type X      : numpy array
    @param y     : Corresponding labels
    @type y      : numpy array
    """
    # make fake dataset (can delete later)
    # X, y = make_classification(n_samples=72, n_features=5, n_informative=5, n_redundant=0, n_classes=3, random_state=1)
    y_true = list(y)
    y_pred = model.predict(X)

    print(classification_report(y_true, y_pred))
    

    # define the model evaluation procedure
    # cv = RepeatedStratifiedKFold(n_splits=1, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    # n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report the model performance

    # print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

def plot_accuracy(model, X, y):
    """
    Runs repeated CV on the model and returns the mean and sd of the accuracy to 3 decimal places.

    @param model : Fitted logisitic regression model
    @type model  : sklearn.linear_model type object
    @param X     : All data (minus labels)
    @type X      : numpy array
    @param y     : Corresponding labels
    @type y      : numpy array
    """

    # define the model evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=1)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report the model performance

    plt.boxplot(list(n_scores))
    plt.title("4-fold CV Accuracy (10 repeats)")
    plt.ylabel("Accuracy")
    plt.xlabel("")
    plt.show()
    # print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

statistics = {"flat_spots" : tf.flat_spots,
              "crossing_points" : tf.crossing_points,
              "acf_features" : tf.acf_features,
              "mean" : np.mean,
              "stdev": np.std} # olga: see lasso_feature_selection

stat_functions = list(statistics.values())

def generate_statistics(wave):
    """
    Generates the globally defined statistics* about a given wave (* see variable 'statistics')

    @param wave        : wave object
    @type wave         : dictionary
    @return colnames   : names of generated statistics (note: has wave_stats ordering)
    @rtype colnames    : list
    @return wave_stats : generated statistics (note: has colnames ordering)
    @rtype wave_stats  : list
    """
    stats = {}
    Y = wave['y']
    i=0
    for fun in stat_functions:
        dic = fun(Y)
        #print(dic)
        try:
            for k,v in dic.items():
                stats[k] = v
        except:
            if (i == 0):
                stats["mean"] = dic
                i+=1
            else:
                stats["stdev"] = dic
                
    colnames = list(stats.keys())
    wave_stats = list(stats.values())
    return colnames, wave_stats

def predict(wave, classifier, standard_scaler):
    colnames, wave_stats = generate_statistics(wave)
    # get rid of nan values
    #wave_stats[np.isnan(wave_stats)] = 0
    wave_stats_reshape = np.reshape(wave_stats, (1,-1))
    # create a dataframe
    wave_stats_dataframe = pd.DataFrame(wave_stats_reshape, columns = colnames)
    # select specific stats
    wave_stats = wave_stats_dataframe[['flat_spots', 'crossing_points', 'diff2_acf10', 'stdev']]
    # just the values in an array
    wave_stats = np.array(wave_stats.values)
    # normalise

    wave_stats = standard_scaler.transform(wave_stats)
    # predict.
    label = classifier.predict(wave_stats)
    return label


if __name__ == "__main__":    
    
    # Globals
    
    wave_types = ["waves", "filtered"]
    classifiers = ["simple", "LR"]
    data_path = "training_data/spiker_"
    
        
    for wave_type in wave_types:
        
        # model building / loading
        
        if "LR" in classifiers:
            
             # saved binary files for equick loading
            model_filename = "model.sav"
            scaler_filename = "scaler.sav"
            
            if os.path.exists(model_filename) and os.path.exists(scaler_filename):                     
                classifier = pickle.load(open(model_filename, 'rb'))
                scaler = pickle.load(open(scaler_filename, 'rb'))
            else:
                classifier, scaler = build_classifier(f"{data_path}{wave_type}", wave_type = wave_type, show_acc=True)
                
            
        # running accuracy test             
        
        for classifier_type in classifiers:
            print(f'TESTING ACCURACY for {wave_type if wave_type == "filtered" else "unfiltered"} waves with {classifier_type} classifier...')

            actual_labels = get_actual_labels(f"{data_path}{wave_type}", classifier = classifier_type)

            if classifier_type == "simple":
                predicted_labels = simple_prediction(f"{data_path}{wave_type}")
            else:
                predicted_labels = logistic_prediction(f"{data_path}{wave_type}", classifier, scaler)
            
            # print(f"Actual labels: {actual_labels}")
            # print(f"Predicted labels: {predicted_labels}")
        
            if len(actual_labels) != len(predicted_labels):
                print("ERROR")
                exit(1)

            #confusion matrix building

            counter = 0
            left_correct = 0
            right_correct = 0
            left_wrong = 0
            right_wrong = 0
            
            for a,p in zip(actual_labels, predicted_labels):
                if a == p:
                    counter += 1
                if a == "0" and p == "0":
                    left_correct += 1
                if a == "1" and p == "1":
                    right_correct += 1
                if a == "0" and p == "1":
                    left_wrong += 1
                if a == "1" and p == "0":
                    right_wrong += 1
            
            print("   | PL | PR")
            print("=="*11)
            print(f"TL | {left_correct} | {left_wrong}")
            print(f"TR | {right_wrong} | {right_correct}")
                
            print("Accuracy: {:0.2f}%\n".format( (counter/len(actual_labels))*100) )