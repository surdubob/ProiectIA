from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import pyphen
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from dale_chall import DALE_CHALL
from nltk.tokenize import word_tokenize
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

nltk.download('punkt')
nltk.download('wordnet')

dtypes = {"sentence": "string", "token": "string", "complexity": "float64"}
train = pd.read_excel('train.xlsx', dtype=dtypes, keep_default_na=False)
test = pd.read_excel('test.xlsx', dtype=dtypes, keep_default_na=False)


def save_predictions(prds):
    test_id = np.arange(7663, 9001)
    np.savetxt("first_try.csv", np.stack((test_id, prds)).T, fmt="%d", delimiter=',', header="id,complex", comments="")


def nr_syllables(word):
    phen = pyphen.Pyphen(lang='en_EN')
    return len(phen.inserted(word, '-').split('-'))


def is_dale_chall(word):
    return int(word.lower() in DALE_CHALL)


def length(word):
    return len(word)


def nr_vowels(word):
    cnt = 0
    for c in word:
        if c in "aeiouAEIOU":
            cnt += 1
    return cnt


def is_title(word):
    return int(word.istitle())


def get_all_tokens(df):
    all_words = []
    global tokens
    for _, row in df.iterrows():
        tokens = word_tokenize(row['sentence'])
        for t in tokens:
            all_words.append(t.lower())

    return all_words


def word_frequency(word):
    global tokens
    return tokens.count(word.lower) / len(tokens)


def get_word_structure_features(word):
    features = []
    features.append(nr_syllables(word))
    features.append(is_dale_chall(word))
    features.append(length(word))
    features.append(nr_vowels(word))
    features.append(is_title(word))

    features.append(word_frequency(word))
    return np.array(features)


def synsets(word):
    return len(wordnet.synsets(word))


def get_wordnet_features(word):
    features = []
    features.append(synsets(word))
    return np.array(features)


def corpus_feature(corpus):
    if corpus == 'bible':
        return [0]
    elif corpus == 'biomed':
        return [1]
    return [2]


def featurize(row):
    word = row['token']
    all_features = []
    all_features.extend(corpus_feature(row['corpus']))
    all_features.extend(get_word_structure_features(word))
    all_features.extend(get_wordnet_features(word))
    return np.array(all_features)


def featurize_df(df):
    nr_of_features = len(featurize(df.iloc[0]))
    nr_of_examples = len(df)
    features = np.zeros((nr_of_examples, nr_of_features))
    for index, row in enumerate(df.iterrows()):
        row_ftrs = featurize(dict(row[1]))
        features[index, :] = row_ftrs
    return features


def kfold_cross_validation():
    fold_number = 10

    for nb in [7]:
        model = KNeighborsClassifier(n_neighbors=nb)
        # model = GaussianNB()
        # model = svm.SVC()

        acc_scores = []
        kf = KFold(fold_number, shuffle=True)
        for train_index, test_index in kf.split(train):
            train_data_x = train.iloc[train_index]

            X_train = featurize_df(train_data_x)
            X_test = featurize_df(train.iloc[test_index])

            y_train = train.iloc[train_index].loc[:, 'complex']
            y_test = train.loc[test_index]['complex']

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = balanced_accuracy_score(preds, y_test)
            acc_scores.append(acc)

        avg_acc_score = sum(acc_scores) / fold_number

        print('k= {} - accuracy of each fold - {}'.format(nb, acc_scores))
        print('Avg accuracy : {}'.format(avg_acc_score))


if __name__ == '__main__':
    print('train data: ', train.shape)
    print('test data: ', test.shape)

    tokens = get_all_tokens(train)

    kfold_cross_validation()

    # X_train = featurize_df(train)
    # X_test = featurize_df(test)
    #
    # y_train = train.loc[:, 'complex']
    #
    # model = KNeighborsClassifier(n_neighbors=5)
    #
    # model.fit(X_train, y_train)
    # preds = model.predict(X_test)
    #
    # save_predictions(preds)
