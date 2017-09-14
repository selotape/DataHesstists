# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def import_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    combine = [train_df, test_df]
    # g = sns.FacetGrid(train_df, col='Survived')
    # g.map(plt.hist, 'Age', bins=20)
    # sns.plt.show()
    return train_df, test_df, combine


def wrangle_data(train_df, test_df, combine):
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)  # dropping Ticket and Cabin
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    title_feature(combine)
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)  # now it safe to drop name and passenger id
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)  # change Sex to integer
    age_feature_complete(combine)  # can improve this
    port_feature_complete(train_df, combine)
    train_df = train_df.drop(['Fare', 'SibSp', 'Parch'], axis=1)  # For now i remove this but i am sure i can use it
    test_df = test_df.drop(['Fare', 'SibSp', 'Parch'], axis=1)
    combine = [train_df, test_df]
    return train_df, test_df, combine


def port_feature_complete(train_df, combine):
    freq_port = train_df.Embarked.dropna().mode()[0]
    print('most coomon port is %s' % freq_port)
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


def age_feature_complete(combine):  # notice we can improve this
    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    #for dataset in combine:  # I am sure we can improve this, it is a wrong cutiing
     #   dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
      #  dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
       # dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        #dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        #dataset.loc[dataset['Age'] > 64, 'Age']


def title_feature(combine):
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)


def main():
    print('Great Success!!')
    train_df, test_df, combine = import_data()
    train_df, test_df, combine = wrangle_data(train_df, test_df, combine)
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()  # what is this command?

    # Random Forest


    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)

    # need to do croos validation on others models as well.

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
