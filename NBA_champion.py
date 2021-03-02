import numpy as np
import scipy as sp
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from nbaRankings import comb_rankings

count3 = 0
while count3 < 29:
    comb_rankings[count3] = comb_rankings[count3].sort_values(by='Team', ascending = True)
    count3 = count3 + 1

for x in comb_rankings:
    for index, row in x.iterrows():
        if row['Team'] == 'Atlanta Hawks':
            x.at[index, 'Team'] = 'ATL'
        elif row['Team'] == 'Boston Celtics':
            x.at[index, 'Team'] = 'BOS'
        elif row['Team'] == 'Charlotte Hornets':
            x.at[index, 'Team'] = 'CHO'
        elif row['Team'] == 'Chicago Bulls':
            x.at[index, 'Team'] = 'CHI'
        elif row['Team'] == 'Cleveland Cavaliers':
            x.at[index, 'Team'] = 'CLE'
        elif row['Team'] == 'Dallas Mavericks':
            x.at[index, 'Team'] = 'DAL'
        elif row['Team'] == 'Denver Nuggets':
            x.at[index, 'Team'] = 'DEN'
        elif row['Team'] == 'Detroit Pistons':
            x.at[index, 'Team'] = 'DET'
        elif row['Team'] == 'Golden State Warriors':
            x.at[index, 'Team'] = 'GSW'
        elif row['Team'] == 'Houston Rockets':
            x.at[index, 'Team'] = 'HOU'
        elif row['Team'] == 'Indiana Pacers':
            x.at[index, 'Team'] = 'IND'
        elif row['Team'] == 'Los Angeles Clippers':
            x.at[index, 'Team'] = 'LAC'
        elif row['Team'] == 'Los Angeles Lakers':
            x.at[index, 'Team'] = 'LAL'
        elif row['Team'] == 'Miami Heat':
            x.at[index, 'Team'] = 'MIA'
        elif row['Team'] == 'Milwaukee Bucks':
            x.at[index, 'Team'] = 'MIL'
        elif row['Team'] == 'New Jersey Nets':
            x.at[index, 'Team'] = 'BRK'
        elif row['Team'] == 'Philadelphia 76ers':
            x.at[index, 'Team'] = 'PHI'
        elif row['Team'] == 'Phoenix Suns':
            x.at[index, 'Team'] = 'PHO'
        elif row['Team'] == 'Portland Trail Blazers':
            x.at[index, 'Team'] = 'POR'
        elif row['Team'] == 'Sacramento Kings':
            x.at[index, 'Team'] = 'SAC'
        elif row['Team'] == 'San Antonio Spurs':
            x.at[index, 'Team'] = 'SAS'
        elif row['Team'] == 'Seattle Supersonics':
            x.at[index, 'Team'] = 'OKC'
        elif row['Team'] == 'Utah Jazz':
            x.at[index, 'Team'] = 'UTA'
        elif row['Team'] == 'Washington Bullets':
            x.at[index, 'Team'] = 'WAS'
        elif row['Team'] == 'Washington Wizards':
            x.at[index, 'Team'] = 'WAS'
        elif row['Team'] == 'New Orleans Pelicans':
            x.at[index, 'Team'] = 'NOP'
        elif row['Team'] == 'New Orleans Hornets':
            x.at[index, 'Team'] = 'NOP'
        elif row['Team'] == 'Charlotte Bobcats':
            x.at[index, 'Team'] = 'CHO'
        elif row['Team'] == 'Memphis Grizzlies':
            x.at[index, 'Team'] = 'MEM'
        elif row['Team'] == 'Toronto Raptors':
            x.at[index, 'Team'] = 'TOR'
        elif row['Team'] == 'Minnesota Timberwolves':
            x.at[index, 'Team'] = 'MIN'
        elif row['Team'] == 'Brooklyn Nets':
            x.at[index, 'Team'] = 'BRK'
        elif row['Team'] == 'Vancouver Grizzlies':
            x.at[index, 'Team'] = 'MEM'
        elif row['Team'] == 'New Orleans/Oklahoma City Hornets':
            x.at[index, 'Team'] = 'NOP'
        elif row['Team'] == 'Orlando Magic':
            x.at[index, 'Team'] = 'ORL'
        elif row['Team'] == 'Oklahoma City Thunder':
            x.at[index, 'Team'] = 'OKC'
        elif row['Team'] == 'New York Knicks':
            x.at[index, 'Team'] = 'NYK'
        elif row['Team'] == 'Seattle SuperSonics':
            x.at[index, 'Team'] = 'OKC'
        else:
            pass

nba_main_old = pd.read_csv('Seasons_Stats.csv')
nba_main_old = nba_main_old[nba_main_old['Year'] >= 1989.0]
nba_main_old = nba_main_old.drop(['Unnamed: 0', 'Pos', 'TS%', '3PAr', 'FTr',
                          '2P', '2PA','2P%','eFG%', 'TRB', 'blanl',
                          'WS/48', 'blank2', 'VORP', 'TRB%', 'WS', 'BPM'], axis = 1)


# teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET',
#         'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
#         'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS',
#         'TOR', 'UTA', 'WAS']

nba_main_old['Years'] = nba_main_old['Year']
nba_main_old = nba_main_old.drop(['Year'], axis = 1)
nba_main_old = nba_main_old.rename(columns = {'Years': 'Year'})
nba_old_lst = []
old_years = []
d_old_lst = [{}] * 29
n_old_lst = [{}] * 29
trans_d_year = [None] * 29
enc_old_dfs = [None] * 29
old_frames = [None] * 29
old_nba_indices = [None] * 29
labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')
counter2 = 0

for index, row in nba_main_old.iterrows():
    if row['Tm'] == 'CHH':
        nba_main_old.at[index, 'Tm'] = 'CHO'
    elif row['Tm'] == 'NJN':
        nba_main_old.at[index, 'Tm'] = 'BRK'
    elif row['Tm'] == 'SEA':
        nba_main_old.at[index, 'Tm'] = 'OKC'
    elif row['Tm'] == 'WSB':
        nba_main_old.at[index, 'Tm'] = 'WAS'
    elif row['Tm'] == 'VAN':
        nba_main_old.at[index, 'Tm'] = 'MEM'
    elif row['Tm'] == 'NOH':
        nba_main_old.at[index, 'Tm'] = 'NOP'
    elif row['Tm'] == 'NOK':
        nba_main_old.at[index, 'Tm'] = 'NOP'
    elif row['Tm'] == 'CHA':
        nba_main_old.at[index, 'Tm'] = 'CHO'
    else:
        pass

for x in range(1989, 2018):
    nba_old_lst.append(nba_main_old[nba_main_old['Year'] == x])
    old_years.append(x)
# nba_old_lst[0].columns

# list2 = list(nba_old_lst[0]['Tm'].unique())
# list2 = sorted(list2)
# list2.remove('TOT')
# list2
#nba_old_lst[0]

for x in nba_old_lst:
    teams = list(x['Tm'].unique())
    teams = sorted(teams)
    teams.remove('TOT')
    x = x.dropna()
    x = x.drop_duplicates(subset=['Player'], keep='last')
    x.drop(x[x['Tm'] == 'TOT'].index, inplace = True)
    x = x.drop(['Player'], axis = 1)
    x = x.sort_values(by=['Year', 'Tm', 'G'], ascending = [True, True, False])
    x = x.drop(['MP'], axis = 1)
    x = x.reset_index(drop=True)
    
    for y in teams:
        x = x.drop(x[x['Tm'] == y].iloc[9: , : ].index)
        d_old_lst[counter2][y] = x[x['Tm'] == y].iloc[1: , : ]
        x = x.drop(x[x['Tm'] == y].iloc[1: , : ].index)
        d_old_lst[counter2][y] = d_old_lst[counter2][y].drop(['Year', 'Tm'], axis = 1)
        n_old_lst[counter2][y] = d_old_lst[counter2][y].to_numpy()
        n_old_lst[counter2][y] = n_old_lst[counter2][y].reshape((1, n_old_lst[counter2][y].shape[0] *
                                                                 n_old_lst[counter2][y].shape[1]))
        d_old_lst[counter2][y] = pd.DataFrame(n_old_lst[counter2][y])

    x = x.reset_index(drop=True)
    trans_d_year[counter2] = d_old_lst[counter2][teams[0]]
    for z in teams[1:]:
        trans_d_year[counter2] = trans_d_year[counter2].append(d_old_lst[counter2][z])
    trans_d_year[counter2] = trans_d_year[counter2].reset_index(drop = True)
    x = pd.concat([x, trans_d_year[counter2]], axis=1)
    old_nba_indices[counter2] = x['Tm']
    x = x.drop(['Year'], axis = 1)
    x['Year'] = old_years[counter2]
#     x['Tm'] = labelencoder.fit_transform(x['Tm'])
    
#     enc_old_dfs[counter2] = pd.DataFrame(enc.fit_transform(x[['Tm']]).toarray())
#     x = x.drop(['Tm'], axis = 1)
#     old_frames[counter2] = [enc_old_dfs[counter2], x]
#     x = pd.concat(old_frames[counter2], axis = 1)
    x['Class'] = 0
    x.columns = np.arange(1,(x.shape[1] + 1))
    x = x.rename(columns = {(x.shape[1] - 1): 'Year', (x.shape[1]) : 'Class'})
    nba_old_lst[counter2] = x 
    counter2 = counter2 + 1

nba_old_lst[20]

nba2020 = pd.read_csv('nbadata2020.csv')
nba2019 = pd.read_csv('nbadata2019.csv')
nba2018 = pd.read_csv('nbadata2018.csv')

nba_data = [nba2020, nba2019, nba2018]
dics = [{}] * 3
n_years = [{}] * 3

years = [2020, 2019, 2018] 
trans_d = [None] * 3
enc_dfs = [None] * 3
frames = [None] * 3
nba_indices = [None] * 3
counter = 0

for x in nba_data:
    for index, row in x.iterrows():
        x.at[index, 'Player'] = row['Player'][0:row['Player'].find('\\')]
    teams = list(x['Tm'].unique())
    teams = sorted(teams)
    teams.remove('TOT')
    x = x.dropna()
    x = x.drop_duplicates(subset=['Player'], keep='last')
    x.drop(x[x['Tm'] == 'TOT'].index, inplace = True)
    x = x.drop(['Player'], axis = 1)
    x = x.sort_values(by=['Tm', 'G'], ascending = [True, False])
    x = x.drop(['MP'], axis = 1)
    x = x.reset_index(drop=True)
    
    for y in teams:
        x = x.drop(x[x['Tm'] == y].iloc[9: , : ].index)
        dics[counter][y] = x[x['Tm'] == y].iloc[1: , : ]
        x = x.drop(x[x['Tm'] == y].iloc[1: , : ].index)
        dics[counter][y] = dics[counter][y].drop(['Year', 'Tm'], axis = 1)
        n_years[counter][y] = dics[counter][y].to_numpy()
        n_years[counter][y] = n_years[counter][y].reshape((1, n_years[counter][y].shape[0] * n_years[counter][y].shape[1]))
        dics[counter][y] = pd.DataFrame(n_years[counter][y])

    x = x.reset_index(drop=True)
    trans_d[counter] = dics[counter][teams[0]]
    for z in teams[1:]:
        trans_d[counter] = trans_d[counter].append(dics[counter][z])
    trans_d[counter] = trans_d[counter].reset_index(drop = True)
    x = pd.concat([x, trans_d[counter]], axis=1)
    nba_indices[counter] = x['Tm']
    x = x.drop(['Year'], axis = 1)
    x['Year'] = years[counter]
    x['Tm'] = labelencoder.fit_transform(x['Tm'])
    
    enc_dfs[counter] = pd.DataFrame(enc.fit_transform(x[['Tm']]).toarray())
    x = x.drop(['Tm'], axis = 1)
    frames[counter] = [enc_dfs[counter], x]
    x = pd.concat(frames[counter], axis = 1)
    x['Class'] = 0
    x.columns = np.arange(1,(x.shape[1] + 1))
    x = x.rename(columns = {(x.shape[1] - 1): 'Year', (x.shape[1]) : 'Class'})
    nba_data[counter] = x 
    counter = counter + 1

nba2020 = nba_data[0]
nba2019 = nba_data[1]
nba2018 = nba_data[2]
nba2020 

nba2020.at[0, 'Class'] = 27
nba2020.at[1, 'Class'] = 4
nba2020.at[2, 'Class'] = 14
nba2020.at[3, 'Class'] = 24
nba2020.at[4, 'Class'] = 22
nba2020.at[5, 'Class'] = 29
nba2020.at[6, 'Class'] = 11
nba2020.at[7, 'Class'] = 3
nba2020.at[8, 'Class'] = 26
nba2020.at[9, 'Class'] = 30
nba2020.at[10, 'Class'] = 6
nba2020.at[11, 'Class'] = 15
nba2020.at[12, 'Class'] = 5
nba2020.at[13, 'Class'] = 1
nba2020.at[14, 'Class'] = 17
nba2020.at[15, 'Class'] = 2
nba2020.at[16, 'Class'] = 8
nba2020.at[17, 'Class'] = 28
nba2020.at[18, 'Class'] = 21
nba2020.at[19, 'Class'] = 25
nba2020.at[20, 'Class'] = 10
nba2020.at[21, 'Class'] = 13
nba2020.at[22, 'Class'] = 16
nba2020.at[23, 'Class'] = 18
nba2020.at[24, 'Class'] = 12
nba2020.at[25, 'Class'] = 20
nba2020.at[26, 'Class'] = 19
nba2020.at[27, 'Class'] = 7
nba2020.at[28, 'Class'] = 9
nba2020.at[29, 'Class'] = 23
nba2020.index = nba_indices[0]

nba2019.at[0, 'Class'] = 26
nba2019.at[1, 'Class'] = 8
nba2019.at[2, 'Class'] = 11
nba2019.at[3, 'Class'] = 27
nba2019.at[4, 'Class'] = 18
nba2019.at[5, 'Class'] = 29
nba2019.at[6, 'Class'] = 24
nba2019.at[7, 'Class'] = 5
nba2019.at[8, 'Class'] = 16
nba2019.at[9, 'Class'] = 2
nba2019.at[10, 'Class'] = 7
nba2019.at[11, 'Class'] = 15
nba2019.at[12, 'Class'] = 10
nba2019.at[13, 'Class'] = 20
nba2019.at[14, 'Class'] = 22
nba2019.at[15, 'Class'] = 19
nba2019.at[16, 'Class'] = 3
nba2019.at[17, 'Class'] = 21
nba2019.at[18, 'Class'] = 23
nba2019.at[19, 'Class'] = 30
nba2019.at[20, 'Class'] = 12
nba2019.at[21, 'Class'] = 14
nba2019.at[22, 'Class'] = 6
nba2019.at[23, 'Class'] = 28
nba2019.at[24, 'Class'] = 4
nba2019.at[25, 'Class'] = 17
nba2019.at[26, 'Class'] = 9
nba2019.at[27, 'Class'] = 1
nba2019.at[28, 'Class'] = 13
nba2019.at[29, 'Class'] = 25
nba2019.index = nba2020.index

nba2018.at[0, 'Class'] = 28
nba2018.at[1, 'Class'] = 4
nba2018.at[2, 'Class'] = 23
nba2018.at[3, 'Class'] = 24
nba2018.at[4, 'Class'] = 20
nba2018.at[5, 'Class'] = 2
nba2018.at[6, 'Class'] = 26
nba2018.at[7, 'Class'] = 17
nba2018.at[8, 'Class'] = 19
nba2018.at[9, 'Class'] = 1
nba2018.at[10, 'Class'] = 3
nba2018.at[11, 'Class'] = 10
nba2018.at[12, 'Class'] = 18
nba2018.at[13, 'Class'] = 21
nba2018.at[14, 'Class'] = 29
nba2018.at[15, 'Class'] = 13
nba2018.at[16, 'Class'] = 9
nba2018.at[17, 'Class'] = 14
nba2018.at[18, 'Class'] = 8
nba2018.at[19, 'Class'] = 22
nba2018.at[20, 'Class'] = 12
nba2018.at[21, 'Class'] = 26
nba2018.at[22, 'Class'] = 6
nba2018.at[23, 'Class'] = 30
nba2018.at[24, 'Class'] = 16
nba2018.at[25, 'Class'] = 24
nba2018.at[26, 'Class'] = 15
nba2018.at[27, 'Class'] = 7
nba2018.at[28, 'Class'] = 5
nba2018.at[29, 'Class'] = 11
nba2018.index = nba2020.index


nba_main = nba2015.append(nba2016)
nba_main = nba_main.append(nba2017)
nba_main = nba_main.append(nba2018)
nba_main = nba_main.append(nba2019)
nba_main_arr = nba_main.to_numpy()
nba2020_arr = nba2020.to_numpy()

def prediction(predictions, Y_given):
    wrong = 0
    counter = 0
    for test, train in zip(predictions, Y_given):
        if test == train:
            wrong = wrong
        else:
            wrong = wrong + 1
        counter = counter + 1
    accuracy = 1 - (wrong/counter)
    return accuracy

for i in range(1):
    data_svm = shuffle(nba_main_arr)
#Data Splitting, train-test-split    
    X_train_svm = data_svm[:, 0:-1]
    Y_train_svm = data_svm[:, -1]
    X_test_svm = nba2020_arr[:, 0:-1]
    Y_test_svm = nba2020_arr[:, -1]
#Scaling training data using StandardScaler
    scaler_svm = preprocessing.StandardScaler().fit(X_train_svm)
    X_train_svm = scaler_svm.transform(X_train_svm)
#Param-grid
    parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0], 'C': [10**-7, 10**-6, 10**-5,
                                                                                                     10**-4, 10**-3, 10**-2,
                                                                                                     10**-1, 10**0, 10**1, 10**2, 10**3]},
                  {'kernel': ['poly'], 'degree': [2, 3], 'C': [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]},
                  {'kernel': ['linear'], 'C': [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]}]
#SVC class, grid search, and fit training data and training labels
    svc = svm.SVC(gamma = 'auto')
    grid_search_svm = GridSearchCV(svc, parameters, cv = 4, error_score = np.nan)
    grid_search2_svm = grid_search_svm.fit(X_train_svm, Y_train_svm)
#Printing and Storing best params into list.
    best_params_svm = grid_search2_svm.best_params_
    print("Best params: {}".format(best_params_svm))
#Best estimator is refitted so just used .predict on training data to find predicted training values.
#Used prediction pre-defined function to count accuracy on predicted training set.
    train_predictions_svm = grid_search2_svm.best_estimator_.predict(X_train_svm) #.best_estimator_
    train_accuracy_svm = prediction(train_predictions_svm, Y_train_svm)
#Printing and storing training accuracy into list.
    print("Train accuracy: {}".format(train_accuracy_svm))
#Scaling testing data with my training data scaler. This ensures training and testing data are scaled the same.
#Printing and storing test accuracy into list.
    X_test_svm = scaler_svm.transform(X_test_svm)
    test_predictions_svm = grid_search2_svm.best_estimator_.predict(X_test_svm) #.best_estimator_
    test_accuracy_svm = prediction(test_predictions_svm, Y_test_svm)
    print("Test accuracy: {}".format(test_accuracy_svm))