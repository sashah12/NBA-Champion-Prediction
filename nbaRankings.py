import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
from time import sleep
from random import randint

ranking_list_playoff = [{}] * 29
ranking_list_reg = [{}] * 29
comb_rankings = [{}] * 29
playoff_rank = [] 
playoff_team = [] 
playoff_year = np.arange(1989, 2018)
reg_team = [] 
reg_wl = []
rankcounter = 0
rankcounter2 = 0
rankcounter3 = 0

playoff_pages = np.arange(1989, 2018)


for playoff_page in playoff_pages: 
   
    playoff_page = requests.get("https://www.basketball-reference.com/playoffs/NBA_" + str(playoff_page) + ".html")
    
    playoff_soup = BeautifulSoup(playoff_page.text, 'html.parser')
   
    playoff_page_tab = playoff_soup.find_all('div', class_ = 'table_container')
    
    sleep(randint(2,3))
    
    for container in playoff_page_tab:
        for tbody in container.find_all(name = 'tbody'):
            for tr in tbody.find_all(name = 'tr'):
            
                for th in tr.find_all(name='th', attrs={'scope':'row'}):
                    playoff_rank.append(th.get_text())


                for td in tr.find_all(name = 'td', attrs={'data-stat': 'team_name'}):
                    for a in td.find_all(name = 'a'):
                        playoff_team.append(a.get_text())
                        
    ranking_list_playoff[rankcounter] = pd.DataFrame(columns = ['Team', 'Year', 'Rank'])
    playoff_rank.remove('')
    ranking_list_playoff[rankcounter]['Team'] = playoff_team
    ranking_list_playoff[rankcounter]['Rank'] = playoff_rank
    ranking_list_playoff[rankcounter]['Year'] = playoff_year[rankcounter]
    playoff_rank.clear()
    playoff_team.clear()
    
    rankcounter = rankcounter + 1

switch_1990 = ranking_list_playoff[1].iloc[1]
ranking_list_playoff[1].iloc[1] = ranking_list_playoff[1].iloc[0]
ranking_list_playoff[1].iloc[0]  = switch_1990 
ranking_list_playoff[1]['Rank'] = ranking_list_playoff[1].index + 1
switch_1991 = ranking_list_playoff[2].iloc[1]
ranking_list_playoff[2].iloc[1] = ranking_list_playoff[2].iloc[0]
ranking_list_playoff[2].iloc[0]  = switch_1991 
ranking_list_playoff[2]['Rank'] = ranking_list_playoff[2].index + 1
switch_1992 = ranking_list_playoff[3].iloc[1]
ranking_list_playoff[3].iloc[1] = ranking_list_playoff[3].iloc[0]
ranking_list_playoff[3].iloc[0]  = switch_1992 
ranking_list_playoff[3]['Rank'] = ranking_list_playoff[3].index + 1
switch_1993 = ranking_list_playoff[4].iloc[1]
ranking_list_playoff[4].iloc[1] = ranking_list_playoff[4].iloc[0]
ranking_list_playoff[4].iloc[0]  = switch_1993 
ranking_list_playoff[4]['Rank'] = ranking_list_playoff[4].index + 1
switch_1996 = ranking_list_playoff[7].iloc[1]
ranking_list_playoff[7].iloc[1] = ranking_list_playoff[7].iloc[0]
ranking_list_playoff[7].iloc[0]  = switch_1996 
ranking_list_playoff[7]['Rank'] = ranking_list_playoff[7].index + 1
switch_1997 = ranking_list_playoff[8].iloc[1]
ranking_list_playoff[8].iloc[1] = ranking_list_playoff[8].iloc[0]
ranking_list_playoff[8].iloc[0]  = switch_1997 
ranking_list_playoff[8]['Rank'] = ranking_list_playoff[8].index + 1
switch_1999 = ranking_list_playoff[10].iloc[1]
ranking_list_playoff[10].iloc[1] = ranking_list_playoff[10].iloc[0]
ranking_list_playoff[10].iloc[0]  = switch_1999 
ranking_list_playoff[10]['Rank'] = ranking_list_playoff[10].index + 1
switch_2001 = ranking_list_playoff[12].iloc[2]
ranking_list_playoff[12].iloc[2] = ranking_list_playoff[12].iloc[1]
ranking_list_playoff[12].iloc[1] = ranking_list_playoff[12].iloc[0]
ranking_list_playoff[12].iloc[0]  = switch_2001 
ranking_list_playoff[12]['Rank'] = ranking_list_playoff[12].index + 1
switch_2002 = ranking_list_playoff[13].iloc[1]
ranking_list_playoff[13].iloc[1] = ranking_list_playoff[13].iloc[0]
ranking_list_playoff[13].iloc[0]  = switch_2002 
ranking_list_playoff[13]['Rank'] = ranking_list_playoff[13].index + 1
switch_2005 = ranking_list_playoff[16].iloc[1]
ranking_list_playoff[16].iloc[1] = ranking_list_playoff[16].iloc[0]
ranking_list_playoff[16].iloc[0]  = switch_2005
ranking_list_playoff[16]['Rank'] = ranking_list_playoff[16].index + 1
switch_2006 = ranking_list_playoff[17].iloc[1]
ranking_list_playoff[17].iloc[1] = ranking_list_playoff[17].iloc[0]
ranking_list_playoff[17].iloc[0]  = switch_2006
ranking_list_playoff[17]['Rank'] = ranking_list_playoff[17].index + 1
switch_2016 = ranking_list_playoff[27].iloc[1]
ranking_list_playoff[27].iloc[1] = ranking_list_playoff[27].iloc[0]
ranking_list_playoff[27].iloc[0]  = switch_2016
ranking_list_playoff[27]['Rank'] = ranking_list_playoff[27].index + 1
switch_2017 = ranking_list_playoff[28].iloc[1]
ranking_list_playoff[28].iloc[1] = ranking_list_playoff[28].iloc[0]
ranking_list_playoff[28].iloc[0]  = switch_2017
ranking_list_playoff[28]['Rank'] = ranking_list_playoff[28].index + 1

reg_pages = np.arange(1989, 2018)

for reg_page in reg_pages: 
   
    reg_page = requests.get("https://www.basketball-reference.com/leagues/NBA_" + str(reg_page) + "_standings.html")
    
    reg_soup = BeautifulSoup(reg_page.text, 'html.parser')
   
    reg_page_tab = reg_soup.find_all(name='div', attrs={'class': 'table_container'})
    
    sleep(randint(2,6))
    
    for container in reg_page_tab:
        for tbody in container.find_all(name = 'tbody'):
            for tr in tbody.find_all(name = 'tr'):
            
                for th in tr.find_all(name='th', attrs={'scope': 'row'}):
                    reg_team.append(th.get_text())


                for td in tr.find_all(name = 'td', attrs={'data-stat': 'win_loss_pct'}):
                        reg_wl.append(td.get_text())
                            
    ranking_list_reg[rankcounter2] = pd.DataFrame(columns = ['Team', 'Winp', 'Year'])
    ranking_list_reg[rankcounter2]['Team'] = reg_team
    ranking_list_reg[rankcounter2]['Winp'] = reg_wl
    ranking_list_reg[rankcounter2]['Year'] = playoff_year[rankcounter2]
    reg_team.clear()
    reg_wl.clear()
    ranking_list_reg[rankcounter2] = ranking_list_reg[rankcounter2].sort_values(by='Winp', ascending = False)
    ranking_list_reg[rankcounter2] = ranking_list_reg[rankcounter2].reset_index(drop = True)
    ranking_list_reg[rankcounter2] = ranking_list_reg[rankcounter2].drop(['Winp'], axis = 1)
    for index, row in ranking_list_reg[rankcounter2].iterrows():
        if '*' in row['Team']:
            ranking_list_reg[rankcounter2] = ranking_list_reg[rankcounter2].drop([index])
    ranking_list_reg[rankcounter2] = ranking_list_reg[rankcounter2].rename(index={15:16})
    ranking_list_reg[rankcounter2]['Rank'] = ranking_list_reg[rankcounter2].index
    ranking_list_reg[rankcounter2]['Rank'] = ranking_list_reg[rankcounter2]['Rank'] + 1
    rankcounter2 = rankcounter2 + 1

ranking_list_reg[27] = ranking_list_reg[27].drop_duplicates(subset=['Team'], keep='first')
ranking_list_reg[28] = ranking_list_reg[28].drop_duplicates(subset=['Team'], keep='first')

while rankcounter3 < 29:
    comb_rankings[rankcounter3] = ranking_list_playoff[rankcounter3].append(ranking_list_reg[rankcounter3])
#    comb_rankings[rankcounter3] = comb_rankings[rankcounter3].iloc[: 25 , : ]
    rankcounter3 = rankcounter3 + 1
comb_rankings[27] = comb_rankings[27].reset_index(drop = True)
comb_rankings[27]['Rank'] = comb_rankings[27].index + 1
comb_rankings[28] = comb_rankings[28].reset_index(drop = True)
comb_rankings[28]['Rank'] = comb_rankings[28].index + 1

indexes = np.arange(0, 29)
ind3 = []
for x in indexes:
    if len(comb_rankings[x]['Rank'].unique()) == 25:
        pass
    else:
        ind3.append(x)
for y in ind3:
    comb_rankings[y] = comb_rankings[y].reset_index(drop = True)
    comb_rankings[y]['Rank'] = comb_rankings[y].index + 1