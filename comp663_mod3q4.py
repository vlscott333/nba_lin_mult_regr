'''Write a program that will do the following tasks:

The nbaallelo_slr.csv database contains information on 126315 NBA games
between 1947 and 2015. The columns report the points made by one team,
the Elo rating of that team coming into the game, the Elo rating of the
team after the game, and the points made by the opposing team.

Load the data set into a data frame.
Use the ols function to perform a multiple linear regression with pts as
the response variable and opp_pts and elo_i as the predictor variables.
Create an analysis of variance table using the results of the multiple regression.'''

#response variable = pts
#predictor variable 1 = opp_pts
#predictor variable 2 = elo_i

# Import the necessary modules
import pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.formula.api import ols
nba = pd.read_csv('nbaallelo_slr.csv')# Code to read in nbaallelo_slr.csv

nba_df = pd.DataFrame(nba)
#print(nba_df.head())

nba_multdf = nba_df[['game_id','pts', 'opp_pts', 'elo_i']]
#print(nba_multdf.head())

print(nba_df.describe().T[['min', 'max', 'mean']])

# Perform multiple linear regression on pts, elo_i, and opp_pts
# Code to perform multiple regression using statsmodels ols
results = smf.ols(f'pts ~ elo_i + opp_pts', nba_multdf).fit()
print(results.summary())

# Create an analysis of variance table
aov_table = (sm.stats.anova_lm(results, type =2 ))

# Print the analysis of variance table
print(aov_table)