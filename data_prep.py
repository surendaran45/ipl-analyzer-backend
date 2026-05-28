import pandas as pd
import numpy as np

df = pd.read_csv('data/cricket_data.csv')

print("Shape:", df.shape)
print("Nulls in winner:", df['winner'].isna().sum())

# Extract team names from match name
df['team1'] = df['name'].str.split(' v ').str[0].str.strip()
df['team2'] = df['name'].str.split(' v ').str[1].str.strip()

# Fill season from start_date
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['season'] = df['start_date'].dt.year

# Drop rows where winner is missing
df = df.dropna(subset=['winner'])

# Clean winner column — keep only team name part
df['winner'] = df['winner'].str.strip()

# Create toss advantage column
df['toss_win_match_win'] = df['toss_won'] == df['winner']

# Create target (2nd innings score)
df['target'] = df['2nd_inning_score']
df['first_innings_score'] = df['1st_inning_score']

# Save cleaned file
df.to_csv('data/matches_clean.csv', index=False)

print("\nCleaned data saved!")
print("Total matches:", len(df))
print("Seasons:", sorted(df['season'].dropna().unique().astype(int)))
print("Sample teams:", df['team1'].value_counts().head(5))
print("Toss advantage:", round(df['toss_win_match_win'].mean() * 100, 1), "%")