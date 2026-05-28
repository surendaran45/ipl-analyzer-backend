import pandas as pd

df = pd.read_csv('data/matches_clean.csv')

code_to_name = {
    'GT': 'Gujarat Titans',
    'MI': 'Mumbai Indians',
    'LSG': 'Lucknow Super Giants',
    'CSK': 'Chennai Super Kings',
    'RR': 'Rajasthan Royals',
    'RCB': 'Royal Challengers Bangalore',
    'DC': 'Delhi Capitals',
    'KKR': 'Kolkata Knight Riders',
    'PBKS': 'Punjab Kings',
    'SRH': 'Sunrisers Hyderabad',
}

df['winner_full'] = df['winner'].map(code_to_name)

def get_team_stats(team_name):
    team_matches = df[(df['team1'] == team_name) | (df['team2'] == team_name)]
    wins = df[df['winner_full'] == team_name]
    total = len(team_matches)
    win_count = len(wins)
    win_pct = round((win_count / total) * 100, 1) if total > 0 else 0
    toss_wins = team_matches[team_matches['toss_won'] == team_name]
    toss_win_pct = round((len(toss_wins) / total) * 100, 1) if total > 0 else 0
    home_scores = pd.to_numeric(team_matches['1st_inning_score'], errors='coerce').dropna()
    avg_score = round(home_scores.mean(), 1) if len(home_scores) > 0 else 0
    return {
        'team': team_name,
        'total_matches': total,
        'wins': win_count,
        'win_pct': win_pct,
        'toss_win_pct': toss_win_pct,
        'avg_first_innings_score': avg_score
    }

def get_head_to_head(team1, team2):
    h2h = df[
        ((df['team1'] == team1) & (df['team2'] == team2)) |
        ((df['team1'] == team2) & (df['team2'] == team1))
    ]
    total = len(h2h)
    t1_wins = len(h2h[h2h['winner_full'] == team1])
    t2_wins = len(h2h[h2h['winner_full'] == team2])
    return {
        'team1': team1,
        'team2': team2,
        'total_matches': total,
        'team1_wins': t1_wins,
        'team2_wins': t2_wins,
        'team1_win_pct': round((t1_wins / total) * 100, 1) if total > 0 else 0,
        'team2_win_pct': round((t2_wins / total) * 100, 1) if total > 0 else 0,
    }

def get_venue_stats(venue_name):
    venue = df[df['venue_name'].str.contains(venue_name, case=False, na=False)]
    total = len(venue)
    bat_first_wins = len(venue[venue['winner_full'] == venue['home_team']])
    chase_wins = total - bat_first_wins
    return {
        'venue': venue_name,
        'total_matches': total,
        'bat_first_wins': bat_first_wins,
        'chase_wins': chase_wins,
        'bat_first_win_pct': round((bat_first_wins / total) * 100, 1) if total > 0 else 0
    }

def get_all_teams():
    teams = pd.concat([df['team1'], df['team2']]).dropna().unique()
    return sorted([t for t in teams if t not in ['TBC', 'TBA']])

def get_all_players():
    batsmen = pd.concat([
        df['home_key_batsman'].dropna(),
        df['away_key_batsman'].dropna()
    ])
    bowlers = pd.concat([
        df['home_key_bowler'].dropna(),
        df['away_key_bowler'].dropna()
    ])
    all_players = pd.concat([batsmen, bowlers])
    # Split comma-separated names into individual players
    split_players = all_players.str.split(',').explode().str.strip()
    players = split_players.value_counts()
    return sorted(players[players >= 3].index.tolist())

def get_player_stats(player_name):
    batting = df[
        (df['home_key_batsman'].str.contains(player_name, case=False, na=False)) |
        (df['away_key_batsman'].str.contains(player_name, case=False, na=False))
    ]
    bowling = df[
        (df['home_key_bowler'].str.contains(player_name, case=False, na=False)) |
        (df['away_key_bowler'].str.contains(player_name, case=False, na=False))
    ]
    batting_wins = batting[batting['winner_full'].notna()]
    bowling_wins = bowling[bowling['winner_full'].notna()]
    return {
        'player': player_name,
        'key_batting_appearances': len(batting),
        'key_bowling_appearances': len(bowling),
        'team_won_when_key_batsman': len(batting_wins),
        'team_won_when_key_bowler': len(bowling_wins),
        'batting_win_pct': round((len(batting_wins) / len(batting)) * 100, 1) if len(batting) > 0 else 0,
        'bowling_win_pct': round((len(bowling_wins) / len(bowling)) * 100, 1) if len(bowling) > 0 else 0,
    }

def get_player_season_stats(player_name):
    results = []
    for season in sorted(df['season'].dropna().unique().astype(int)):
        season_df = df[df['season'] == season]
        batting = season_df[
            (season_df['home_key_batsman'].str.contains(player_name, case=False, na=False)) |
            (season_df['away_key_batsman'].str.contains(player_name, case=False, na=False))
        ]
        bowling = season_df[
            (season_df['home_key_bowler'].str.contains(player_name, case=False, na=False)) |
            (season_df['away_key_bowler'].str.contains(player_name, case=False, na=False))
        ]
        if len(batting) > 0 or len(bowling) > 0:
            results.append({
                'season': int(season),
                'batting_appearances': len(batting),
                'bowling_appearances': len(bowling),
            })
    return results

def get_season_stats(season):
    season_df = df[df['season'] == season]
    wins = season_df['winner_full'].value_counts().reset_index()
    wins.columns = ['team', 'wins']
    return {
        'season': season,
        'total_matches': len(season_df),
        'team_wins': wins.to_dict(orient='records')
    }

def get_all_seasons():
    return sorted(df['season'].dropna().unique().astype(int).tolist())