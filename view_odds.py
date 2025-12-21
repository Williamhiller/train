import json

with open('train-data/odds/2017-2018_odds_features.json', 'r') as f:
    data = json.load(f)

matches = data.get('matches', [])
if matches:
    print('赔率数据结构:')
    print(json.dumps(matches[0], indent=2))
