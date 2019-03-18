import json

a = {"boxi": 1}
data = {"a":a}

with open('data.json', 'w') as fp:
    json.dump(data, fp)

open("adfasd")