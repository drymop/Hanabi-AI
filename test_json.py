import json
from utils.attributedict import AttributeDict as AttDict

load_dir = 'save28_buffer'
json_file = 'configs.json'

with open(load_dir + '/' + json_file, 'r') as f:
    configs = json.load(f, object_pairs_hook=AttDict)

game_configs = configs.game_configs
model_configs = configs.model_configs
train_configs = configs.train_configs

for k, v in game_configs.items():
    print(k, end=': ')
    print(v)
print()
for k, v in model_configs.items():
    print(k, end=': ')
    print(v)
print()
for k, v in train_configs.items():
    print(k, end=': ')
    print(v)