import wandb
import json
import os
import numpy as np

with open("performance_thresh.json", 'rt') as f:
    thresh = json.load(f)
    print("Thresholds on file:")
    print(thresh)

api = wandb.Api()
failures = {}
alg_means = {}
data = {}
for model in os.listdir("models"):
    config_file = "models/" + model + "/" + model + "_init.json"
    data[model] = {}
    with open(config_file, 'rt') as f:
        data[model] = json.load(f)
        algorithm = data[model]['algorithm']
        if not data[model]['algorithm'] in alg_means:
            alg_means[algorithm] = {}
            failures[algorithm] = {}
        failures[algorithm][model] = {}
        run = api.run("squirl/squirl/" + model)
        wandb_mean_rewards = run.history(keys=['mean_episode_reward'],
                                         pandas=False)
        mean_reward = np.mean(
            [i['mean_episode_reward'] for i in wandb_mean_rewards][-100:])
        print(model, mean_reward)
        alg_means[data[model]['algorithm']][model] = mean_reward
        if mean_reward < thresh[data[model]['env']]:
            failures[algorithm][model]["env"] = data[model]['env']
            failures[algorithm][model]["threshold"] = thresh[data[model]
                                                             ['env']]
            failures[algorithm][model]["mean_last_100_steps"] = mean_reward

alg_failures = {}
for k, v in alg_means.items():
    means = []
    for nv in v.values():
        means.append(nv)
    alg_mean = np.mean(means)
    print(alg_mean)
    if alg_mean < thresh[data[model]['env']]:
        alg_failures[k] = alg_mean

assert not bool(
    alg_failures) == True, "The following algorithms have failed:\n" + str(
        alg_failures.keys(
        )) + "\nHere are all failed runs of each algorithm:\n" + str(failures)
