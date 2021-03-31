# squiRL
An RL library in PyTorch embedded within the PyTorch Lightning framework. Aiming to provide a comprehensive platform for the development and testing of RL algorithms. 

## Performance checker
DRL research is painful. Writing DRL code is even more so. Throughout development, this repos is bound to go through many changes and some of those changes may break the performance of older code.

To ensure major pull requests don't have undesirable conequences, and to build a comprehensive zoo of algorithms and envs, we introduce the `performance checker` feature. This is a Github workflow automatically triggered on a pull request if labelled `check_performance`.

The workflow runs all experiments specified in the `configs` folder (5 random seeds each). It then compares the average `mean_episode_reward` of the 5 seeds against the respective `env` thresholds specified in `performance_thresh.json`.

For example `configs/cartpole_ppo.json` has the experiment configurations to run `PPO` on Gym's `CartPole-v0`. The workflow runs 5 random seeds. Getting a mean reward larger than `150` means the env is solved. This value, `150` is saved in `performance_thresh.json` under the env name `CartPole-v0`. So the workflow knows that if the mean reward of the 5 seeds doesn't exceed `150`, something is wrong and an error is returned including the specific runs that failed to meet the threshold.

All runs can be found [here](https://wandb.ai/squirl/squirl). They are grouped under their respective git commits.

We ask that any new algorithm implemented be provided with a respective config file as a benchmark. Also any pull request benchmarking on any new env is more than welcome.

## Branch names
Branches should be using one of these groups to start with:
wip - Works in progress; stuff I know won't be finished soon (like a release)
feat - Feature I'm adding or expanding
bug - Bug fix or experiment
junk - Throwaway branch created to experiment

Groups should be split using "-". For example: junk-id-test

Skip the ID for now, since we don't have unique id generation in trello. Please add a label with the branch name to the card.

## Commit messages
Commit your work as often as possible. Push the changes in batches.
Each commit should have one line for each feature/change added.

Example of commit:
File x.py added 
Gradient clipping fixed  
Feature Y implemented  

## Pull Requests
When finished with your work, create a pull request between the relevant branches. This would be discussed in our next meeting. Please add a label to trello to mark cards in need of review.

## Unit Testing
Any script you add in the tests directory with the name test_….py like the script already there: `test_MLP_output_shape.py` will run automatically when you merge to master. Or when you create a pull requests.  
So just come up with a test, add it to the tests folder and voila.  
When you are developing on the command line and you want to run the tests locally, go to the tests directory and run `pytest` on the command line. This will run all the tests in the directory.  
You would need to install the `pytest` module from pip first of course.

## Cite
To cite this repository in publications:

    @misc{squiRL,
      author = {Khalil, Ahmed and Anca, Mihai and Thomas, Jonathan},
      title = {squiRL},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/squiReinforcementLearning/squiRL}},
    }
