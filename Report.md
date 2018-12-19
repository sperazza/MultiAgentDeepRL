
## Continuous Control Report
##### for Udacity Deep Reinforcment Learning Nanodegree
###### Andrew R Sperazza
[trained_image]: assets/reacher_trained_fast_rnd.gif
[trainScore]: assets/trainScore.png
[trainedScore]: assets/trainedScore.png
[trainedRawScore]: assets/trainedRawScore.png
[DDPG]: assets/DDPG.png

#

 


#### Summary
- This project solves Unity's Reacher, a multi agent environment.  This is part of a Udacity Deep Reinforcement Learning NanoDegree program.
- The DDPG Reinforcement Learning algorithm is used.
- This implementation trains fast, reaching a solution in only  **42 episodes**.

     ![trainScore]
- It has a mean score on a trained network of **36.7** (validated on 100 episodes, solved is > 30).

     ![trainedScore]


#
#### Implementation

The project is implemented in utilizing a python notebook, Continuous_Control.ipynb.
- The notebook imports classes from :
  - *Parameters.py*: A class for encapsultating parameters used throughout the algorithm
  - *CriticNN.py*: contains the Critic NN model
  - *ActorNN.py*: contains the Actor NN model
  - *ReplayBuffer.py*: contains the ReplayBuffer class, used to store experiences
  - *OrnsteinNoise.py*: contains a implementation of the Ornstein-Uhlenbeck process for generating noise


#### Learning Algorithm

This project uses the DDPG Reinforcement Learning algorithm.

![DDPG]

Internally, It consists of 4 neural networks. 1 Actor, 1 Critic, and a copy of each.  As well as a shared replay buffer.

The network is a standard MLP which utilizes batch normalization, and dropout(20%)


The Neural Network Architecture is:
```
Actor(
  (fc1): Linear(in_features=66, out_features=600, bias=True)
  (do1): Dropout(p=0.2)
  (bn1): BatchNorm1d(600, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=600, out_features=300, bias=True)
  (bn2): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc5): Linear(in_features=300, out_features=4, bias=True)
)


Critic(
  (fcs1): Linear(in_features=66, out_features=600, bias=True)
  (bn0): BatchNorm1d(600, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=604, out_features=300, bias=True)
  (do1): Dropout(p=0.2)
  (fc5): Linear(in_features=300, out_features=1, bias=True)
)
```




