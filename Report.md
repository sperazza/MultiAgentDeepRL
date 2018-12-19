
## Continuous Control Report
##### for Udacity Deep Reinforcment Learning Nanodegree
###### Andrew R Sperazza
[trained_image]: assets/reacher_trained_fast_rnd.gif
[trainScore]: assets/trainScore.png
[trainedScore]: assets/trainedScore.png
[trainedRawScore]: assets/trainedRawScore.png
[trainingRawScore]: assets/TrainingRawScore.png
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

*Taken from : https://arxiv.org/pdf/1509.02971.pdf*

###
Internally, It consists of 4 neural networks. 1 Actor, 1 Critic, and a copy of each.  As well as a shared replay buffer.

The network is a standard MLP which utilizes batch normalization, and dropout(20%)

##
The Layer breakdown for the Neural Network Architecture is:

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


#### Key Algorithm modifications
###
###### Predictive state velocities
The idea behind the reacher, is there is a series of target balls moving.  The target balls have a location and a velocity vector.  The algorithm  was modified to accept 2 states(previous state, and current state), instead of just the current state.  This dual state information was used to capture the predictive velocity of the state components, and this was concatenated with the current state, as shown below:

```
        pred_state = state + (state - prev_state)
        xs = torch.cat((pred_state, state), dim=1)
```
###

###### Multiple training runs per agent step
There is more information in the captured experiences that can be learned by the MLP networks.  This is accomplished by capturing multiple random samples, and training the network multiple times, as shown below:


```
        if len(self.memory) > self.p.BATCH_SIZE and self.update_count>self.p.STEPS_BEFORE_LEARN:
            self.update_count=0
            for _ in range(self.p.NUM_LEARN_STEPS):
                experiences = self.memory.sample()
                self.learn(experiences, self.p.GAMMA)
```

###

###### Semi-dynamic Learning Rate
The learning rate was adjusted as a percentage during training,as well as the number of steps used for learning, as seen below:

*(comments elided)*
```
def updateLrSteps(i_episode,score_average):
    if i_episode == 20:
        p.STEPS_BEFORE_LEARN=40
        p.NUM_LEARN_STEPS=30
        agent.lr_step()
    if i_episode  == 30:
        p.STEPS_BEFORE_LEARN=50
        p.NUM_LEARN_STEPS=20
    if  i_episode == 40:
        p.STEPS_BEFORE_LEARN=80
        p.NUM_LEARN_STEPS=10
    if score_average > 30.:
        p.STEPS_BEFORE_LEARN=10
        p.NUM_LEARN_STEPS=10
```
##### Hyper-parameters

Almost all the hyperparameters were encapsulated in a Parameters object, with getters and setters.  Tuning the system manually was challenging, mainly modifying the learning steps, learning rate, epsilon decay, gamma decay, and batch size.  The parameters used for optimal training were:
```
        Parameters:
        ===========
        STATE_SIZE(0):33
        ACTION_SIZE(0):4
        NUM_AGENTS(0):20
        RANDOM_SEED(1):1
        BUFFER_SIZE(ie5):100000
        BATCH_SIZE(512):256
        STEPS_BEFORE_LEARN(15) :10
        NUM_LEARN_STEPS(10):50
        GAMMA(.99):0.94
        GAMMA_MAX(.99):0.99
        GAMMA_DECAY(1.001):1.001
        TAU Size(1e-3):0.06
        LR_ACTOR(ie-4):0.0001
        LR_CRITIC(1e-5):0.001
        WEIGHT_DECAY(0):0
        DEVICE(cpu):cuda:0
        EPSILON(1.0):0.99
        EPSILON_MIN(.1) :0.1
        EPSILON_DECAY(.995) :0.998
        NOISE_SIGMA(0.2):0.1
```


#### Results

This implementation trains fast, reaching a solution in only  **42 episodes**.

Below is average training score as well as raw training scores(with training noise)

![trainScore]
![trainingRawScore]
#####

The solution has a mean score on a trained network of **36.7** (validated on 100 episodes)


![trainedScore]
![trainedRawScore]

#####
Sample output of training steps:
```

Episode 1, Average Score: 1.77, Std Dev: 1.07, Eps: 0.99, gam: 0.94
Episode 2, Average Score: 2.20, Std Dev: 0.84, Eps: 0.99, gam: 0.94
Episode 3, Average Score: 3.27, Std Dev: 1.59, Eps: 0.98, gam: 0.94
Episode 4, Average Score: 3.99, Std Dev: 2.83, Eps: 0.98, gam: 0.94
Episode 5, Average Score: 4.76, Std Dev: 3.02, Eps: 0.98, gam: 0.94
Episode 6, Average Score: 5.47, Std Dev: 1.67, Eps: 0.98, gam: 0.95

...

Episode 40, Average Score: 29.78, Std Dev: 2.26, Eps: 0.91, gam: 0.98
Episode 41, Average Score: 29.88, Std Dev: 3.94, Eps: 0.91, gam: 0.98
Episode 42, Average Score: 30.01, Std Dev: 4.42, Eps: 0.91, gam: 0.98
Environment solved in 42 episodes!	Average Score: 30.01
Episode 43, Average Score: 30.13, Std Dev: 2.37, Eps: 0.91, gam: 0.98
Episode 44, Average Score: 30.27, Std Dev: 1.43, Eps: 0.91, gam: 0.98
Episode 45, Average Score: 30.42, Std Dev: 2.15, Eps: 0.90, gam: 0.98
Episode 46, Average Score: 30.58, Std Dev: 1.58, Eps: 0.90, gam: 0.98
...

```


#####
#### Ideas for Future Work

Here are some ideas for getting better results:
- Splitting up the velocity and position from the state space, the position is not as helpful to the neural network, and subsequently may introduce a level of noise in the network.
- Currently there is one MLP for the actor/critic, which learns from a common ReplayBuffer.  This helps speed training, however the network architecture can be improved by having either seperate networks, one for each agent, each learning independently, or having several heads into the first layer of each of the networks.  This will offer decreased dependency on a complete state space, and allow more freedom for the network to learn individual components.
- The usage of other network architectures such as
  - Trust Region Policy Optimization (TRPO)
  - and Truncated Natural Policy Gradient (TNPG)
  - Policy Optimization (PPO)
  - Distributed Distributional Deterministic Policy Gradients (D4PG)
- Utilize hyper parameter tuning
- Implement an architecture utilizing Docker Containers for both hyper-parameter tuning as well as training/capturing multiple shared experiences.

