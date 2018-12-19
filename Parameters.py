class Parameters():
    def __init__(self, RANDOM_SEED=0):
        self.set_defaults()
        self.RANDOM_SEED = RANDOM_SEED

    def set_defaults(self):
        self._num = 3
        self._STATE_SIZE = 0  # state size
        self._ACTION_SIZE = 0  # action size
        self._NUM_AGENTS = 0  # number of agents
        self._RANDOM_SEED = 1  # seed for random
        self._BUFFER_SIZE = int(1e5)  # replay buffer size
        self._BATCH_SIZE = 512  # minibatch size
        self._GAMMA = 0.94  # discount factor
        self._GAMMA_MAX = 0.99  # maximum gamma
        self._GAMMA_DECAY = 1.001
        self._TAU = 1e-3  # for soft update of target parameters
        self._LR_ACTOR = 1e-4  # learning rate of the actor
        self._LR_CRITIC = 1e-3  # learning rate of the critic
        self._WEIGHT_DECAY = 0  # L2 weight decay
        self._DEVICE = 'cpu'  # L2 weight decay
        self._EPSILON = 1.0  # starting epsilon for explore/exploit
        self._EPSILON_MIN = 0.1  # starting epsilon for explore/exploit
        self._EPSILON_DECAY = .995  # epsilon decay
        self._NOISE_SIGMA = 0.2  # noise decay
        self._STEPS_BEFORE_LEARN = 15  # noise decay
        self._NUM_LEARN_STEPS = 10  # noise decay


    def __str__(self):
        print("Parameters:\n===========")
        st = "STATE_SIZE(0):" + str(self.STATE_SIZE) + "\n"
        st += "ACTION_SIZE(0):" + str(self.ACTION_SIZE) + "\n"
        st += "NUM_AGENTS(0):" + str(self.NUM_AGENTS) + "\n"
        st += "RANDOM_SEED(1):" + str(self.RANDOM_SEED) + "\n"
        st += "BUFFER_SIZE(ie5):" + str(self.BUFFER_SIZE) + "\n"
        st += "BATCH_SIZE(512):" + str(self.BATCH_SIZE) + "\n"
        st += "STEPS_BEFORE_LEARN(15) :" + str(self.STEPS_BEFORE_LEARN) + "\n"
        st += "NUM_LEARN_STEPS(10):" + str(self.NUM_LEARN_STEPS) + "\n"
        st += "GAMMA(.99):" + str(self.GAMMA) + "\n"
        st += "GAMMA_MAX(.99):" + str(self.GAMMA_MAX) + "\n"
        st += "GAMMA_DECAY(1.001):" + str(self.GAMMA_DECAY) + "\n"
        st += "TAU Size(1e-3):" + str(self.TAU) + "\n"
        st += "LR_ACTOR(ie-4):" + str(self.LR_ACTOR) + "\n"
        st += "LR_CRITIC(1e-5):" + str(self.LR_CRITIC) + "\n"
        st += "WEIGHT_DECAY(0):" + str(self.WEIGHT_DECAY) + "\n"
        st += "DEVICE(cpu):" + str(self.DEVICE) + "\n"
        st += "EPSILON(1.0):" + str(self.EPSILON) + "\n"
        st += "EPSILON_MIN(.1) :" + str(self.EPSILON_MIN) + "\n"
        st += "EPSILON_DECAY(.995) :" + str(self.EPSILON_DECAY) + "\n"
        st += "NOISE_SIGMA(0.2):" + str(self.NOISE_SIGMA) + "\n"
        return st

    @property
    def GAMMA_MAX(self):
        return self._GAMMA_MAX

    @GAMMA_MAX.setter
    def GAMMA_MAX(self, value):
        self._GAMMA_MAX = value

    @property
    def GAMMA_DECAY(self):
        return self._GAMMA_DECAY

    @GAMMA_DECAY.setter
    def GAMMA_DECAY(self, value):
        self._GAMMA_DECAY = value

    @property
    def EPSILON_MIN(self):
        return self._EPSILON_MIN

    @EPSILON_MIN.setter
    def EPSILON_MIN(self, value):
        self._EPSILON_MIN = value

    @property
    def STEPS_BEFORE_LEARN(self):
        return self._STEPS_BEFORE_LEARN

    @STEPS_BEFORE_LEARN.setter
    def STEPS_BEFORE_LEARN(self, value):
        self._STEPS_BEFORE_LEARN = value

    @property
    def NUM_LEARN_STEPS(self):
        return self._NUM_LEARN_STEPS

    @NUM_LEARN_STEPS.setter
    def NUM_LEARN_STEPS(self, value):
        self._NUM_LEARN_STEPS = value


    @property
    def num(self):
        return self._num

    @num.setter
    def num(self, value):
        self._num = value

    @property
    def DEVICE(self):
        return self._DEVICE

    @DEVICE.setter
    def DEVICE(self, value):
        self._DEVICE = value

    @property
    def NUM_AGENTS(self):
        return self._NUM_AGENTS

    @NUM_AGENTS.setter
    def NUM_AGENTS(self, value):
        self._NUM_AGENTS = value

    @property
    def BUFFER_SIZE(self):
        return self._BUFFER_SIZE

    @BUFFER_SIZE.setter
    def BUFFER_SIZE(self, value):
        self._BUFFER_SIZE = value

    @property
    def BATCH_SIZE(self):
        return self._BATCH_SIZE

    @BATCH_SIZE.setter
    def BATCH_SIZE(self, value):
        self._BATCH_SIZE = value

    @property
    def STATE_SIZE(self):
        return self._STATE_SIZE

    @STATE_SIZE.setter
    def STATE_SIZE(self, value):
        self._STATE_SIZE = value

    @property
    def ACTION_SIZE(self):
        return self._ACTION_SIZE

    @ACTION_SIZE.setter
    def ACTION_SIZE(self, value):
        self._ACTION_SIZE = value

    @property
    def RANDOM_SEED(self):
        return self._RANDOM_SEED

    @RANDOM_SEED.setter
    def RANDOM_SEED(self, value):
        self._RANDOM_SEED = value

    @property
    def GAMMA(self):
        return self._GAMMA

    @GAMMA.setter
    def GAMMA(self, value):
        self._GAMMA = value

    @property
    def TAU(self):
        return self._TAU

    @TAU.setter
    def TAU(self, value):
        self._TAU = value

    @property
    def LR_ACTOR(self):
        return self._LR_ACTOR

    @LR_ACTOR.setter
    def LR_ACTOR(self, value):
        self._LR_ACTOR = value

    @property
    def LR_CRITIC(self):
        return self._LR_CRITIC

    @LR_CRITIC.setter
    def LR_CRITIC(self, value):
        self._LR_CRITIC = value

    @property
    def WEIGHT_DECAY(self):
        return self._WEIGHT_DECAY

    @WEIGHT_DECAY.setter
    def WEIGHT_DECAY(self, value):
        self._WEIGHT_DECAY = value

    @property
    def EPSILON(self):
        return self._EPSILON

    @EPSILON.setter
    def EPSILON(self, value):
        self._EPSILON = value

    @property
    def EPSILON_DECAY(self):
        return self._EPSILON_DECAY

    @EPSILON_DECAY.setter
    def EPSILON_DECAY(self, value):
        self._EPSILON_DECAY = value

    @property
    def NOISE_SIGMA(self):
        return self._NOISE_SIGMA

    @NOISE_SIGMA.setter
    def NOISE_SIGMA(self, value):
        self._NOISE_SIGMA = value
