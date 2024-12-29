from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self,eps=0.04):

        self.eps = eps
        # reward model
        self.Q = 0.1
        self.R1 = 20000.0
        self.R2 = 20000.0
        self.S = 1000.0

        # patient parameters
        self.k1 = 8e-7  # infection rate (mL per virions and per day)
        self.k2 = 1e-4  # infection rate (mL per virions and per day)
        self.f = 0.34  # treatment efficacy reduction for type 2 cells
        # cell type 1
        self.lambda1 = 1e4  # production rate (cells per mL and per day)
        self.d1 = 1e-2  # death rate (per day)
        self.m1 = 1e-5  # immune-induced clearance rate (mL per cells and per day)
        self.rho1 = 1  # nb virions infecting a cell (virions per cell)
        # cell type 2
        self.lambda2 = 31.98  # production rate (cells per mL and per day)
        self.d2 = 1e-2  # death rate (per day)
        self.m2 = 1e-5  # immune-induced clearance rate (mL per cells and per day)
        self.rho2 = 1  # nb virions infecting a cell (virions per cell)
        # infected cells
        self.delta = 0.7  # death rate (per day)
        self.NT = 100  # virions produced (virions per cell)
        self.c = 13  # virus natural death rate (per day)
        # immune response (immune effector cells)
        self.lambdaE = 1  # production rate (cells per mL and per day)
        self.bE = 0.3  # maximum birth rate (per day)
        self.Kb = 100  # saturation constant for birth (cells per mL)
        self.dE = 0.25  # maximum death rate (per day)
        self.Kd = 500  # saturation constant for death (cells per mL)
        self.deltaE = 0.1  # natural death rate (per day)
        self.action_set = [
            np.array(pair) for pair in [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]
        ]
        self.loss = []

    def act(self, observation, use_random=False):
        estimates_s = [self.transition(observation,a,5) for a in self.action_set]
        rewards  = [self.reward(estimates_s[i],self.action_set[i]) for i in range(4)]
        
        if np.random.random()<self.eps:
            selected_action = np.random.choice(np.arange(4))
        else:
            selected_action = np.argmax(rewards)
        #print(f'rewards {rewards}')
        #print(selected_action)
        #self.loss.append(observation) 
        return selected_action

    def save(self, path):
        pass

    def load(self):
        pass
    
    def reward(self, state, action):
        rew = -(
            self.Q * state[4]
            + self.R1 * action[0] ** 2
            + self.R2 * action[1] ** 2
            - self.S * state[5]
        )
        return rew
    
    def der(self, state, action):
        T1 = state[0]
        T1star = state[1]
        T2 = state[2]
        T2star = state[3]
        V = state[4]
        E = state[5]

        eps1 = action[0]
        eps2 = action[1]

        T1dot = self.lambda1 - self.d1 * T1 - self.k1 * (1 - eps1) * V * T1

        T1stardot = (
            self.k1 * (1 - eps1) * V * T1 - self.delta * T1star - self.m1 * E * T1star
        )
        T2dot = self.lambda2 - self.d2 * T2 - self.k2 * (1 - self.f * eps1) * V * T2
        T2stardot = (
            self.k2 * (1 - self.f * eps1) * V * T2
            - self.delta * T2star
            - self.m2 * E * T2star
        )
        Vdot = (
            self.NT * self.delta * (1 - eps2) * (T1star + T2star)
            - self.c * V
            - (
                self.rho1 * self.k1 * (1 - eps1) * T1
                + self.rho2 * self.k2 * (1 - self.f * eps1) * T2
            )
            * V
        )
        Edot = (
            self.lambdaE
            + self.bE * (T1star + T2star) * E / (T1star + T2star + self.Kb)
            - self.dE * (T1star + T2star) * E / (T1star + T2star + self.Kd)
            - self.deltaE * E
        )
        return np.array([T1dot, T1stardot, T2dot, T2stardot, Vdot, Edot])
    
    def transition(self, state, action, duration):
        """duration should be a multiple of 1e-3"""
        state0 = np.copy(state)
        state0_orig = np.copy(state)
        nb_steps = int(duration // 1e-3)
        for i in range(nb_steps):
            der = self.der(state0, action)
            state1 = state0 + der * 1e-3

            # np.clip(state1, self.lower, self.upper, out=state1)
            state0 = state1
        return state1

