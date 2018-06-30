class Config(object):
    DIRS           =     ['epsilon', 'boltzmann', 'noisy', 'qEntropy']
    GAMES          =     ['BoxingNoFrameskip-v0', 'ChopperCommandNoFrameskip-v0', 'DemonAttackNoFrameskip-v0', 'TennisNoFrameskip-v0', 'PongNoFrameskip-v0']
    ## Hyperparameters
    REPLAY_INIT    =     10000 
    REPLAY_BUFFER  =     100000 
    EPSILON_START  =     1.0
    EPSILON_FINAL  =     0.01
    EPSILON_DECAY  =     30000 
    TAU_START      =     10.0
    TAU_FINAL      =     0.1
    TAU_DECAY      =     30000 
    NUM_FRAMES     =     1400000
    BATCH_SIZE     =     32
    GAMMA          =     0.99
    LEARNING_RATE  =     1e-5
    Q_ENTROPY_THRESHOLD = 0.9
    MODEL_DIR      =    "model"
    VAR_DIR        =    "var"
    INITIALIZER    =    "Xavier"
