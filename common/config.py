class Config(object):
    DIRS           =     ['epsilon', 'boltzmann', 'noisy', 'qEntropy']
    GAMES          =     ['Boxing-v0', 'ChopperCommand-v0', 'DemonAttack-v0', 'Tennis-v0']
    ## Hyperparameters
    REPLAY_INIT    =     50000 
    REPLAY_BUFFER  =     1000000 
    EPSILON_START  =     1.0
    EPSILON_FINAL  =     0.1
    EPSILON_DECAY  =     1000000 
    TAU_START      =     10.0
    TAU_FINAL      =     0.1
    TAU_DECAY      =     150000 
    NUM_FRAMES     =     5000000
    BATCH_SIZE     =     32
    GAMMA          =     0.99
    LEARNING_RATE  =     0.00025
    Q_ENTROPY_THRESHOLD = 0.7
    MODEL_DIR      =    "model"
    VAR_DIR        =    "var"
    INITIALIZER    =    "Xavier"
