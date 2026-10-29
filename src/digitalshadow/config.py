# ===== GENERAL PARAMETERS =====
DT = 1.0                    # Time interval between consevutive timestamps (seconds)
SYNTH_FS = 96000            # Target sampling frequency of synthetic tracks (Hz)
NUMBER_OF_HYDROPHONES = 5   # Number of hydrophones per floater (Values allowed: 3, 4, 5)
SL_DB = 150                 # Source level SL (dB)

# ====== BELLHOP  ======
BELLHOP_opt1 = "CVWT"       # "Option1" parameter of BELLHOP
BELLHOP_alphas = [-89,89]   # Angles of departure, degrees
BELLHOP_frequency = 1000    # Hz

# ===== AMBIENT NOISE =====
AMBIENT_NOISE = True        # Turn ON/OFF ambient noise generation
NOISE_SHIPPING = 0.5        # Shipping activity factor (real number between 0 and 1)
NOISE_WIND = 5.0            # Wind velocity (meter/second)
NOISE_EXTRA_DB =  4.0       # Extra noise to be added on tracks (dB)
NOISE_SPACING = 0.3         # TODO: check

# ===== KALMAN FILTER ======
KF_MAX_REJECTION = 10       # Maximum number of consevutive measurement rejection
KF_MAX_ACCEL = 10           # Maximum acceleration expected for the vessel between two consecutives timestamps
KF_MAX_INIT_DEVSTD = 30     # Maximum standard deviation for a point to be used to initialize the filter
KF_INIT_VELOCITY_STD = 15   # Initial velocity uncertainty
