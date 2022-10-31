DATASET_PATH = 'data/original/'
BASE_PATH = 'data/processed/'
PLOTS_PATH = BASE_PATH + '/{0}/{1}/{2}'

EXTRACTED_DATA_FILE = 'extracted_dataset.csv'
SELECTED_FEATURES_FILE = 'selected_features.json'
WINDOWED_ATTACK_DATA_FILE = 'windowed_dataset.csv'
WINDOWED_DATA_FILE = 'windowed_dataset_no_attacks.csv'

UNUSED_FEATURES = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', \
                   'service', 'Stime', 'Ltime', 'attack_cat', 'dur', 'ct_ftp_cmd']

FULL_SELECTED_FEATURES_FILE = BASE_PATH + '{0}/selected_features.json'
FULL_EXTRACTED_DATA_FILE = BASE_PATH + '{0}/extracted_dataset.csv'

FULL_WINDOWED_ATTACK_DATA_FILE = BASE_PATH + '{0}/{1}/windowed_dataset.csv'
FULL_WINDOWED_DATA_FILE = BASE_PATH + '{0}/{1}/windowed_dataset_no_attacks.csv'

CORELLATIONS_FILE_PATH = BASE_PATH + '{0}/correlations/'
CORELLATIONS_PNG_FILE = CORELLATIONS_FILE_PATH + '{1}_matrix.png'

EXTRACTED_DATASET_PATH = BASE_PATH + '{0}/extracted_dataset.csv'

PROTOCOLS = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'smtp', 'ssh']  # pop3 was removed(no attacks)

MODEL_PATH = 'models/model_{0}/'
MODEL_PREDICTIONS_PATH = 'models/model_{0}/predictions'
SAVE_MODEL_PATH = 'models/model_{0}/model.h5'