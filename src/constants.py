## BASE PATHS ##
RAW_DATASET_PATH = 'data/original/'
PROCESSED_DATASET_PATH = 'data/processed/'

## FILENAMES ##
SELECTED_FEATURES_FILENAME = 'selected_features.json'
EXTRACTED_BENIGN_DATA_FILENAME = 'extracted_benign_dataset.csv'
EXTRACTED_ATTACK_DATA_FILENAME = 'extracted_attacks_dataset.csv'
ATTACKS_FILENAME = 'attacks_dataset.csv'
BENIGN_FILENAME = 'benign_dataset.csv'

## PROCESSED DATASETS GROUPED PER PROTOCOL ##
TS_ATTACK_DATASET = PROCESSED_DATASET_PATH + '{0}/{1}/' + ATTACKS_FILENAME
TS_BENIGN_DATASET = PROCESSED_DATASET_PATH + '{0}/{1}/' + BENIGN_FILENAME

## PLOTS PER PROTOCOL FILENAME ##
BENIGN_PLOTS_PATH = PROCESSED_DATASET_PATH + '/{0}/{1}/benign_plots/'
ATTACKS_PLOTS_PATH = PROCESSED_DATASET_PATH + '/{0}/{1}/attacks_plots/'

## EXTRACTED CSV DATASET PATH ##
EXTRACTED_BENIGN_DATASET_PATH = PROCESSED_DATASET_PATH + '{0}/' + EXTRACTED_BENIGN_DATA_FILENAME
EXTRACTED_ATTACK_DATASET_PATH = PROCESSED_DATASET_PATH + '{0}/' + EXTRACTED_ATTACK_DATA_FILENAME

## TRAINED MODEL PATHS ##
MODEL_PATH = 'models/models_{0}/'
MODEL_METRICS_PATH = MODEL_PATH + 'metrics.txt'
MODEL_PREDICTIONS_BENIGN_PATH = MODEL_PATH + 'predictions_benign/'
MODEL_PREDICTIONS_ATTACK_PATH = MODEL_PATH + 'predictions_attack/'
SAVE_ANOMALY_MODEL_PATH = MODEL_PATH + 'model_anomaly_{1}.h5'
SAVE_CAT_MODEL_PATH = MODEL_PATH + 'model_cat_{1}.h5'

## OTHERS ##
UNUSED_FEATURES = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', \
                   'service', 'Stime', 'Ltime', 'attack_cat', 'dur', 'ct_ftp_cmd']
PROTOCOLS = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'smtp', 'ssh']  # pop3 was removed(no attacks)

REAL_DATASET = 'data/real/buffer_2021-04-15_2021-05-26.tsv'
REAL_DATASET_FEATURES = ['conn_count_uid_in', 'conn_count_uid_out', 'dns_count_uid_out', 'http_count_uid_in', 'ssl_count_uid_in']

CORELLATIONS_FILE_PATH = PROCESSED_DATASET_PATH + '{0}/correlations/'
CORELLATIONS_PNG_FILE = CORELLATIONS_FILE_PATH + '{1}_matrix.png'

SELECTED_FEATURES_JSON = PROCESSED_DATASET_PATH + '{0}/' + SELECTED_FEATURES_FILENAME