## FOLDERS ##
DATA_FOLDER = 'data/'
MODEL_PATH = 'models/models_{0}/'
ANOMALY_MODEL_PATH = 'anomaly_model/'
CLASSIFICATION_MODEL_PATH = 'classification_model/'
ORIGINAL_FOLDER = 'original/'
PLOTS_FOLDER = 'plots/'
PREDICTIONS_BENIGN = 'predictions_benign/'
PREDICTIONS_ATTACK = 'predictions_attack/'
PREPROCESSED_CATEGORY_FOLDER = 'preprocessed_category/'
PROCESSED_ANOMALY_FOLDER = 'processed_anomaly/'
EXTRACTED_DATASETS_FOLDER = 'extracted_datasets/'
SAVINGS_FOLDER = 'savings/'
METRICS_CLASSIFICATION_FOLDER = 'metrics_classification/'
METRICS_REGRESSION_FOLDER = 'metrics_regression/'

## CSV FILENAMES ##
ANOMALY_WINDOWS_FILENAME = 'anomaly_windows.txt'
DATASET_FILENAME = 'dataset.csv'
CAT_TEST_DATASET_FILENAME = 'test_dataset.csv'
CAT_TEST_UPSAMPLED_DATASET_FILENAME = 'test_dataset_upsampled.csv'
CAT_TRAIN_DATASET_FILENAME = 'train_dataset.csv'
CAT_TRAIN_UPSAMPLED_DATASET_FILENAME = 'train_dataset_upsampled.csv'
CONFUSION_MATRIX_FILENAME = 'confusion_matrix_window'
CONFUSION_TEST_MATRIX_FILENAME = 'confusion_matrix_test_set'
EXTRACTED_ATTACK_CAT_DATA_FILENAME = '{1}_dataset.csv'
PARTIAL_CSVS_FILENAME = 'UNSW-NB15_{0}.csv'
ROC_CURVE_FILENAME = 'roc_curve_window'
ROC_CURVE_TEST_FILENAME = 'roc_curve_test_set'
SELECTED_FEATURES_FILENAME = 'selected_features.json'
TEST_SET_FILENAME = 'test_set.txt'
UNPROCESSED_TRAINING_SET_FILENAME = 'UNSW_NB15_training-set.csv'
UNPROCESSED_TESTING_SET_FILENAME = 'UNSW_NB15_testing-set.csv'
WHOLE_DATASET_FILENAME = 'whole_dataset.csv'


## FOLDER PATHS ##
PROCESSED_PROTOCOL_FOLDER = DATA_FOLDER + PROCESSED_ANOMALY_FOLDER + '{0}/{1}/{2}/'
PROTOCOL_PLOTS_FOLDER     = PROCESSED_PROTOCOL_FOLDER + PLOTS_FOLDER
TS_ATTACK_CATEGORY_DATASET_PATH = PROCESSED_PROTOCOL_FOLDER + DATASET_FILENAME

## CSV FILE PATHS - UNPROCESSED RAW ##
UNPROCESSED_PARTIAL_CSV  = DATA_FOLDER + ORIGINAL_FOLDER + PARTIAL_CSVS_FILENAME
UNPROCESSED_TESTING_SET  = DATA_FOLDER + ORIGINAL_FOLDER + UNPROCESSED_TESTING_SET_FILENAME
UNPROCESSED_TRAINING_SET = DATA_FOLDER + ORIGINAL_FOLDER + UNPROCESSED_TRAINING_SET_FILENAME

## CSV FILE PATHS - PROCESSED RAW ##
PREPROCESSED_CAT_FOLDER = DATA_FOLDER + PREPROCESSED_CATEGORY_FOLDER
CAT_TRAIN_DATASET = PREPROCESSED_CAT_FOLDER + CAT_TRAIN_DATASET_FILENAME
CAT_TEST_DATASET = PREPROCESSED_CAT_FOLDER + CAT_TEST_DATASET_FILENAME
CAT_UPSAMPLED_TRAIN_DATASET = PREPROCESSED_CAT_FOLDER + CAT_TRAIN_UPSAMPLED_DATASET_FILENAME
CAT_UPSAMPLED_TEST_DATASET  = PREPROCESSED_CAT_FOLDER + CAT_TEST_UPSAMPLED_DATASET_FILENAME
WHOLE_DATASET = PREPROCESSED_CAT_FOLDER + WHOLE_DATASET_FILENAME

## CSV FILE PATHS - EXTRACTED DATASET PATH ##
EXTRACTED_DATASETS_FOLDER = DATA_FOLDER + EXTRACTED_DATASETS_FOLDER + '{0}/'
EXTRACTED_DATASETS_PLOTS_FOLDER = EXTRACTED_DATASETS_FOLDER + PLOTS_FOLDER + '{1}/'
EXTRACTED_ATTACK_CAT_DATASET_PATH = EXTRACTED_DATASETS_FOLDER + EXTRACTED_ATTACK_CAT_DATA_FILENAME


## TRAINED MODEL PATHS ##
WHOLE_ANOMALY_MODEL_PATH = MODEL_PATH + ANOMALY_MODEL_PATH
WHOLE_CLASSIFICATION_MODEL_PATH = MODEL_PATH + CLASSIFICATION_MODEL_PATH

SAVE_ANOMALY_MODEL_PATH = WHOLE_ANOMALY_MODEL_PATH + SAVINGS_FOLDER + '{1}/'
SAVE_CAT_MODEL_PATH = WHOLE_CLASSIFICATION_MODEL_PATH + SAVINGS_FOLDER + '{1}/'

## MODELS METRICS ##
METRICS_CLASSIFICATION_FOLDER_PATH = WHOLE_CLASSIFICATION_MODEL_PATH + METRICS_CLASSIFICATION_FOLDER
METRICS_REGRESSION_FOLDER_PATH = WHOLE_ANOMALY_MODEL_PATH + METRICS_REGRESSION_FOLDER

MODEL_REGRESSION_WINDOW_METRICS_PATH = METRICS_REGRESSION_FOLDER_PATH + ANOMALY_WINDOWS_FILENAME
MODEL_REGRESSION_TEST_METRICS_PATH = METRICS_REGRESSION_FOLDER_PATH + TEST_SET_FILENAME

MODEL_CLASSIFICATION_METRICS_WINDOW_PATH = METRICS_CLASSIFICATION_FOLDER_PATH + ANOMALY_WINDOWS_FILENAME
MODEL_CLASSIFICATION_METRICS_TEST_PATH = METRICS_CLASSIFICATION_FOLDER_PATH + TEST_SET_FILENAME
MODEL_CONF_MATRIX_PATH = METRICS_CLASSIFICATION_FOLDER_PATH + CONFUSION_MATRIX_FILENAME
MODEL_CONF_TEST_MATRIX_PATH = METRICS_CLASSIFICATION_FOLDER_PATH + CONFUSION_TEST_MATRIX_FILENAME
MODEL_METRICS_ROC_PATH = METRICS_CLASSIFICATION_FOLDER_PATH + ROC_CURVE_FILENAME
MODEL_METRICS_ROC_TEST_PATH = METRICS_CLASSIFICATION_FOLDER_PATH + ROC_CURVE_TEST_FILENAME

MODEL_PREDICTIONS_BENIGN_PATH = WHOLE_ANOMALY_MODEL_PATH + PREDICTIONS_BENIGN
MODEL_PREDICTIONS_ATTACK_PATH = WHOLE_ANOMALY_MODEL_PATH + PREDICTIONS_ATTACK


## OTHERS ##
USELESS_FEATURES_FOR_PARTIAL_CSVS = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', \
                                     'dur', 'state', 'proto', 'ct_ftp_cmd'] # ct_ftp_cmd is mainly empty
USELESS_FEATURES_FOR_CATEGORIZE = ['id', 'label', 'rate', 'dur', 'state', 'proto', 'service', 'ct_ftp_cmd']

PROTOCOLS = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'smtp', 'ssh']  # pop3 was removed(no attacks)
ATTACK_CATEGORIES = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms', 'Normal', 'All']
TO_DELETE = ['Analysis', 'Backdoor', 'Shellcode', 'Worms']

REAL_DATASET = 'data/real/buffer_2021-04-15_2021-05-26.tsv'
REAL_DATASET_FEATURES = ['conn_count_uid_in', 'conn_count_uid_out', 'dns_count_uid_out', 'http_count_uid_in', 'ssl_count_uid_in']

CORELLATIONS_FILE_PATH = DATA_FOLDER + PROCESSED_ANOMALY_FOLDER + '{0}/correlations/'
CORELLATIONS_PNG_FILE = CORELLATIONS_FILE_PATH + '{1}_matrix.png'

SELECTED_FEATURES_JSON = DATA_FOLDER + PROCESSED_ANOMALY_FOLDER + '{0}/' + SELECTED_FEATURES_FILENAME

TIME = 'time'
FULL_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TIME_FORMAT = '%m-%d %H:%M'
PRETTY_TIME_FORMAT = '%d.%m %H:%M'