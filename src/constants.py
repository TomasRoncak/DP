## FOLDERS ##
ANOMALY_MODEL_FOLDER = 'anomaly_model/'
CLASSIFICATION_MODEL_BINARY_FOLDER = 'classification_model_bin/'
CLASSIFICATION_MODEL_MULTICLASS_FOLDER = 'classification_model_mult/'
DATA_FOLDER = 'data/'
EXTRACTED_DATASETS_PATH = 'extracted_datasets/'
METRICS_FOLDER = 'metrics/'
MODELS_FOLDER = 'models/'
MODEL_FOLDER = 'models/models_{0}/'
ORIGINAL_FOLDER = 'original/'
TEST_SET_FOLDER = 'test_set/'
TIME_WINDOW_SET_FOLDER = 'time_window/'
PLOTS_FOLDER = 'plots/'
PREDICTIONS_BENIGN_FOLDER = 'predictions_benign/'
PREDICTIONS_ATTACK_FOLDER = 'predictions_attack/'
PREPROCESSED_CATEGORY_FOLDER = 'preprocessed_category/'
PROCESSED_ANOMALY_FOLDER = 'processed_anomaly/'
SAVINGS_FOLDER = 'savings/'


## FILES ##
ANOMALY_WINDOWS_FILE = 'anomaly_windows.txt'
CAT_TEST_DATASET_FILE = 'test_dataset.csv'
CAT_TEST_UPSAMPLED_DATASET_FILE = 'test_dataset_upsampled.csv'
CAT_TRAIN_DATASET_FILE = 'train_dataset.csv'
CAT_TRAIN_UPSAMPLED_DATASET_FILE = 'train_dataset_upsampled.csv'
CONFUSION_MATRIX_FILE = 'confusion_matrix'
DATASET_FILE = 'dataset.csv'
EXTRACTED_ATTACK_CAT_DATA_FILE = '{1}_dataset.csv'
PARTIAL_CSVS_FILE = 'UNSW-NB15_{0}.csv'
RADAR_CHART_FILE = 'radar_chart_{0}'
ROC_CURVE_FILE = 'roc_curve'
SELECTED_FEATURES_FILE = 'selected_features.json'
REPORT_FILE = 'report.txt'
UNPROCESSED_TRAINING_SET_FILE = 'UNSW_NB15_training-set.csv'
UNPROCESSED_TESTING_SET_FILE = 'UNSW_NB15_testing-set.csv'
WHOLE_DATASET_PATH_FILE = 'whole_dataset.csv'


## FOLDER PATHS ##
PROCESSED_ANOMALY_PROTOCOL_PATH = DATA_FOLDER + PROCESSED_ANOMALY_FOLDER + '{0}/{1}/{2}/'
PROTOCOL_PLOTS_PATH = PROCESSED_ANOMALY_PROTOCOL_PATH + PLOTS_FOLDER
TS_DATASET_BY_CATEGORY_PATH = PROCESSED_ANOMALY_PROTOCOL_PATH + DATASET_FILE

## CSV FILE PATHS - UNPROCESSED RAW ##
UNPROCESSED_PARTIAL_CSV_PATH  = DATA_FOLDER + ORIGINAL_FOLDER + PARTIAL_CSVS_FILE
UNPROCESSED_TESTING_SET_PATH  = DATA_FOLDER + ORIGINAL_FOLDER + UNPROCESSED_TESTING_SET_FILE
UNPROCESSED_TRAINING_SET_PATH = DATA_FOLDER + ORIGINAL_FOLDER + UNPROCESSED_TRAINING_SET_FILE

## CSV FILE PATHS - PROCESSED RAW ##
PREPROCESSED_CAT_PATH = DATA_FOLDER + PREPROCESSED_CATEGORY_FOLDER
CAT_TRAIN_DATASET_PATH = PREPROCESSED_CAT_PATH + CAT_TRAIN_DATASET_FILE
CAT_TEST_DATASET_PATH = PREPROCESSED_CAT_PATH + CAT_TEST_DATASET_FILE
CAT_UPSAMPLED_TRAIN_DATASET_PATH = PREPROCESSED_CAT_PATH + CAT_TRAIN_UPSAMPLED_DATASET_FILE
CAT_UPSAMPLED_TEST_DATASET_PATH  = PREPROCESSED_CAT_PATH + CAT_TEST_UPSAMPLED_DATASET_FILE
WHOLE_DATASET_PATH = PREPROCESSED_CAT_PATH + WHOLE_DATASET_PATH_FILE

## CSV FILE PATHS - EXTRACTED DATASET PATH ##
EXTRACTED_DATASETS_PATH = DATA_FOLDER + EXTRACTED_DATASETS_PATH + '{0}/'
EXTRACTED_DATASETS_PLOTS_PATH = EXTRACTED_DATASETS_PATH + PLOTS_FOLDER + '{1}/'
EXTRACTED_ATTACK_CAT_DATASET_PATH = EXTRACTED_DATASETS_PATH + EXTRACTED_ATTACK_CAT_DATA_FILE


## TRAINED MODEL PATHS ##
WHOLE_ANOMALY_MODEL_PATH = MODEL_FOLDER + ANOMALY_MODEL_FOLDER
WHOLE_CLASSIFICATION_BINARY_MODEL_PATH = MODEL_FOLDER + CLASSIFICATION_MODEL_BINARY_FOLDER
WHOLE_CLASSIFICATION_MULTICLASS_MODEL_PATH = MODEL_FOLDER + CLASSIFICATION_MODEL_MULTICLASS_FOLDER

SAVE_ANOMALY_MODEL_PATH = WHOLE_ANOMALY_MODEL_PATH + '{1}/' + SAVINGS_FOLDER
SAVE_CAT_BINARY_MODEL_PATH = WHOLE_CLASSIFICATION_BINARY_MODEL_PATH + '{1}/' + SAVINGS_FOLDER
SAVE_CAT_MULTICLASS_MODEL_PATH = WHOLE_CLASSIFICATION_MULTICLASS_MODEL_PATH + '{1}/' + SAVINGS_FOLDER

## MODELS METRICS ##
METRICS_CLASSIFICATION_TEST_FOLDER_PATH = '{0}{1}/' + METRICS_FOLDER + TEST_SET_FOLDER
METRICS_CLASSIFICATION_WINDOW_FOLDER_PATH = '{0}{1}/' + METRICS_FOLDER + TIME_WINDOW_SET_FOLDER
METRICS_REGRESSION_FOLDER_PATH = WHOLE_ANOMALY_MODEL_PATH + '{1}/' + METRICS_FOLDER

MODEL_REGRESSION_WINDOW_METRICS_PATH = METRICS_REGRESSION_FOLDER_PATH + TIME_WINDOW_SET_FOLDER + REPORT_FILE
MODEL_REGRESSION_TEST_METRICS_PATH = METRICS_REGRESSION_FOLDER_PATH + TEST_SET_FOLDER + REPORT_FILE

MODEL_REGRESSION_RADAR_CHART_PATH = MODELS_FOLDER + RADAR_CHART_FILE

MODEL_CLASSIFICATION_METRICS_WINDOW_PATH = METRICS_CLASSIFICATION_WINDOW_FOLDER_PATH + REPORT_FILE
MODEL_CLASSIFICATION_METRICS_TEST_PATH = METRICS_CLASSIFICATION_TEST_FOLDER_PATH + REPORT_FILE
MODEL_CONF_MATRIX_PATH = METRICS_CLASSIFICATION_WINDOW_FOLDER_PATH + TIME_WINDOW_SET_FOLDER + CONFUSION_MATRIX_FILE
MODEL_CONF_TEST_MATRIX_PATH = METRICS_CLASSIFICATION_TEST_FOLDER_PATH + CONFUSION_MATRIX_FILE
MODEL_METRICS_ROC_PATH = METRICS_CLASSIFICATION_WINDOW_FOLDER_PATH + TIME_WINDOW_SET_FOLDER + ROC_CURVE_FILE
MODEL_METRICS_ROC_TEST_PATH = METRICS_CLASSIFICATION_TEST_FOLDER_PATH + ROC_CURVE_FILE

MODEL_PREDICTIONS_BENIGN_PATH = WHOLE_ANOMALY_MODEL_PATH + '{1}/' + PREDICTIONS_BENIGN_FOLDER
MODEL_PREDICTIONS_ATTACK_PATH = WHOLE_ANOMALY_MODEL_PATH + '{1}/' + PREDICTIONS_ATTACK_FOLDER


## OTHERS ##
USELESS_FEATURES_FOR_PARTIAL_CSVS = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', \
                                     'dur', 'state', 'proto', 'ct_ftp_cmd']  # ct_ftp_cmd is mainly empty
USELESS_FEATURES_FOR_CATEGORIZE = ['id', 'rate', 'dur', 'state', 'proto', 'service', 'ct_ftp_cmd']

PROTOCOLS = ['all', 'dns', 'ftp', 'ftp-data', 'http', 'smtp', 'ssh']  # pop3 was removed(no attacks)
ATTACK_CATEGORIES = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', \
                     'Reconnaissance', 'Shellcode', 'Worms', 'Normal', 'All']
TO_DELETE = ['Analysis', 'Backdoor', 'Shellcode', 'Worms']

REAL_DATASET = 'data/real/buffer_2021-04-15_2021-05-26.tsv'
REAL_DATASET_FEATURES = ['conn_count_uid_in', 'conn_count_uid_out', 'dns_count_uid_out', \
                         'http_count_uid_in', 'ssl_count_uid_in']

CORELLATIONS_FILE_PATH = DATA_FOLDER + PROCESSED_ANOMALY_FOLDER + '{0}/Correlations/'
CORELLATIONS_PNG_FILE = CORELLATIONS_FILE_PATH + '{1}_matrix.png'

SELECTED_FEATURES_JSON = DATA_FOLDER + PROCESSED_ANOMALY_FOLDER + '{0}/' + SELECTED_FEATURES_FILE

LABEL_SUM = 'label_sum'
TIME = 'time'
FULL_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TIME_FORMAT = '%m-%d %H:%M'
PRETTY_TIME_FORMAT = '%d.%m %H:%M'