import transformers
import os

# docker: fetch home directory, in our docker container, its home/user_ubuntu
HOME_DIR = os.path.expanduser("~")

# maximum number of tokens in a sentence
MAX_LEN = 512

# small batch_size cause huge model
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# let's train for a maximum of 10 epochs
EPOCHS = 10

# define path for BERT model files
BERT_PATH = 'bert-base-uncased'

# path where model is saved
MODEL_PATH = os.path.join(HOME_DIR, "model", 'model.bin')

# training_file
TRAINING_FILE = os.path.join(HOME_DIR, "input", "imdb.csv")

# define the tokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
