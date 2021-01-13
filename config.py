import transformers

# maximum number of tokens in a sentence
MAX_LEN = 512

# small batch_size cause huge model
TRAIN_BACTCH_SIZE = 8
VALID_BATCH_SIZE = 4

# let's train for a maximum of 10 epochs
EPOCHS = 10

# define path for BERT model files
BERT_PATH = 'bert-base-uncased'

# path where model is saved
MODEL_PATH = 'model.bin'

# training_file
TRAINING_FILE = "./input/imdb.csv"

# define the tokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)