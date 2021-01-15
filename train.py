import config
from dataset import BERTDataset
import pandas as pd
from model import BERTBaseUncased
from sklearn import model_selection
import tez


def train():
    dfx = pd.read_csv(
        config.TRAINING_FILE).fillna(
        "none")
    dfx.sentiment = dfx.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.3, random_state=42, stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = BERTDataset(
        review=df_train.review.values, target=df_train.sentiment.values
    )

    valid_dataset = BERTDataset(
        review=df_valid.review.values, target=df_valid.sentiment.values
    )

    n_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    model = BERTBaseUncased(num_train_steps=n_train_steps)

    es = tez.callbacks.EarlyStopping(monitor="valid_loss",
                                     model_path="model.bin")
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=config.TRAIN_BATCH_SIZE,
        device="cuda",
        epochs=config.EPOCHS,
        callbacks=[es],
        fp16=True,
    )
    model.save("model.bin")


if __name__ == '__main__':
    train()
