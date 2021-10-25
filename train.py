import argparse
import os
import pickle
import torch
import numpy as np
import random
import time
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import PolynomialLanguage, train_test_split
from model import Seq2Seq, sentence_to_tensor, Encoder, Decoder
from utils import Collater, SimpleDataset, get_device

device = get_device()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def pairs_to_tensors(pairs, src_lang, trg_lang):
    tensors = [
        (sentence_to_tensor(src, src_lang), sentence_to_tensor(trg, trg_lang))
        for src, trg in tqdm(pairs, desc="creating tensors")
    ]
    return tensors


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(tqdm(iterator)):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        print("TRAIN BATCH LOSS: ", loss )

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            print("EVAL BATCH LOSS: ", loss)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the model.")
    parser.add_argument("dirpath", type=str, default="models/best")
    parser.add_argument("--train_path", type=str, default="data/train_set.txt")
    parser.add_argument("--test_path", type=str, default="data/test_set.txt")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--train_val_split_ratio", type=float, default=0.95)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    train_set_pairs = PolynomialLanguage.load_pairs(args.train_path)
    test_set_pairs = PolynomialLanguage.load_pairs(args.test_path)

    src_lang, trg_lang = PolynomialLanguage.create_vocabs(train_set_pairs)
    train_pairs, val_pairs = train_test_split(
        train_set_pairs, train_test_split_ratio=args.train_val_split_ratio
    )

    train_tensors = pairs_to_tensors(train_pairs, src_lang, trg_lang)
    val_tensors = pairs_to_tensors(val_pairs, src_lang, trg_lang)

    collate_fn = Collater(src_lang, trg_lang)
    train_dataloader = DataLoader(
        SimpleDataset(train_tensors),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        SimpleDataset(val_tensors),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    save_to_pickle = {
        "src_lang.pickle": src_lang,
        "trg_lang.pickle": trg_lang,
    }

    for k, v in save_to_pickle.items():
        with open(os.path.join(args.dirpath, k), "wb") as fo:
            pickle.dump(v, fo)

    INPUT_DIM = src_lang.n_words
    OUTPUT_DIM = trg_lang.n_words
    ENC_EMB_DIM = int(INPUT_DIM * 0.25)
    DEC_EMB_DIM = int(INPUT_DIM * 0.25)
    HID_DIM = ENC_EMB_DIM * 2
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device, src_lang, trg_lang)
    model.apply(init_weights)
    best_valid_loss = float('inf')
    print("********BEGINNING TRAINING********")
    print(f'The model has {count_parameters(model):,} trainable parameters')

    N_EPOCHS = 10
    CLIP = 1
    optimizer = optim.Adam(model.parameters())
    # TRG_PAD_IDX = trg_lang.word2index[trg_lang.PAD_idx]

    criterion = nn.CrossEntropyLoss(ignore_index=trg_lang.PAD_idx)

    for epoch in range(N_EPOCHS):
        print("EPOCH: ", epoch)

        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_dataloader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), 'best-expander.pt')
            torch.save(model, 'best-expander.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
