from utils.device import DeviceManager
from data.loader import NMTDataLoader
from data.vocab import Vocab
from data.tokenizer import Tokenizer 
from data.dataset import TranslationDataset

def main():
    # 1. Device
    device = DeviceManager.get_device()

    # 2. Load raw data
    with open("fra-eng.txt") as f:
        raw_text = f.read()

    loader = NMTDataLoader(raw_text)
    text = loader.preprocess()
    src_sentences, tgt_sentences = loader.tokenize(text)

    # 3. Build vocabularies
    src_vocab = Vocab(
        tokens=[t for sent in src_sentences for t in sent],
        reserved_tokens=["<pad>", "<bos>", "<eos>"]
    )
    tgt_vocab = Vocab(
        tokens=[t for sent in tgt_sentences for t in sent],
        reserved_tokens=["<pad>", "<bos>", "<eos>"]
    )

    # 4. Tokenizers
    num_steps = 10
    src_tokenizer = Tokenizer(src_vocab, num_steps)
    tgt_tokenizer = Tokenizer(tgt_vocab, num_steps)

    # 5. Dataset & DataLoader
    dataset = TranslationDataset(
        src_sentences, tgt_sentences,
        src_tokenizer, tgt_tokenizer
    )
    data_iter = DataLoader(dataset, batch_size=32, shuffle=True)

    # 6. Model
    encoder = Encoder(len(src_vocab), 256, 256, 2)
    decoder = Decoder(len(tgt_vocab), 256, 256, 2)
    model = Seq2Seq(encoder, decoder)

    # 7. Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=src_vocab['<pad>'])

    trainer = Trainer(model, optimizer, loss_fn, device)

    for epoch in range(10):
        loss = trainer.train_epoch(data_iter)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # 8. Prediction + BLEU
    predictor = Seq2SeqPredictor(model, tgt_vocab, device, num_steps)

    src = src_sentences[0]
    tgt = ' '.join(tgt_sentences[0])

    pred = predictor.predict(src, src_tokenizer)
    bleu = BLEU.score(pred, tgt)

    print("Prediction:", pred)
    print("Target:", tgt)
    print("BLEU:", bleu)


if __name__ == "__main__":
    main()
