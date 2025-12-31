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



# import torch
# from src.data_loader import MachineTranslationDataset
# from src.modules import Seq2SeqEncoder, Seq2SeqDecoder, Seq2SeqModel
# from src.trainer import EliteTrainer
# from torch.utils.data import DataLoader

# def run_experiment():
#     # 1. Constants (Config Space)
#     DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
#     EMBED_SIZE, NUM_HIDDENS, NUM_LAYERS = 256, 256, 2
    
#     # 2. Data Pipeline
#     dataset = MachineTranslationDataset(DATA_URL, 'fra-eng/fra.txt', num_steps=10)
#     train_iter = DataLoader(dataset, batch_size=128, shuffle=True)

#     # 3. Model Assembly
#     encoder = Seq2SeqEncoder(len(dataset.src_vocab), EMBED_SIZE, NUM_HIDDENS, NUM_LAYERS)
#     decoder = Seq2SeqDecoder(len(dataset.tgt_vocab), EMBED_SIZE, NUM_HIDDENS, NUM_LAYERS)
#     model = Seq2SeqModel(encoder, decoder, tgt_pad=dataset.tgt_vocab['<pad>'])

#     # 4. Training
#     trainer = EliteTrainer(max_epochs=30, gradient_clip_val=1)
#     trainer.fit(model, train_iter)

#     # 5. Scientific Validation (Inference)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()

#     engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
#     fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

#     # --- Vectorization via your native Dataset class ---
#     # dataset.build converts raw strings into (src, tgt_input, src_len, label) tensors
#     batch = dataset.build(engs, fras)

#     # --- Autoregressive Prediction ---
#     # We unpack the batch into the predict_step
#     # num_steps=10 matches your dataset configuration
#     with torch.no_grad(): # Disable gradient tracking for speed and memory efficiency
#         preds, _ = model.predict_step(batch, device, num_steps=10)

#     # --- Post-Processing and BLEU Scoring ---
#     print(f"{'Source (EN)':<15} | {'Translation (FR)':<25} | {'BLEU'}")
#     print("-" * 55)

#     for en, fr, p in zip(engs, fras, preds):
#         translation = []
#         # Convert predicted indices back to tokens using your Vocab.to_tokens
#         # p is a tensor of indices for one sentence
#         tokens = dataset.tgt_vocab.to_tokens(p.tolist())
        
#         for token in tokens:
#             if token == '<eos>':
#                 break
#             if token not in ['<pad>', '<bos>']:
#                 translation.append(token)
                
#         # Join tokens into a string for BLEU calculation
#         pred_str = " ".join(translation)
        
#         # Calculate score using your native bleu function
#         score = bleu(pred_str, fr, k=2)
        
#         print(f"{en:<15} | {pred_str:<25} | {score:.3f}")

# if __name__ == "__main__":
#     run_experiment()