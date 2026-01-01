class Seq2SeqPredictor:
    def __init__(self, model, tgt_vocab, device, max_len):
        self.model = model
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.max_len = max_len

    def predict(self, src_sentence, src_tokenizer):
        self.model.eval()

        src, src_len = src_tokenizer.tokenize_and_pad([src_sentence])
        src, src_len = src.to(self.device), src_len.to(self.device)

        _, enc_state = self.model.encoder(src, src_len)

        dec_input = torch.tensor([[self.tgt_vocab['<bos>']]], device=self.device)
        output_tokens = []

        for _ in range(self.max_len):
            dec_output, enc_state = self.model.decoder(dec_input, enc_state)
            pred = dec_output.argmax(dim=2)
            token = pred.item()

            if token == self.tgt_vocab['<eos>']:
                break

            output_tokens.append(token)
            dec_input = pred

        return ' '.join(self.tgt_vocab.decode(output_tokens))
