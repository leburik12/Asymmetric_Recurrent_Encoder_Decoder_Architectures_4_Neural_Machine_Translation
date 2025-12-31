class Tokenizer:
    def __init__(self, vocab, num_steps):
        self.vocab = vocab
        self.num_steps = num_steps

    def tokenize_and_pad(self, sentences):
        all_tokens, valid_lens = [], []

        for sent in sentences:
            tokens = self.vocab.encode(sent)
            valid_lens.append(min(len(tokens), self.num_steps))

            if len(tokens) < self.num_steps:
                tokens += [self.vocab['<pad>']] * (self.num_steps - len(tokens))
            else:
                tokens = tokens[:self.num_steps]

            all_tokens.append(tokens)

        return torch.tensor(all_tokens), torch.tensor(valid_lens)
