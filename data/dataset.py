class TranslationDataset:
    def __init__(self, src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer):
        self.src, self.src_len = src_tokenizer.tokenize_and_pad(src_sentences)
        self.tgt, self.tgt_len = tgt_tokenizer.tokenize_and_pad(tgt_sentences)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_len[idx], self.tgt_len[idx]
