class NMTDataLoader:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def preprocess(self):
        text = self.raw_text.replace('\u202f', ' ').replace('\xa0', ' ')
        return text.lower()

    def tokenize(self, text):
        source, target = [], []
        for line in text.split('\n'):
            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0].split(' '))
                target.append(parts[1].split(' '))
        return source, target
