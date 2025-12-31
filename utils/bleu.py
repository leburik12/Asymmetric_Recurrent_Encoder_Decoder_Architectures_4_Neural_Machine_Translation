class BLEUEvaluator:
    @staticmethod
    def bleu(pred_seq, label_seq, k):
        pred_tokens = pred_seq.split(' ')
        label_tokens = label_seq.split(' ')
        len_pred, len_label = len(pred_tokens), len(label_tokens)

        score = math.exp(min(0, 1 - len_label / len_pred))

        for n in range(1, k + 1):
            num_matches = 0
            label_subs = collections.Counter(
                [' '.join(label_tokens[i:i+n]) for i in range(len_label - n + 1)]
            )
            for i in range(len_pred - n + 1):
                sub = ' '.join(pred_tokens[i:i+n])
                if label_subs[sub] > 0:
                    num_matches += 1
                    label_subs[sub] -= 1
            score *= (num_matches / max(len_pred - n + 1, 1)) ** (0.5 ** n)

        return score
