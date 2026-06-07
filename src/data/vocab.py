from collections import Counter


class Vocabulary:

    def __init__(self, min_freq=2):

        self.min_freq = min_freq

        self.word_to_idx = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.idx_to_word = {
            0: "<PAD>",
            1: "<UNK>"
        }

    def build_vocab(self, words):

        counter = Counter(words)

        idx = 2

        for word, freq in counter.items():

            if freq >= self.min_freq:

                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word

                idx += 1

        return self.word_to_idx
    
    def numericalize(self, words):
        return [
            self.word_to_idx.get(word, self.word_to_idx["<UNK>"])
            for word in words
        ]