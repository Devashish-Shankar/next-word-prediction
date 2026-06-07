class SequenceGenerator:

    def __init__(self, sequence_length=5):

        self.sequence_length = sequence_length

    def generate(self, token_ids):

        X = []
        y = []

        for i in range(
            len(token_ids) - self.sequence_length
        ):

            input_seq = token_ids[
                i:i+self.sequence_length
            ]

            target = token_ids[
                i+self.sequence_length
            ]

            X.append(input_seq)
            y.append(target)

        return X, y