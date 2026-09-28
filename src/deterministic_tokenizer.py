import re


class DeterministicTokenizer:

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self, max_length=16):

        self.max_length = max_length

        # Fixed vocabulary used by our current MNIST captions
        words = [
            "a",
            "handwritten",
            "digit",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine"
        ]

        self.vocab = {
            word: index + 10
            for index, word in enumerate(words)
        }

    def tokenize(self, text):

        words = re.findall(
            r"[a-z0-9]+",
            text.lower()
        )

        token_ids = [self.START]

        for word in words:

            token_ids.append(
                self.vocab.get(
                    word,
                    self.UNK
                )
            )

        token_ids.append(self.END)

        token_ids = token_ids[
            :self.max_length
        ]

        while len(token_ids) < self.max_length:

            token_ids.append(
                self.PAD
            )

        return token_ids


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    tokenizer = DeterministicTokenizer()

    prompts = [
        "a handwritten digit zero",
        "a handwritten digit seven",
        "a handwritten digit nine"
    ]

    for prompt in prompts:

        print()
        print("Prompt:")
        print(prompt)

        print("Token IDs:")
        print(
            tokenizer.tokenize(prompt)
        )