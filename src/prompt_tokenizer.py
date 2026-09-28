import re


# ============================================================
# SIMPLE TEXT TOKENIZER
# ============================================================

class SimpleTokenizer:

    def __init__(self, vocab_size=10000):

        self.vocab_size = vocab_size

        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3
        }

    def tokenize(self, text):

        text = text.lower()

        words = re.findall(
            r"[a-z0-9]+",
            text
        )

        return words

    def word_to_id(self, word):

        if word in self.special_tokens:

            return self.special_tokens[word]

        # Simple deterministic word ID

        value = 0

        for character in word:

            value = (
                value * 31
                + ord(character)
            ) % (
                self.vocab_size - 4
            )

        return value + 4

    def encode(
        self,
        text,
        max_length=16
    ):

        words = self.tokenize(text)

        token_ids = [
            self.special_tokens["<START>"]
        ]

        for word in words:

            token_ids.append(
                self.word_to_id(word)
            )

        token_ids.append(
            self.special_tokens["<END>"]
        )

        # Padding

        if len(token_ids) < max_length:

            token_ids += [
                self.special_tokens["<PAD>"]
            ] * (
                max_length - len(token_ids)
            )

        # Truncate

        token_ids = token_ids[
            :max_length
        ]

        return token_ids


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    tokenizer = SimpleTokenizer()

    prompts = [
        "red car",
        "blue sky",
        "cute cat",
        "mountain landscape"
    ]

    for prompt in prompts:

        tokens = tokenizer.encode(
            prompt
        )

        print()
        print("Prompt:")
        print(prompt)

        print("Tokens:")
        print(tokens)