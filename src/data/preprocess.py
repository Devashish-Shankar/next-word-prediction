import re


class TextPreprocessor:

    def clean_text(self, text):

        # lowercase
        text = text.lower()

        # remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()