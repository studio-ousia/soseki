from typing import List, Union

from transformers import AutoTokenizer, T5Tokenizer
from transformers.tokenization_utils_base import BatchEncoding


class EncoderTokenization:
    def __init__(self, base_model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def tokenize_questions(self, questions: Union[str, List[str]], **kwargs) -> BatchEncoding:
        if isinstance(questions, str):
            questions = self._preprocess_question(questions)
        else:
            questions = [self._preprocess_question(question) for question in questions]

        return self.tokenizer(questions, **kwargs)

    def tokenize_passages(self, titles: Union[str, List[str]], texts: Union[str, List[str]], **kwargs) -> BatchEncoding:
        if isinstance(titles, str):
            titles = self._preprocess_passage_title(titles)
        else:
            titles = [self._preprocess_passage_title(title) for title in titles]

        if isinstance(texts, str):
            texts = self._preprocess_passage_text(texts)
        else:
            texts = [self._preprocess_passage_text(text) for text in texts]

        return self.tokenizer(titles, texts, **kwargs)

    def _preprocess_question(self, text: str) -> str:
        text = text.rstrip("?")
        return text

    def _preprocess_passage_title(self, text: str) -> str:
        return text

    def _preprocess_passage_text(self, text: str) -> str:
        return text
