import string
import re
import unicodedata
from typing import List, Tuple

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from ..utils.data_utils import AnswerSpan, find_sublist_slices


class ReaderTokenization:
    def __init__(self, base_model_name: str, answer_normalization_type: str = "simple") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.answer_normalization_type = answer_normalization_type

    def tokenize_input(self, question: str, passage_title: str, passage_text: str, **kwargs) -> BatchEncoding:
        question = self._preprocess_question(question)
        passage_title = self._preprocess_passage_title(passage_title)
        passage_text = self._preprocess_passage_text(passage_text)

        passage = passage_title + self.tokenizer.sep_token + passage_text

        return self.tokenizer(question, passage, **kwargs)

    def tokenize_input_with_answers(
        self, question: str, passage_title: str, passage_text: str, answers: List[str], **kwargs
    ) -> Tuple[BatchEncoding, List[Tuple[int, int]]]:
        tokenized_input = self.tokenize_input(question, passage_title, passage_text, **kwargs)
        input_token_ids = tokenized_input["input_ids"]

        # The special tokens for separating two input texts are not consistent across models in Huggiing Face.
        # Here we introduce some heuristics which can cover most cases.
        if self.tokenizer.eos_token_id is not None:
            sep_token_id = self.tokenizer.eos_token_id
        elif self.tokenizer.sep_token_id is not None:
            sep_token_id = self.tokenizer.eos_token_id
        else:
            sep_token_id = -1

        answer_spans = []
        for answer in answers:
            answer = self._preprocess_answer(answer)
            answer_token_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            answer_slices = find_sublist_slices(
                answer_token_ids,
                input_token_ids,
                start=input_token_ids.index(sep_token_id) if sep_token_id in input_token_ids else 0,
            )
            for start, stop in answer_slices:
                answer_spans.append((start, stop - 1))

        return tokenized_input, answer_spans

    def _preprocess_question(self, text: str) -> str:
        text = text.rstrip("?")
        return text

    def _preprocess_passage_title(self, text: str) -> str:
        return text

    def _preprocess_passage_text(self, text: str) -> str:
        return text

    def _preprocess_answer(self, text: str) -> str:
        return text

    def _is_subword_id(self, token_id: int) -> bool:
        # Here we only consider BERT-style subwords.
        return self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith("##")

    def _compute_best_answer_spans(
        self,
        input_ids: List[int],
        start_logits: List[float],
        end_logits: List[float],
        num_answer_spans: int = 1,
        max_answer_length: int = 10,
        allow_overrapping_spans: bool = True,
        expand_subwords_on_edges: bool = True,
    ) -> List[AnswerSpan]:
        # https://github.com/facebookresearch/DPR/blob/10ca7200a2549dbf4e9542aa61a6f541e535499f/dpr/data/reader_data.py#L370

        # Extract all possible spans no longer than max_answer_length.
        possible_spans = []
        for s, start_logit in enumerate(start_logits):
            for e, end_logit in enumerate(end_logits[s : s + max_answer_length], start=s):
                possible_spans.append(AnswerSpan(s, e, start_logit, end_logit))

        possible_spans = sorted(possible_spans, key=lambda s: s.start_logit + s.end_logit, reverse=True)

        # Select spans with the highest logit scores from all the possible spans.
        selected_spans = []
        for span in possible_spans:
            if not allow_overrapping_spans:
                # Skip spans which are overrapping any of the already selected spans.
                is_overrapping = False
                for selected_span in selected_spans:
                    if span.start <= selected_span.start <= selected_span.end <= span.end:
                        is_overrapping = True
                        break
                    if selected_span.start <= span.start <= span.end <= selected_span.end:
                        is_overrapping = True
                        break

                if is_overrapping:
                    continue

            if expand_subwords_on_edges:
                # Expand fragmented subwords on the edges of spans.
                while span.start > 0 and self._is_subword_id(input_ids[span.start]):
                    span.start -= 1
                while span.end < len(input_ids) - 1 and self._is_subword_id(input_ids[span.end + 1]):
                    span.end += 1

            selected_spans.append(span)
            if len(selected_spans) == num_answer_spans:
                break

        return selected_spans

    def _get_input_and_answer_texts(
        self,
        input_ids: List[int],
        answer_span: AnswerSpan,
    ) -> Tuple[str, str, Tuple[int, int]]:
        left_input_ids = input_ids[: answer_span.start]
        answer_input_ids = input_ids[answer_span.start : answer_span.end + 1]

        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        left_input_tokens = self.tokenizer.convert_ids_to_tokens(left_input_ids)
        answer_tokens = self.tokenizer.convert_ids_to_tokens(answer_input_ids)

        input_text = self.tokenizer.convert_tokens_to_string(input_tokens)
        left_input_text = self.tokenizer.convert_tokens_to_string(left_input_tokens)
        answer_text = self.tokenizer.convert_tokens_to_string(answer_tokens)

        answer_start = input_text.find(answer_text, len(left_input_text))
        answer_end = answer_start + len(answer_text)

        return input_text, answer_text, (answer_start, answer_end)

    def _normalize_answer(self, text: str) -> str:
        if self.answer_normalization_type == "dpr":
            # Answer text normalization in the original DPR.
            # https://github.com/facebookresearch/DPR/blob/6b7e36d31850498c253e26f4b2628eccf1a8924e/dpr/data/qa_validation.py#L146-L164
            text = unicodedata.normalize("NFD", text)
            text = text.lower()
            text = "".join(c for c in text if c not in frozenset(string.punctuation))
            text = re.sub(r"\b(a|an|the)\b", " ", text)
            text = " ".join(text.split())

        elif self.answer_normalization_type == "simple_nfd":
            # Simple matching of texts after unicode normalization to the NFD form.
            text = unicodedata.normalize("NFD", text)
            text = text.lower()
            text = " ".join(text.split())

        elif self.answer_normalization_type == "simple_nfkc":
            # Simple matching of texts after unicode normalization to the NFKC form.
            # Most suitable for Japanese texts.
            text = unicodedata.normalize("NFKC", text)
            text = text.lower()
            text = "".join(text.split())  # Remove whitespaces to avoid problems with Japanese tokenization

        elif self.answer_normalization_type == "none":
            # No normalization.
            pass

        else:
            raise ValueError("Invalid answer_normalization_type is specified.")

        return text
