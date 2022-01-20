import string
import re
import unicodedata
from typing import List, Tuple

from transformers import AutoTokenizer, T5Tokenizer
from transformers.tokenization_utils_base import BatchEncoding

from ..utils.data_utils import AnswerSpan, find_sublist_slices


class ReaderTokenization:
    def __init__(
        self,
        base_model_name: str,
        include_title_in_passage: bool = False,
        answer_normalization_type: str = "simple",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        self.include_title_in_passage = include_title_in_passage
        self.answer_normalization_type = answer_normalization_type

    def tokenize_input(self, question: str, passage_title: str, passags_text: str, **kwargs) -> BatchEncoding:
        question = self._preprocess_question(question)
        passage_title = self._preprocess_passage_title(passage_title)
        passags_text = self._preprocess_passage_text(passags_text)

        if self.include_title_in_passage:
            text_a = question
            text_b = passage_title + self.tokenizer.sep_token + passags_text
        else:
            text_a = question + self.tokenizer.sep_token + passage_title
            text_b = passags_text

        return self.tokenizer(text_a, text_b, **kwargs)

    def tokenize_input_with_answers(
        self, question: str, passage_title: str, passags_text: str, answers: List[str], **kwargs
    ) -> Tuple[BatchEncoding, List[Tuple[int, int]]]:
        tokenized_input = self.tokenize_input(question, passage_title, passags_text, **kwargs)

        answer_spans = []
        for answer in answers:
            answer = self._preprocess_answer(answer)
            answer_token_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            answer_slices = find_sublist_slices(
                answer_token_ids,
                tokenized_input["input_ids"],
                start=tokenized_input["token_type_ids"].index(1)
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
        return self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith("##")

    def _compute_best_answer_spans(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        token_type_ids: List[int],
        start_logits: List[float],
        end_logits: List[float],
        num_answer_spans: int = 1,
        max_answer_length: int = 10,
        allow_overrapping_spans: bool = True,
        expand_subwords_on_edges: bool = True,
    ) -> List[AnswerSpan]:
        # extract all possible spans no longer than max_answer_length
        possible_spans: List[AnswerSpan] = []
        for start, start_logit in enumerate(start_logits):
            for end, end_logit in enumerate(end_logits[start:start + max_answer_length], start=start):
                # ignore spans stretching out of the passage
                if any(x != 1 for x in attention_mask[start:end + 1]):
                    continue
                if any(x != 1 for x in token_type_ids[start:end + 1]):
                    continue

                # ignore spans containing [SEP] tokens
                if any(x == self.tokenizer.sep_token_id for x in input_ids[start:end + 1]):
                    continue

                possible_spans.append(AnswerSpan(start, end, start_logit, end_logit))

        possible_spans = sorted(possible_spans, key=lambda s: s.start_logit + s.end_logit, reverse=True)

        # select spans with highest logit scores from all the possible spans
        selected_spans: List[AnswerSpan] = []
        for span in possible_spans:
            if not allow_overrapping_spans:
                # skip spans which are overrapping any of the already selected spans
                is_overrapping = False
                for selected_span in selected_spans:
                    if span.start <= selected_span.start <= selected_span.end <= span.end:
                        is_overrapping = True
                        break
                    elif selected_span.start <= span.start <= span.end <= selected_span.end:
                        is_overrapping = True
                        break

                if is_overrapping:
                    continue

            if expand_subwords_on_edges:
                # expand fragmented subwords on the edges of spans
                while (
                    span.start > 0
                    and self._is_subword_id(input_ids[span.start])
                    and token_type_ids[span.start - 1] == 1
                ):
                    span.start -= 1
                while (
                    span.end < len(input_ids) - 1
                    and self._is_subword_id(input_ids[span.end + 1])
                    and token_type_ids[span.end + 1] == 1
                ):
                    span.end += 1

            selected_spans.append(span)
            if len(selected_spans) == num_answer_spans:
                break

        return selected_spans

    def _get_answer_passage_texts_from_input_ids(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        token_type_ids: List[int],
        answer_span: AnswerSpan,
    ) -> Tuple[str, str]:
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        answer_tokens = input_tokens[answer_span.start:answer_span.end + 1]

        passage_start = token_type_ids.index(1)
        passage_length = token_type_ids.count(1)
        passage_tokens = input_tokens[passage_start:passage_start + passage_length]

        answer_text = self.tokenizer.convert_tokens_to_string(answer_tokens)
        passage_text = self.tokenizer.convert_tokens_to_string(passage_tokens)

        return answer_text, passage_text

    def _normalize_answer(self, text: str) -> str:
        if self.answer_normalization_type == "dpr":
            # Answer text normalization in the original DPR
            # https://github.com/facebookresearch/DPR/blob/6b7e36d31850498c253e26f4b2628eccf1a8924e/dpr/data/qa_validation.py#L146-L164
            text = unicodedata.normalize("NFD", text)
            text = text.lower()
            text = "".join(c for c in text if c not in frozenset(string.punctuation))
            text = re.sub(r"\b(a|an|the)\b", " ", text)
            text = " ".join(text.split())

        elif self.answer_normalization_type == "simple_nfd":
            # Simple matching of texts after unicode normalization to the NFD form
            text = unicodedata.normalize("NFD", text)
            text = text.lower()
            text = " ".join(text.split())

        elif self.answer_normalization_type == "simple_nfkc":
            # Simple matching of texts after unicode normalization to the NFKC form
            # Most suitable for Japanese texts
            text = unicodedata.normalize("NFKC", text)
            text = text.lower()
            text = "".join(text.split())  # Remove whitespaces to avoid problems with Japanese tokenization

        elif self.answer_normalization_type == "none":
            # No normalization
            pass

        else:
            raise ValueError("Invalid answer_normalization_type is specified.")

        return text
