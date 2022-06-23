import argparse

import streamlit as st


DEFAULT_QUESTION = "Who won the first Nobel Prize in Physics?"


st.set_page_config(
    page_title="Question Answering Machine",
    page_icon=":robot_face:",
)


parser = argparse.ArgumentParser()
parser.add_argument("--onnx_model_dir", type=str)
parser.add_argument("--biencoder_ckpt_file", type=str)
parser.add_argument("--reader_ckpt_file", type=str)
parser.add_argument("--passage_file", type=str)
parser.add_argument("--passage_db_file", type=str)
parser.add_argument("--passage_embeddings_file", type=str)
parser.add_argument("--device", type=str)
args = parser.parse_args()


if args.onnx_model_dir is None and (args.biencoder_ckpt_file is None or args.biencoder_ckpt_file is None):
    raise ValueError("if --onnx_model_dir is unset, both of --biencoder_ckpt_file and --reader_ckpt_file must be set.")
if args.passage_embeddings_file is None:
    raise ValueError("--passage_embeddings_file must be set.")
if args.passage_db_file is None and args.passage_file is None:
    raise ValueError("--passage_db_file or --passage_file must be set.")


@st.cache(allow_output_mutation=True)
def get_qa_model():
    if args.onnx_model_dir is not None:
        from soseki.end_to_end.onnx_modeling import OnnxEndToEndQuestionAnswering

        model = OnnxEndToEndQuestionAnswering(
            onnx_model_dir=args.onnx_model_dir,
            passage_embeddings_file=args.passage_embeddings_file,
            passage_db_file=args.passage_db_file,
            passage_file=args.passage_file,
        )
    else:
        from soseki.end_to_end.modeling import EndToEndQuestionAnswering

        model = EndToEndQuestionAnswering(
            biencoder_ckpt_file=args.biencoder_ckpt_file,
            reader_ckpt_file=args.reader_ckpt_file,
            passage_embeddings_file=args.passage_embeddings_file,
            passage_db_file=args.passage_db_file,
            passage_file=args.passage_file,
            device=args.device,
        )

    return model


qa_model = get_qa_model()

# Write the page title.
st.title(":robot_face: Question Answering Machine :robot_face:")

# Write the question input form.
with st.form("question_form"):
    question = st.text_input("Input a question:", value=DEFAULT_QUESTION, max_chars=200)
    submitted = st.form_submit_button("Go")

if submitted and len(question) > 0:
    # Write the question.
    st.header("Question")
    st.write(question)

    # Write the answer candidates.
    st.header("Answer Candidates")
    # Fetch the answer candidates.
    with st.spinner("Computing..."):
        answer_candidates = qa_model.answer_question(question, num_passages_to_read=3)

    for i, answer_candidate in enumerate(answer_candidates):
        input_text = answer_candidate.input_text
        answer_text = answer_candidate.answer_text
        answer_start, answer_end = answer_candidate.answer_text_span

        with st.container():
            # Write the header of the candidate.
            if i == 0:
                emoji = ":first_place_medal:"
            elif i == 1:
                emoji = ":second_place_medal:"
            elif i == 2:
                emoji = ":third_place_medal:"
            else:
                emoji = ""

            st.subheader(emoji + " " + answer_text)
            st.caption("Score: {:.4f}".format(answer_candidate.score))

            # Write the candidate's reader input to an expander component.
            with st.expander("Show the Reader Input", expanded=i == 0):
                # Highlight the answer span.
                input_text = input_text[:answer_start] + "**{}**".format(answer_text) + input_text[answer_end:]

                # Remove the pad tokens.
                pad_token = getattr(qa_model.reader_tokenization.tokenizer, "pad_token", None) or None
                if pad_token is not None:
                    input_text = input_text.replace(pad_token, "")

                # Format other special tokens.
                for special_token in qa_model.reader_tokenization.tokenizer.all_special_tokens:
                    input_text = input_text.replace(special_token, f"`{special_token}`")

                input_text = input_text.replace("``", "")

                st.markdown(input_text.strip())
