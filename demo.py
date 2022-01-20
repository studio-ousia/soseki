import argparse

import streamlit as st


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

st.title(":robot_face: Question Answering Machine :robot_face:")

default_question = "Who won the first Nobel Prize in Physics?"

with st.form("question_form"):
    question = st.text_input("Input a question:", value=default_question, max_chars=200)
    submitted = st.form_submit_button("Go")

if submitted and len(question) > 0:
    st.header("Question")
    st.write(question)

    st.header("Answer Candidates")
    with st.spinner("Computing..."):
        answer_candidates = qa_model.answer_question(question, num_reading_passages=3)

    for i, answer_candidate in enumerate(answer_candidates):
        answer_text = answer_candidate.answer_text
        passage_text = answer_candidate.passage_text

        with st.container():
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

            with st.expander("Show Passage", expanded=i==0):
                passage_text = passage_text.replace(answer_text, "**{}**".format(answer_text))
                passage_text = passage_text.replace("[SEP]", "--", 1)
                passage_text = passage_text.replace("[SEP]", "")
                st.markdown(passage_text.strip())
