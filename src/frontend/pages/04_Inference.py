import streamlit as st


st.title("Inference")
st.markdown(
    """Choose a topic
    """
)


n_topics = st.session_state["n_topics"]
bert_topic_inst = st.session_state["bert_topic_inst"]


topic_id = st.number_input(
    'Select a number for more details on the related topic',
    value=0,
    min_value=0,
    max_value=n_topics
)
st.plotly_chart(
    bert_topic_inst.model.visualize_barchart(
        topics=[topic_id], n_words=15
    )
)
