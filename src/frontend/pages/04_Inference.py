import streamlit as st

df_docs = st.session_state["df_docs"]
target_var = st.session_state["target_var"]
raw_docs, docs = (
    df_docs[target_var].tolist(),
    df_docs[f"clean_{target_var}"].tolist()
)
n_topics = st.session_state["n_topics"]
bert_topic_inst = st.session_state["bert_topic_inst"]


st.markdown(
    f"""
    # Inference

    Finally we have an output of {n_topics} topics
    """
)


topic_id = st.number_input(
    'Select a number for more details on the related topic',
    value=0,
    min_value=0,
    max_value=n_topics-1
)
st.pyplot(
    bert_topic_inst.topic_plot(topic_id),
    use_container_width=True
)
df_topic_stat = bert_topic_inst.topic_stat(topic_id)
col1, col2, col3 = st.columns(3)
col1.metric(
    label="Topic Id", value=topic_id
)
col2.metric(
    label="Docs Frequency", value=df_topic_stat.Count.values[0]
)
col3.metric(
    label="Topic Name", value=df_topic_stat.Name.values[0]
)
df_doc_representative = bert_topic_inst.representative_docs(docs, raw_docs)
st.dataframe(
    df_doc_representative.query(
        f"topic_id == {topic_id}"
    ).reset_index(drop=True),
    use_container_width=True
)
with open(f"./data/topics_wc/topic_{topic_id}.png", "rb") as file:
    btn = st.download_button(
        label=f"Download topic {topic_id} wordcloud",
        data=file,
        file_name=f"wc-topic-{topic_id}.png",
        mime="image/png"
    )
st.download_button(
    label="Download representative docs as CSV",
    data=df_doc_representative.to_csv(encoding="utf-8"),
    file_name='df_doc_representative.csv',
    mime='text/csv',
)
