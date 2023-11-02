import os
import sys
import streamlit as st

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.config import st_sess_data


df_docs = st.session_state[st_sess_data.DF_DOCS]
target_var = st.session_state[st_sess_data.TARGET_VAR]
raw_docs, docs = (
    df_docs[target_var].tolist(),
    df_docs[f"clean_{target_var}"].tolist()
)
n_topics = st.session_state[st_sess_data.N_TOPICS]
bert_topic_inst = st.session_state[st_sess_data.BERTOPIC_INST]


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
    max_value=-1+n_topics
)
st.pyplot(
    bert_topic_inst.topic_plot(topic_id),
    use_container_width=True
)
df_topic_stat = bert_topic_inst.topic_stat(topic_id)
col1, col2, col3 = st.columns([0.2, 0.2, 0.6])
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
df_topic_info = bert_topic_inst.model.get_topic_info()
st.dataframe(
    df_doc_representative.query(
        f"topic_id == {topic_id}"
    ).reset_index(drop=True),
    use_container_width=True
)
with open(f"./data/topics_wc/topic_{topic_id}.svg", "rb") as file:
    btn = st.download_button(
        label=f"Download topic {topic_id} wordcloud",
        data=file,
        file_name=f"wc-topic-{topic_id}.svg",
        mime="image/png"
    )
st.download_button(
    label="Download representative docs as CSV",
    data=df_doc_representative.to_csv(
        encoding="utf-8",
        index=False
    ),
    file_name='df_doc_representative.csv',
    mime='text/csv',
)
st.download_button(
    label="Download topics info docs as CSV",
    data=df_topic_info.iloc[:, :4].to_csv(
        encoding="utf-8",
        index=False
    ),
    file_name='topics_info.csv',
    mime='text/csv',
)
