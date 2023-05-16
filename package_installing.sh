pip install --upgrade pip

pip install -r requirements.txt

python -m spacy download fr_core_news_md

python -m spacy download en_core_web_sm

cat <<EOF>>nltk_download.py
import nltk


nltk.download('punkt')
EOF

python nltk_download.py

rm nltk_download.py
