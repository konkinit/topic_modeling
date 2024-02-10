while getopts e: flag
do
    case "${flag}" in
        e) env=${OPTARG};;
    esac
done

if [ $env == 'ci_environment' ]
then
    pip install --user --upgrade pip

    pip install --user flake8 pytest

    pip install --user -r requirements.txt

else
    pip install --upgrade pip

    pip install flake8 pytest

    pip install -r requirements.txt

    pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==23.12.* cuml-cu12==23.12.*
fi

python -m spacy download fr_core_news_md

python -m spacy download en_core_web_sm

cat <<EOF>>nltk_download.py
import nltk


nltk.download('punkt')
EOF

python nltk_download.py

rm nltk_download.py
