while getopts e: flag
do
    case "${flag}" in
        e) env=${OPTARG};;
    esac
done

if [ $env == 'github_action_env' ]
then
    pip install --user --upgrade pip

    pip install --user flake8 pytest

    pip install --user -r requirements.txt

    pip install --user cuml-cu11 --extra-index-url=https://pypi.nvidia.com
else
    pip install --upgrade pip

    pip install flake8 pytest

    pip install -r requirements.txt

    pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
fi

python -m spacy download fr_core_news_md

python -m spacy download en_core_web_sm

cat <<EOF>>nltk_download.py
import nltk


nltk.download('punkt')
EOF

python nltk_download.py

rm nltk_download.py
