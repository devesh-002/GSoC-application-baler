bash try.sh $1
poetry run python baler --project=$1 --mode=train
poetry run python baler --project=$1 --mode=compress
poetry run python baler --project=$1 --mode=decompress
poetry run python baler --project=$1 --mode=analysis
