# reproduce kaggle model
# $1 for save loss record path, $2 for save model path, $3 for input dir

python -u Train.py -lop $! -mop $2 --data_path $3