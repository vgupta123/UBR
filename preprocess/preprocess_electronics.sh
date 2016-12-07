python json_to_csv.py electronics_raw.csv ../data/reviews_Electronics_5.json

python preprocess.py electronics_raw.csv electronics_saved.csv
#Split the data in 80:20 train test-set
python partition.py electronics_saved.csv sampling_electronics.txt