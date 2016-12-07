python json_to_csv.py movies_raw.csv ../data/reviews_Movies_and_TV_5.json

python preprocess.py movies_raw.csv movies_saved.csv
#Split the data in 80:20 train test-set
python partition.py movies_saved.csv sampling_movies.txt