python preprocess.py ../data/food_raw.csv food_saved.csv
#Split the data in 80:20 train test-set
python partition.py food_saved.csv sampling_food.txt