wget https://www.kaggle.com/snap/amazon-fine-food-reviews/downloads/amazon-fine-foods-release-2016-01-08-20-34-54.zip

unzip amazon-fine-foods-release-2016-01-08-20-34-54.zip

rm -rf amazon-fine-foods-release-2016-01-08-20-34-54.zip

cd amazon-fine-foods

mv Reviews.csv ../food_raw.csv

cd ..