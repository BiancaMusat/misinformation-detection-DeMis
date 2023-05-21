#!/bin/sh

echo ">>>>> Balancing weak labels for: covid lies"
python code/get_balanced_weak_labeled_data.py \
    --input_dir="similarity_matrix/unlabeled_tweets/covid" \
    --output_filepath="datasets/tweets/weak_label_covid.csv" \
    --each_class_size=200

# echo ">>>>> Balancing weak labels for: weather"
# python code/get_balanced_weak_labeled_data.py \
#     --input_dir="similarity_matrix/unlabeled_tweets/weather" \
#     --output_filepath="datasets/tweets/weak_label_weather.csv" \
#     --each_class_size=60


# echo ">>>>> Balancing weak labels for: home_remedies"
# python code/get_balanced_weak_labeled_data.py \
#     --input_dir="similarity_matrix/unlabeled_tweets/home_remedies" \
#     --output_filepath="datasets/tweets/weak_label_home_remedies.csv" \
#     --each_class_size=200