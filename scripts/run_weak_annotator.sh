#!/bin/sh

UNLABELED_TWEET_FILEPATH="datasets/unlabeled_tweets/sample_unlabeled_tweets.csv"


echo ">>>>> Generating weak labels for: covid-lies"
CLAIM_IDS_OF_INTEREST="1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62"
CUDA_VISIBLE_DEVICES=0 \
python code/weak_classifier.py \
    --input_filepath_tweet=$UNLABELED_TWEET_FILEPATH \
    --input_filepath_claim="datasets/claims/misconceptions.csv" \
    --similarity_model="sentence-transformers/bert-base-nli-stsb-mean-tokens" \
    --correct_claim_ids="$CLAIM_IDS_OF_INTEREST" \
    --top_k="5" \
    --save_dir="similarity_matrix/unlabeled_tweets/covid" \
    --sample_size=3000

# echo ">>>>> Generating weak labels for: weather"
# CLAIM_IDS_OF_INTEREST="135, 545, 586"
# CUDA_VISIBLE_DEVICES=0 \
# python code/weak_classifier.py \
#     --input_filepath_tweet=$UNLABELED_TWEET_FILEPATH \
#     --input_filepath_claim="datasets/claims/claims-covid-19.csv" \
#     --similarity_model="sentence-transformers/bert-base-nli-stsb-mean-tokens" \
#     --correct_claim_ids="$CLAIM_IDS_OF_INTEREST" \
#     --top_k="10" \
#     --save_dir="similarity_matrix/unlabeled_tweets/weather" \
#     --sample_size=3000


# echo ">>>>> Generating weak labels for: home_remedies"
# CLAIM_IDS_OF_INTEREST="145, 243, 259, 283, 336, 360, 506, 583"
# CUDA_VISIBLE_DEVICES=0 \
# python code/weak_classifier.py \
#     --input_filepath_tweet=$UNLABELED_TWEET_FILEPATH \
#     --input_filepath_claim="datasets/claims/claims-covid-19.csv" \
#     --similarity_model="sentence-transformers/bert-base-nli-stsb-mean-tokens" \
#     --correct_claim_ids="$CLAIM_IDS_OF_INTEREST" \
#     --top_k="10" \
#     --save_dir="similarity_matrix/unlabeled_tweets/home_remedies" \
#     --sample_size=3000