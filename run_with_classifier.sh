for i in 12346 12347 12348
do
    python with_classifier.py --seed $i --number-of-batch 5 --batch-size 210 --batch-time 60
done
python calculate_averaged_result.py --approach with_classifier