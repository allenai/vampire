model=$1
input=$2
output=$3
parallel --ungroup \
        python -m scripts.run_vampire \
        $model \
        {1} \
        --batch 64 \
        --include-package vampire \
        --predictor vampire \
        --output-file $output \
        --silent ::: $input