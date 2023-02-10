#!/bin/bash
model_path=models/crestereo_eth3d.pth
left=../shared/data/test/left.png
right=../shared/data/test/right.png
size_h_list=(1024 512 256 128)
size_w_list=(1536 768 384 192)
n_iter_list=(20 10 5 2)
n_samp=10
output=../shared/data/meas

if [ -d "$output" ]; then
    rm -fr "$output"
fi
mkdir -p "$output"

for i in "${!size_h_list[@]}"
do
    size_h="${size_h_list[$i]}"
    size_w="${size_w_list[$i]}"

    for n_iter in "${n_iter_list[@]}"
    do
        fname=""
        for j in $(seq 0 $((n_samp-1)))
        do
            python3 test_model.py \
                    --model_path "$model_path" \
                    --left "$left" \
                    --right "$right" \
                    --size_h "$size_h" \
                    --size_w "$size_w" \
                    --n_iter "$n_iter" \
                    --output "$output"/"${size_w}x${size_h}_iter"$(printf "%02d" "$n_iter").png \
                    --meas "$output"/t_"${size_w}x${size_h}"_iter$(printf "%02d" "$n_iter")
        done
    done
done
