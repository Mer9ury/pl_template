for seed in 1
do
    python3 run.py --config /workspace/inrct/configs/nilut_baseline.yaml --project INRCT --name inrct_baseline --lr 1e-4 --gpu 0,1,2,3 --seed $seed --use_wandb
done