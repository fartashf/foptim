# Benchmark Framework for Mini-batch Optimization Methods

## Examples
Single run:
```
CUDA_VISIBLE_DEVICES=0 python -m main.gvar --dataset mnist --lr 0.01 --weight_decay 0 --epochs 30 --lr_decay_epoch 30 --arch mlp --optim sgd --gvar_start 0 --g_bsnap_iter 1 --g_epoch  --g_estim sgd,sgd --g_batch_size 128 --logger_name runs/X
```

Grid run:
```
rm jobs/*.sh jobs/log/* -f && python grid_run.py --grid gvar --run_name mnist
pattern=""; for i in 1 2 3; do ./kill.sh $i $pattern; done
./start.sh
```
