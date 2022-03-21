# python train_ppo.py --env CartPole-v0 --randomize --epoch 5000 --tb_log --gpu 0 &
# python train_ppo.py --env Pendulum-v0 --randomize --epoch 5000 --tb_log --gpu 1 &

python train_diffdac.py --env CartPole-v0 --randomize --epoch 5000 --tb_log --gpu 2 &
python train_diffdac.py --env Pendulum-v0 --randomize --epoch 5000 --tb_log --gpu 3 &