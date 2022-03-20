python train_ppo.py --env CartPole-v0 --randomize --tb_log
python train_ppo.py --env Pendulum-v0 --randomize --tb_log

python train_diffdac.py --env CartPole-v0 --randomize --tb_log
python train_diffdac.py --env Pendulum-v0 --randomize --tb_log