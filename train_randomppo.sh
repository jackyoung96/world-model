python train_ppo.py --env CartPole-v0 --tb_log --gpu $1 --randomize
python train_ppo.py --env Pendulum-v0 --tb_log --gpu $1 --randomize
python train_ppo.py --env takeoff-aviary-v0 --tb_log --gpu $1 --randomize