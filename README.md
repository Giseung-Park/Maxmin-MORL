# Maxmin-MORL
Implementation for the ICML 2024 paper: "The Max-Min Formulation of Multi-Objective Reinforcement Learning: From Theory to a Model-Free Algorithm"

**Openreview Paper Link**

https://openreview.net/forum?id=cY9g0bwiZx

**ArXiv**

https://arxiv.org/abs/2406.07826

**Installation** 
1. conda env create -f maxmin_mo_env.yaml
2. conda activate maxmin_mo_env
3. sudo add-apt-repository ppa:sumo/stable
4. sudo apt-get update
5. sudo apt-get install sumo sumo-tools sumo-doc
6. echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
7. source ~/.bashrc

The same description is written in Installation.txt.
You may also refer to https://github.com/LucasAlegre/sumo-rl for SUMO installation.

**Note**
- You can report the return values after installing wandb. (We used wandb==0.15.12 version.)
  If you do not want to use wandb, you may check maxmin_algorithms.py and common/off_policy_algorithm.py.
- If installation fails, you can first install torch, remove conda maxmin_mo_env, and then reinstall it.


**Run**

python maxmin_algorithms.py -se 0

- We used a hardware of Intel Core i9-10900X CPU @ 3.70GHz.
- We used five random seeds: 0-4.

**Citation**

@inproceedings{
park2024the,
title={The Max-Min Formulation of Multi-Objective Reinforcement Learning: From Theory to a Model-Free Algorithm},
author={Giseung Park and Woohyeon Byeon and Seongmin Kim and Elad Havakuk and Amir Leshem and Youngchul Sung},
booktitle={Forty-first International Conference on Machine Learning},
year={2024}
}

**Contact**

If you have any question or discussion, feel free to send an e-mail to gs.park@kaist.ac.kr.
