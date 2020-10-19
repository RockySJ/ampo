params = {
    'type': 'AMPO_MMD',
    'universe': 'gym',
    'domain': 'Hopper',
    'task': 'v2',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -1,
        'max_model_t': None,
        'rollout_schedule': [5, 45, 1, 15],

        'n_adapt_per_epoch': 1000,
        'epoch_stop_adapt': 45,
        'kernel_sigma': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 1.0, 5, 10],
        'adapt_batch_size': 256
    }
}

