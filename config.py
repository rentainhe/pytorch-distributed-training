class global_config():
    def __init__(self):
        self.batch_size = 128
        self.num_workers = 8
        self.pin_memory = True
        self.eval_every_epoch = True
        self.save_epoch = 15
        self.tensorboard_dir = 'runs'
        self.mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)