import logging
import time
import torch
import utils
import torch.utils.data as data

from torch.backends import cudnn
from datasets import get_dataset, data_transform
from functions.losses import noise_estimation_loss2, x0_estimation_loss2
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion
from utils import count_parameters


class DiffusionLostats(Diffusion):
    """Loss Stats"""
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)

    def get_data_loaders(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        batch_size = args.batch_size or config.training.batch_size
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        logging.info(f"train data:")
        logging.info(f"  root       : {dataset.root}")
        logging.info(f"  split      : {dataset.split}") if hasattr(dataset, 'split') else None
        logging.info(f"  len        : {len(dataset)}")
        logging.info(f"  batch_cnt  : {len(train_loader)}")
        logging.info(f"  batch_size : {batch_size}")
        logging.info(f"  shuffle    : True")
        logging.info(f"  num_workers: {config.data.num_workers}")

        logging.info(f"test data:")
        logging.info(f"  root          : {test_dataset.root}")
        logging.info(f"  len           : {len(test_dataset)}")
        logging.info(f"  test_data_dir : {args.test_data_dir}")
        logging.info(f"  batch_cnt     : {len(test_loader)}")
        logging.info(f"  batch_size    : {batch_size}")
        logging.info(f"  shuffle       : False")
        logging.info(f"  num_workers   : {config.data.num_workers}")
        return train_loader, test_loader

    def get_model(self):
        args, config = self.args, self.config
        in_ch = args.model_in_channels
        out_ch = args.model_in_channels
        resolution = args.data_resolution
        model = Model(config, in_channels=in_ch, out_channels=out_ch, resolution=resolution)
        cnt, str_cnt = count_parameters(model, log_fn=None)
        model_name = type(model).__name__
        logging.info(f"model: {model_name} ===================")
        logging.info(f"  config type: {config.model.type}")
        logging.info(f"  param size : {cnt} => {str_cnt}")
        logging.info(f"  ema_flag   : {self.args.ema_flag}")
        logging.info(f"  lr         : {self.args.lr}")
        logging.info(f"  model.to({self.device})")
        model.to(self.device)

        ckpt_path = self.args.sample_ckpt_path
        logging.info(f"  load model: {ckpt_path}")
        states = torch.load(ckpt_path, map_location=self.device)
        if self.args.todo == 'lostats:x0':
            if 'model_x0' in states:
                model.load_state_dict(states['model_x0'])
                logging.info(f"  load x0 model from states")
            else:
                model.load_state_dict(states)
                logging.info(f"  load pure model as states")
            ema_key = 'ema_x0'
        else:
            if 'model' in states:
                model.load_state_dict(states['model'])
                logging.info(f"  load et model from states")
            else:
                model.load_state_dict(states)
                logging.info(f"  load pure model as states")
            ema_key = 'ema_helper'
        if self.args.ema_flag:
            ema_helper = EMAHelper()
            ema_helper.register(model)
            logging.info(f"  load ema_helper from states[{ema_key}]")
            ema_helper.load_state_dict(states[ema_key])
            logging.info(f"  ema_helper: apply to model {type(model).__name__}")
            ema_helper.ema(model)

        if len(args.gpu_ids) > 1:
            logging.info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        cudnn.benchmark = True
        return model

    def run(self):
        train_loader, test_loader = self.get_data_loaders()
        model = self.get_model()
        model.eval()

        jump = self.args.timesteps
        h_cnt = 1000 // jump        # hit count
        b_cnt = len(test_loader)   # batch count
        hb_cnt = h_cnt * b_cnt      # hit * batch
        b_sz = self.args.batch_size or self.config.training.batch_size
        loss_ttl_arr = [0.] * 1000
        loss_cnt_arr = [0] * 1000
        data_start = time.time()
        logging.info(f"h_cnt  : {h_cnt}")
        logging.info(f"jump   : {jump}")
        logging.info(f"b_cnt  : {b_cnt}")
        logging.info(f"b_sz   : {b_sz}")
        logging.info(f"hb_cnt : {hb_cnt}")
        with torch.no_grad():
            for b_idx, (x, y) in enumerate(test_loader):
                x = x.to(self.device)
                x = data_transform(self.config, x)
                b_sz = x.size(0)
                for ts in range(0, 1000, jump):
                    t = torch.ones((b_sz,), dtype=torch.int, device=self.device)
                    t *= ts
                    e = torch.randn_like(x)
                    if self.args.todo == 'lostats:x0':
                        loss, _ = x0_estimation_loss2(model, x, t, e, self.alphas_cumprod)
                    else:
                        loss, _ = noise_estimation_loss2(model, x, t, e, self.alphas_cumprod)
                    loss_ttl_arr[ts] += loss.item()
                    loss_cnt_arr[ts] += 1

                    ts_idx = ts // jump
                    if ts_idx % 10 == 0 or ts_idx == h_cnt - 1:
                        elp, eta = utils.get_time_ttl_and_eta(data_start, b_idx * h_cnt + ts_idx, hb_cnt)
                        avg = loss_ttl_arr[ts] / loss_cnt_arr[ts]
                        logging.info(f"B{b_idx:03d}/{b_cnt}.ts{ts:3d}"
                                     f" loss:{loss.item():7.2f} avg:{avg:7.2f}; elp:{elp}, eta:{eta}")
                # for ts
            # for loader
        # with
        loss_avg_arr = [0.] * 1000
        for i in range(len(loss_avg_arr)):
            if loss_cnt_arr[i] > 0:
                loss_avg_arr[i] = loss_ttl_arr[i] / loss_cnt_arr[i]
        # for
        utils.output_list(loss_ttl_arr, 'loss_ttl')
        utils.output_list(loss_cnt_arr, 'loss_cnt')
        utils.output_list(loss_avg_arr, 'loss_avg')
    # run(self)

# class
