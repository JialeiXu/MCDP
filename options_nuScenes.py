# Copyright Niantic 2019. Patent Pending. All save_code_localSrights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class nuScenes_Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="nuScenes options")

        self.parser.add_argument('--notes',type=str,default='')
        self.parser.add_argument('--wandb',type=bool,default=False)
        self.parser.add_argument('--tensorboardX',type=bool,default=True)
        self.parser.add_argument("--model_name",type=str,help="the name of the folder to save the model in",
                                 default="nuScnenes_clean_data")
        self.parser.add_argument('--self_sup',type=bool,default=True)
        self.parser.add_argument('--code_num',type=int,default=32)
        self.parser.add_argument('--depth_code',type=bool,default=True)
        self.parser.add_argument('--ddp',type=bool,default=True)
        self.parser.add_argument('--GRU',type=bool, default=False)
        self.parser.add_argument('--depth_consistency_loss',type=bool,default=True)
        self.parser.add_argument('--weight_depth_consistency_loss',type=float,default=1e-4)
        self.parser.add_argument('--cross_cam_photometric_loss',type=bool,default=False)
        self.parser.add_argument('--refine_times',type=int,default=2)
        self.parser.add_argument('--json_file',type=str,
                                 default='./datasets/nuScenes/sweep/sweeps_Cam6.json') #new_data_Cam_01
        self.parser.add_argument('--json_file_val', type=str,
                                 default='./datasets/nuScenes/sweep/sweeps_F.json')  # new_data_Cam_01
        self.parser.add_argument('--json_file_val_list', type=list,
                                 default=['./datasets/nuScenes/sweep/sweeps_F.json',
                                          ])
        self.parser.add_argument('--train_forward_back_l_r_pc', type=list,
                                 default=[True, True, True, False, False])

        self.parser.add_argument('--save_local_files',type=list,default=['Camera.py','datasets/dataload_nuScenes.py','eval.py','options_nuScenes.py','self_sup.py','self_sup_loss.py',
                                                                         'train_nuScenes.py','utils.py'])
        self.parser.add_argument('--makedirs',type=list,default=['datasets'])
        self.parser.add_argument('--save_local_folders',type=list,default=['networks'])
        self.parser.add_argument('--seed',type=int,default=0)

        self.parser.add_argument('--val_forward_back_l_r_pc', type=list,
                                 default=[False, False, True, False, False])
        self.parser.add_argument('--rank', type=int, default=0)
        self.parser.add_argument('--depth_metric_names',type=list,default=["init/abs_rel", "init/sq_rel", "init/rms", "init/log_rms", "init/a1", "init/a2", "init/a3","init/depth_con"])
        self.parser.add_argument('--error_depth_con',type=bool,default=True)
        self.parser.add_argument('--best_rel',type=float,default=100.)
        self.parser.add_argument('--world_size', type=int, default=1)
        self.parser.add_argument('--data_aug',type=bool,default=True)
        self.parser.add_argument('--robust_loss',type=bool,default=False)
        self.parser.add_argument('--robust_loss_scale',type=float,default=0.2)
        self.parser.add_argument('--val_step',type=int,default=15000)
        self.parser.add_argument('--origin_size',type=list,default=[900,1600])

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='/data/disk_a/xujl/Datasets/nuScenes/resize/'
                                 )
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join("./logs"))

        # TRAINING options
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=448)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=768)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=80.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION optionsF
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=10)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default='./logs/Cam6/models'
                                 )
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=200)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument('--save_code_files',type=list,default=['./Camera.py','./new_data_load.py',
                                    './eval.py','./layers.py','./options.py','./self_sup.py',
                                    './self_sup_loss.py','./train_ddp.py','./utils.py',
                                    './networks/depth_decoder.py','./networks/pose_cnn.py','./networks/pose_decoder.py','./networks/pose_decoder.py','./networks/feature2weight.py','./networks/GRU_refine.py'
                                    ])
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options