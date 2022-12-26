import numpy as np
import torch.nn
from dataload_DDAD import DDAD_RAWDataset
from torch.utils.data import DataLoader
from utils import *
from layers import disp_to_depth
import networks,argparse
import copy

def val(args,eval_models,writers,val_loader,val_whole):
    """Validate the model on a single minibatch
    """
    models = eval_models

    error = AverageMeter(i=len(args.depth_metric_names))
    models = set_eval(models)
    len_val = len(val_loader)
    for batch_i,inputs in enumerate(val_loader):
        print(batch_i)
        if writers!=None:
            print('epoch%d val %d/%d'%(args.epoch,batch_i,len_val))
        for key, ipt in inputs.items():
            inputs[key] = ipt.cuda()
        cameras = ['l', 'f'] if args.error_depth_con else ['f']
        outputs = {}
        for camera_i in cameras:
            with torch.no_grad():
                features = models["encoder"](inputs["color", camera_i, 0, 0])
                outputs.update(models["depth"](features,camera_i))
        if args.code_num > 1 and args.depth_code:
            if args.GRU:
                with torch.no_grad():
                    outputs,losses = models['GRU'](args,inputs,outputs, None, is_train=False)
            else:
                for refine_time_i in range(args.refine_times):
                    with torch.no_grad():
                        outputs,W_code = models['refine_net_'+str(refine_time_i)](args,inputs,outputs,refine_time_i)

        losses = []
        losses = compute_depth_losses(args, inputs, outputs, losses, 'init')
        if args.depth_code and args.code_num > 1:
            if args.GRU:
                losses = compute_depth_losses(args, inputs, outputs,losses, 'refine',refine_time=0)
            else:
                for refine_time_i in range(args.refine_times):
                    losses = compute_depth_losses(args, inputs, outputs,losses, 'refine',refine_time_i)
        error.update(losses,n=inputs['color','f',0,0].shape[0])

        if (not val_whole) and batch_i==0: break

    return error


class val_Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")
        self.parser.add_argument('--GRU', type=bool, default=False)
        self.parser.add_argument('--ddp', type=bool, default=True)
        self.parser.add_argument('--code_num',type=int,default=32)
        self.parser.add_argument('--depth_code',type=bool,default=True)
        self.parser.add_argument('--json_file', type=str,
                                 default='./datasets/new_mask_data_Cam_01.json')
        self.parser.add_argument('--json_file_val', type=str,
                                 default='./datasets/new_mask_data_Cam_01.json')
        self.parser.add_argument('--refine_times',type=int,default=2)
        self.parser.add_argument('--train_forward_back_l_r_pc', type=list,
                                 default=[True, True, True, True, False])
        self.parser.add_argument('--val_forward_back_l_r_pc', type=list,
                                 default=[False, False, True, True, False])
        self.parser.add_argument('--rank', type=int, default=0)
        self.parser.add_argument('--origin_size', type=list, default=[1216, 1936])
        self.parser.add_argument('--depth_metric_names', type=list,
                                 default=["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3",
                                          "de/depth_con"])
        self.parser.add_argument('--error_depth_con', type=bool, default=True)

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='/data/disk_a/xujl/Datasets/DDAD/my_ddad_resize/'
                                 )
        self.parser.add_argument("--log_dir", type=str, help="log directory", default=os.path.join("./logs"))

        # TRAINING options
        self.parser.add_argument("--num_layers",type=int,help="number of resnet layers",default=18,choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--height",type=int,help="input image height",default=384)
        self.parser.add_argument("--width",type=int,help="input image width",default=640)
        self.parser.add_argument("--scales",nargs="+",type=int,help="scales used in the loss",default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",type=float,help="minimum depth",default=0.1)
        self.parser.add_argument("--max_depth",type=float,help="maximum depth",default=200.0)
        self.parser.add_argument("--frame_ids",nargs="+",type=int,help="frames to load",default=[0, -1, 1])
        self.parser.add_argument("--weights_init", type=str, default="pretrained", choices=["pretrained", "scratch"])

        # OPTIMIZATION optionsF
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=6)
        self.parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=6)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",type=str,help="name of model to load",)
        self.parser.add_argument("--models_to_load__",nargs="+",type=str,help="models to load",default=["encoder", "depth", "pose_encoder", "pose"])

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    options = val_Options()
    args = options.parse()
    json_file_list = ['./datasets/new_mask_data_Cam_01.json',
                      './datasets/new_mask_data_Cam_05.json',
                      './datasets/new_mask_data_Cam_06.json',
                      './datasets/new_mask_data_Cam_07.json',
                      './datasets/new_mask_data_Cam_08.json',
                      './datasets/new_mask_data_Cam_09.json'
                      ]
    if args.depth_code and args.code_num>1:
        args.depth_metric_names=["init/abs_rel", "init/sq_rel", "init/rms", "init/log_rms", "init/a1", "init/a2", "init/a3", "init/depth_con"]
        for refine_time_i in range(args.refine_times if not args.GRU else 1):
            args.depth_metric_names.append('refine_'+str(refine_time_i)+'/abs_rel')
            args.depth_metric_names.append('refine_'+str(refine_time_i)+'/sq_rel')
            args.depth_metric_names.append('refine_'+str(refine_time_i)+'/rms')
            args.depth_metric_names.append('refine_'+str(refine_time_i)+'/log_rms')
            args.depth_metric_names.append('refine_'+str(refine_time_i)+'/a1')
            args.depth_metric_names.append('refine_'+str(refine_time_i)+'/a2')
            args.depth_metric_names.append('refine_'+str(refine_time_i)+'/a3')
            args.depth_metric_names.append('refine_'+str(refine_time_i)+'/depth_con')

    tmp_result = open('./tmp/tmp_result.txt','w')
    tmp_result.write(args.load_weights_folder+'\n')
    real_refine_times = 1
    total_abs_rel = [0.] * (real_refine_times+1)
    total_depth_con = [0.] * (real_refine_times+1)

    for json_file_i in json_file_list:
        args.json_file_val = json_file_i
        train_dataset = DDAD_RAWDataset(args, 4, is_train=True)
        val_dataset = DDAD_RAWDataset(args, 4, is_train=False)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=10,drop_last=True)
        models = {}
        models["encoder"] = networks.ResnetEncoder(
            args.num_layers, args.weights_init == "pretrained")
        models["encoder"] = models["encoder"].cuda()

        models["depth"] = networks.DepthDecoder(
            models["encoder"].num_ch_enc, args.scales,num_output_channels=32)
        models["depth"] = models["depth"].cuda()

        if args.code_num > 1 and args.depth_code:
            if args.GRU:
                models['GRU'] = networks.Refine_GRU(models['encoder'].num_ch_enc,B=args.batch_size).cuda()
            else:
                for refine_time_i in range(args.refine_times):
                    models['refine_net_'+str(refine_time_i)] = networks.weight_net_for_visualize(models["encoder"].num_ch_enc,B=args.batch_size).cuda()

        if args.ddp:
            for k,v in models.items():
                pass
        models = load_model(args, models)

        tmp_result.write(json_file_i+'\n')
        print('json_file=', json_file_i)
        error = val(args, models, None, val_loader,val_whole=False)
        for i in range(len(args.depth_metric_names)):
            print(args.depth_metric_names[i],error.avg[i])
            tmp_result.write(args.depth_metric_names[i]+' '+str(error.avg[i])+'\n')
        for i in range(real_refine_times+1):
            total_abs_rel[i] += error.avg[i*8]
            total_depth_con[i] += error.avg[i*8+7]
        tmp_result.write('\n')
    for i in range(real_refine_times+1):
        tmp_result.write('total '+args.depth_metric_names[i*8]+'='+str(total_abs_rel[i]/6.)+'\n')
    for i in range(real_refine_times + 1):
        tmp_result.write('total '+args.depth_metric_names[i*8+7]+'='+str(total_depth_con[i]/6.)+'\n')
