import torch.multiprocessing as mp
import torch.optim as optim
import wandb,time
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from options_DDAD import DDAD_Options
import networks
#import datasets
from datasets.dataload_DDAD import DDAD_RAWDataset
from networks.layers import *
from utils import *
import torch.distributed as dist
from self_sup_loss import oneCamera_photometricLoss,cross_cam_photometric_loss,depth_consistency_loss
from self_sup import predict_poses,generate_images_pred_forward_back,generate_images_pred_l_r,generate_cross_camera_project_depth
from eval import val

def train(gpu,ngpus_per_node,args):
    if args.ddp:
        args.rank = args.rank * args.ngpus_per_node + gpu
        args.gpu = gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size,rank=args.rank)
        args.batch_size = int(args.batch_size/ngpus_per_node)
        args.num_workers = int(args.num_workers/ngpus_per_node)
        print("==>gpu:",args.gpu,",rank:",args.rank,",batch_size:",args.batch_size,",workers:",args.num_workers)
        torch.cuda.set_device(args.gpu)

    setup_seed(args.seed + args.rank)
    args.log_path = os.path.join(args.log_dir, args.model_name)
    if args.rank==0 and args.wandb:
        wandb.init(name=args.model_name, project='baseline', entity="full-surround-depth-estimation",config=args,notes=args.notes)
        wandb_save_code(args)

    models = {}
    parameters_to_train = []
    args.num_scales = len(args.scales)
    args.num_pose_frames = 2

    #### model ####
    models["encoder"] = networks.ResnetEncoder(
        args.num_layers, args.weights_init == "pretrained")
    models["depth"] = networks.DepthDecoder(
        models["encoder"].num_ch_enc, args.scales,num_output_channels=args.code_num)
    models["pose_encoder"] = networks.ResnetEncoder(args.num_layers,
        args.weights_init == "pretrained", num_input_images= args.num_pose_frames)
    models["pose"] = networks.PoseDecoder(models["pose_encoder"].num_ch_enc,
        num_input_features=1, num_frames_to_predict_for=2)
    if args.code_num > 1 and args.depth_code:
        if args.GRU:
            models['GRU'] = networks.Refine_GRU(models['encoder'].num_ch_enc,B=args.batch_size)
        else:
            for refine_time_i in range(args.refine_times):
                models['refine_net_'+str(refine_time_i)] = networks.wight_net(args,models["encoder"].num_ch_enc,B=args.batch_size)

    if args.ddp:
        for k,v in models.items():
            models[k] = nn.SyncBatchNorm.convert_sync_batchnorm(models[k])
            models[k] = models[k].cuda(args.gpu)
            models[k] = torch.nn.parallel.DistributedDataParallel(models[k],device_ids=[args.gpu],
                                output_device=args.gpu,find_unused_parameters=True)
    else:
        for k,v in models.items():
            models[k] = v.cuda()

    ### optimizer ###
    for k,v in models.items():
        if args.ddp:
            parameters_to_train = parameters_to_train + list(models[k].module.parameters())
        else:
            parameters_to_train = parameters_to_train + list(models[k].parameters())

    model_optimizer = optim.Adam(parameters_to_train, args.learning_rate)
    model_lr_scheduler = optim.lr_scheduler.StepLR(
        model_optimizer, args.scheduler_step_size, 0.1)

    if args.load_weights_folder is not None and False:
        load_model()####### need edit but not now   now error
    if args.rank==0:
        print("Training model named:\n  ", args.model_name)
        print("Models and tensorboard events files are saved to:\n  ", args.log_dir)
        SaveCode_Local(args)

    # dataloader
    '''
    train_filenames = readlines_len(args,"train")
    num_train_samples = len(train_filenames)
    args.num_total_steps = num_train_samples // args.batch_size * args.num_epochs
    if args.ddp: args.num_total_steps/=2
    '''

    train_dataset = DDAD_RAWDataset(args,4,is_train=True)
    train_sampler = None
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,sampler=train_sampler)
    args.num_total_steps = len(train_loader) * args.num_epochs

    if len(args.json_file_val_list)>0:
        val_loader = []
        for json_file_val_i in args.json_file_val_list:
            args.json_file_val = json_file_val_i
            val_dataset = DDAD_RAWDataset(args,4,is_train=False)
            val_loader.append(DataLoader(
                val_dataset, batch_size=args.batch_size,shuffle=True,pin_memory=True, drop_last=True))


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
    ### log ###
    if args.rank==0 and args.tensorboardX:
        writers = SummaryWriter(os.path.join(args.log_path, 'tensorboard'))
    else: writers = None

    if args.rank==0:
        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))
    #train
    train_net(args,models,train_loader,val_loader,train_sampler,model_optimizer,model_lr_scheduler,writers)

def train_net(args,models,train_loader,val_loader,train_sampler,model_optimizer,model_lr_scheduler,writers):
    args.epoch, args.step = -1, 0
    args.start_time = time.time()
    step_time = time.time()
    if args.rank==0 and True:
        val(args, models, writers, val_loader, val_whole=False)

    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.num_epochs):
        models = set_train(models)
        args.epoch += 1
        model_lr_scheduler.step()
        if args.ddp:
            train_sampler.set_epoch(epoch)

        for batch_idx, inputs in enumerate(train_loader):
            losses = {}
            losses['loss'] = torch.zeros(1).cuda()
            outputs = {}
            for k,v in inputs.items():
                inputs[k]=v.cuda()
            cameras = ['l','f']

            for camera_i in cameras:
                features = models["encoder"](inputs["color_aug", camera_i,0,0])
                outputs.update( models["depth"](features,camera_i))

            #predict pose
            outputs = predict_poses(args, inputs, models, outputs, camera='f')

            ###depth code, refine
            if args.code_num > 1 and args.depth_code:
                if args.GRU:
                    outputs,losses = models['GRU'](args,inputs,outputs, losses,is_train=True)
                else:
                    for refine_time_i in range(args.refine_times):
                        outputs = models['refine_net_'+str(refine_time_i)](args, inputs, outputs,refine_time_i)

            #init photometric loss
            outputs = generate_images_pred_forward_back(args, inputs, outputs, 'f',disp_type='init')
            losses = oneCamera_photometricLoss(args, inputs, outputs, losses, 'f', disp_type='init')

            if args.code_num > 1 and args.depth_code:
                if not args.GRU:
                    for refine_time_i in range(args.refine_times):
                        outputs = generate_images_pred_forward_back(args, inputs, outputs, 'f','refine',refine_time_i)
                        losses.update(oneCamera_photometricLoss(args, inputs, outputs, losses, 'f', 'refine',refine_time_i))
            if args.cross_cam_photometric_loss:
                outputs = generate_images_pred_l_r(args,inputs,outputs)
                losses, outputs = cross_cam_photometric_loss(args,inputs,outputs,losses)
            if args.depth_consistency_loss :
                outputs = generate_cross_camera_project_depth(args,inputs,outputs,'init')
                losses = depth_consistency_loss(args,inputs,outputs,losses,'init')
                if args.code_num > 1 and args.depth_code:
                    if args.GRU:
                        refine_time_i = 0
                        outputs = generate_cross_camera_project_depth(args, inputs, outputs, 'refine',refine_time_i)
                        losses = depth_consistency_loss(args, inputs, outputs, losses, 'refine',refine_time_i)
                    else:
                        for refine_time_i in range(args.refine_times):
                            outputs = generate_cross_camera_project_depth(args, inputs, outputs, 'refine',refine_time_i)
                            losses = depth_consistency_loss(args, inputs, outputs, losses, 'refine',refine_time_i)
            model_optimizer.zero_grad()
            losses["loss"].backward()
            model_optimizer.step()

            # log
            if args.step%10==0 and args.rank==0:
                step_time = log_time(args, writers, time.time() - step_time)
            run_val = args.step % args.log_frequency == 0 and args.step!=0 and args.step > args.val_step
            if args.rank==0 and run_val:
                log_train(args,losses,writers)
                val(args,models,writers,val_loader,val_whole=False)
                models = set_train(models)
            args.step += 1

def save_opts():
    """Save options to disk so we know what we ran this experiment with
    """
    models_dir = os.path.join(self.log_path, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    to_save = self.opt.__dict__.copy()

    with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
        json.dump(to_save, f, indent=2)

def save_model(self):
    """Save model weights to disk
    """
    save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in self.models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()
        if model_name == 'encoder':
            # save the sizes - these are needed at prediction time
            to_save['height'] = self.opt.height
            to_save['width'] = self.opt.width
            to_save['use_stereo'] = self.opt.use_stereo
        torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(self.model_optimizer.state_dict(), save_path)

if __name__=='__main__':
    options = DDAD_Options()
    args = options.parse()
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.ddp:
        print("==>",'DDP')
        mp.set_start_method('forkserver')
        port = np.random.randint(10000,10300)
        nodes="127.0.0.1"
        args.dist_url = 'tcp://{}:{}'.format(nodes,port)
        args.dist_backend='nccl'
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(train,nprocs=args.ngpus_per_node,args=(args.ngpus_per_node,args))
    else:
        train(0,1,args)
