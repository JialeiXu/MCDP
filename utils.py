from Camera import Project_depth,BackprojectDepth
from networks.layers import disp_to_depth
import cv2
import torch,time
import numpy as np
import torch.nn.functional as F
import os,wandb
import hashlib
import zipfile
from six.moves import urllib
import json,random
from networks.layers import compute_depth_errors
import shutil
def set_eval(models):
    """Convert all models to testing/evaluation mode
    """
    for m in models.values():
        m.eval()
    return models

def set_train(models):
    """Convert all models to training mode
    """
    for m in models.values():
        m.train()
    return models

def compute_depth_losses(args, inputs, outputs,losses,disp_type,refine_time=None):

    camera='f'

    add_name = '' if disp_type=='init' else '_refine_'+str(refine_time)
    disp = outputs[('disp'+add_name, camera, 0)]
    _, depth = disp_to_depth(disp, args.min_depth, args.max_depth)
    outputs[("depth"+add_name, camera, 0)] = depth

    depth_pred_f = depth
    depth_pred_f = torch.clamp(F.interpolate(
        depth_pred_f, [args.origin_size[0],args.origin_size[1]], mode="bilinear", align_corners=False), 1e-3, args.max_depth) # my edit 80
    depth_pred = depth_pred_f.detach()

    depth_gt = inputs["depth_gt", camera, 0].clamp(max=args.max_depth)
    mask = depth_gt > 0

    if ('self_mask',camera,0) in inputs.keys() :
        self_mask = inputs['self_mask',camera,0]
        #self_mask = F.interpolate(self_mask,[1216, 1936],mode='bilinear',align_corners=False)
        self_mask = F.interpolate(self_mask,[args.origin_size[0],args.origin_size[1]],mode='bilinear',align_corners=False)
        self_mask = self_mask > 0
        mask = torch.logical_and(mask,self_mask)
    ###
    #depth_gt = depth_gt[mask]
    #depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt[mask]) / torch.median(depth_pred[mask])

    depth_pred = torch.clamp(depth_pred, min=1e-3, max=args.max_depth)
    depth_l_scale = output2depth_use_scale(args,inputs,outputs,'l',disp_type)

    if args.error_depth_con:
        B, _, H, W = inputs['color', 'f', 0, 0].shape
        depth2point = BackprojectDepth(B, H, W).cuda()
        point2depth = Project_depth(B, H, W)
        cam_points_l = depth2point(depth_l_scale, inputs['inv_K', 'l',0, 0], inputs['extrinsics', 'l',0].float())

        warp_depths_l = point2depth(cam_points_l, inputs['K', 'f',0, 0], inputs['extrinsics_inv', 'f', 0].float())
        warp_depth = warp_depths_l #+ warp_depths_r
    warp_depth = F.interpolate(warp_depth,[args.origin_size[0],args.origin_size[1]],mode='bilinear',align_corners=False)

    depth_errors = compute_depth_errors(depth_gt, depth_pred, mask, warp_depth)


    #for i, metric in enumerate(args.depth_metric_names):
    for i in range(8):
        losses.append(np.array(depth_errors[i].cpu()))
    return losses


def readlines_len(filename):
    """Read all the lines in a text file and return as a list
    """
    if filename=="train":
        lines = json.load(open(args.train_json))['train']
    else:
        lines = json.load(open('datasets/new_data.json'))['val'] # datasets/new_data_Cam_01.json
    return lines

    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
    }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def log_time__(args, batch_idx, duration, loss):# origin
    """Print a logging statement to the terminal
    """
    samples_per_sec = args.batch_size / duration
    time_sofar = time.time() - args.start_time
    training_time_left = (args.num_total_steps / args.step - 1.0) * time_sofar if args.step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                   " | time elapsed: {} | time left: {}"
    print(print_string.format(args.epoch, batch_idx, samples_per_sec,
                              sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

def log_time(args,writers,cost_time):
    if args.tensorboardX:
        writers.add_scalar('time/10 step', cost_time, args.step)
    if args.wandb:
        wandb.log({'time/10 step': cost_time}, step=args.step)
    time_sofar = time.time() - args.start_time
    training_time_left = (args.num_total_steps / args.step - 1.0) * time_sofar if args.step > 0 else 0
    print('train epoch=%d, step=%d, spend %.2f s. left time:%.2f h' % (
        args.epoch, args.step, cost_time, training_time_left / 3600.))
    return time.time()

def log(args, inputs, outputs, losses,writer,val_whole):
    """Write an event to the tensorboard events file
    """
    if args.wandb:
        wandb.log({'Epoch':args.epoch},step=args.step)
    if args.tensorboardX:
        writer.add_scalar('Epoch',args.epoch,args.step)
    for i in range(len(losses)):
        if args.tensorboardX:
            writer.add_scalar("{}".format(args.depth_metric_names[i]), losses[i], args.step)
        if args.wandb:
            wandb.log({"{}".format(args.depth_metric_names[i]): losses[i]},step=args.step)
    if args.tensorboardX:
        for j in range(min(4, args.batch_size)):  # write a maxmimum of four images
            s = 0 #scale
            frame_id = args.frame_ids[0]
            #for frame_id in args.frame_ids: #log multi frame img, will be used
            writer.add_image(
                "color_{}_{}/{}".format(frame_id, s, j),
                inputs[("color", 'f', 0, s)][j].data, args.step)
            '''
                #log warp color images
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, args.step)
            '''
            writer.add_image(
                "disp_{}/{}".format(s, j),
                normalize_image(outputs[("disp",'f',  s)][j]), args.step)
            if args.depth_code and args.code_num > 1:
                if args.GRU:
                    writer.add_image(
                        "disp_refine_{}/{}".format(s, j),
                        normalize_image(outputs[("disp_refine_"+str(0), 'f', s)][j]), args.step)
                else:
                    for refine_time_i in range(args.refine_times):
                        writer.add_image(
                            "disp_refine_{}/{}".format(s, j),
                            normalize_image(outputs[("disp_refine_"+str(refine_time_i), 'f', s)][j]), args.step)

            '''
            #log identity mask
            writer.add_image(
                "automask_{}/{}".format(s, j),
                outputs["identity_selection/{}".format(s)][j][None, ...], args.step)
            '''

def log_train(args,losses,writer):
    for k,v in losses.items():
        if args.tensorboardX:
            writer.add_scalar(k, v, args.step)
        if args.wandb:
            wandb.log({k:v.cpu()},step=args.step)

'''
def load_model(self):
    """Load model(s) from disk
    """
    self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

    assert os.path.isdir(self.opt.load_weights_folder), \
        "Cannot find folder {}".format(self.opt.load_weights_folder)
    print("loading model from folder {}".format(self.opt.load_weights_folder))

    for n in self.opt.models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
        model_dict = self.models[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.models[n].load_state_dict(model_dict)

    # loading adam state
    optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
    if os.path.isfile(optimizer_load_path):
        print("Loading Adam weights")
        optimizer_dict = torch.load(optimizer_load_path)
        self.model_optimizer.load_state_dict(optimizer_dict)
    else:
        print("Cannot find Adam weights so Adam is randomly initialized")
'''

def save_model(args,models,last=None):
    """Save model weights to disk
    """
    if last!=None:
        save_folder = os.path.join(args.log_path, "models",'last')
    else:
        save_folder = os.path.join(args.log_path, "models")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()
        to_save['epoch'] = args.epoch
        torch.save(to_save, save_path)
        #if model_name == 'encoder':
            # save the sizes - these are needed at prediction time
            #to_save['height'] = self.opt.height
            #to_save['width'] = self.opt.width
            #to_save['use_stereo'] = self.opt.use_stereo
    #save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    #torch.save(self.model_optimizer.state_dict(), save_path)


class AverageMeter():
    def __init__(self,i=1,precision=3):
        self.meters=i
        self.precision = precision
        self.reset(self.meters)

    def reset(self,i):
        self.val=[0]*i
        self.avg=[0]*i
        self.sum=[0]*i
        self.count=0

    def update(self,val,n=1):
        if not isinstance(val,list):
            val=[val]
        assert (len(val)==self.meters)
        self.count+=n
        for i,v in enumerate(val):
            self.val[i]=v
            self.sum[i]+=v*n
            self.avg[i]=self.sum[i]/self.count

'''
def depth_color(img,args=None, color = cv2.COLORMAP_JET):
    if args.gpu!=None:
        img = img.cpu()
    img = img.squeeze().squeeze()
    invalid = img == 0.
    img = (img-img.min())/(img.max()-img.min())*255
    img = img.numpy().astype(np.uint8)
    #img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
    img = cv2.applyColorMap(img,color)
    img[:,:,0][invalid]=0
    img[:,:,1][invalid]=0
    img[:,:,2][invalid]=0
    return img
'''

def load_model(args,models):
    for k,v in models.items():
        path = os.path.join(args.load_weights_folder,'{}.pth'.format(k))
        model_dict = v.state_dict()
        pretrain_dict = torch.load(path)
        pretrain_dict = {k1:v1 for k1,v1 in pretrain_dict.items() if k1 in model_dict}
        model_dict.update(pretrain_dict)
        models[k].load_state_dict(model_dict)
    return models

def save_img_and_depth(args,input,output):
    rgb = F.interpolate(input['color',0,0],input["depth_gt",0].shape[-2:])[0]
    #rgb = input['color',0,0][0].permute(1,2,0).cpu().numpy()*255.
    rgb = rgb.permute(1,2,0).cpu().numpy()*255.
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    depth = output['depth',0,0]#[0]
    depth = F.interpolate(depth,input["depth_gt",0].shape[-2:])[0]
    depth = depth_color(depth)

    gt =  input["depth_gt",0][0]
    gt = depth_color(gt)
    shape = depth.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gt[i][j][0] == 0 and gt[i][j][1] == 0 and gt[i][j][2] == 0:
                for k in range(shape[2]):
                    gt[i][j][k]=rgb[i][j][k]
    img = np.vstack([rgb,gt,depth])
    #img = np.vstack([rgb,depth])
    return img

def save_rgb(img):
    img = img[0:1,:,:,:]
    img = img.squeeze().permute(1,2,0).cpu().numpy()*255
    img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR)
    return img

def depth_color(img,color=cv2.COLORMAP_JET,inverse=False):
    if inverse:
        img = img.max()-img

    #print(img.device,type(img.device))
    #if args.gpu!=None:

    img = img.cpu()
    img = img.squeeze().squeeze()
    invalid = img == 0.
    img = (img-img.min())/(img.max()-img.min())*255
    img = img.numpy().astype(np.uint8)
    img = cv2.applyColorMap(img,color)
    img[:,:,0][invalid]=0
    img[:,:,1][invalid]=0
    img[:,:,2][invalid]=0
    return img

def depth_color_edit(img,color=cv2.COLORMAP_JET,inverse=False):
    if inverse:
        img = img.max()-img

    #print(img.device,type(img.device))
    #if args.gpu!=None:

    img = img.cpu()
    img = img.squeeze().squeeze()
    invalid = img == 0.
    img[invalid]=img.max()
    img=1/img
    img = ((img-img.min())/(img.max()-img.min())*300).clamp(max=255)
    #img = 255-img
    img = img.numpy().astype(np.uint8)
    img = cv2.applyColorMap(img,color)
    img[:,:,0][invalid]=0
    img[:,:,1][invalid]=0
    img[:,:,2][invalid]=0
    return img



def output2depth_use_scale(args,sample,outputs,camera,disp_type,scale=0,refine_time=0):
    #args.height,args.width =1216, 1936

    disp_name = 'disp' if disp_type=='init' or args.code_num==1 else 'disp_refine_'+str(refine_time)

    disp = outputs[(disp_name, camera, scale)]
    disp = F.interpolate(
        disp, [args.height, args.width], mode="bilinear", align_corners=False)
    _, depth_pred = disp_to_depth(disp, args.min_depth, args.max_depth)

    depth_gt = sample["depth_gt",camera,0].clamp(max=args.max_depth)#my edit
    mask = depth_gt > 0
    if ('self_mask',camera,0) in sample.keys() :
        self_mask = sample['self_mask',camera,0]
        zero = torch.zeros_like(depth_pred)
        depth_pred = torch.where(self_mask < 0.1, zero, depth_pred)
                                            #[1216, 1936]
        self_mask = F.interpolate(self_mask,[args.origin_size[0],args.origin_size[1]],mode='bilinear',align_corners=False)
        self_mask = self_mask > 0

        mask = torch.logical_and(mask,self_mask)
    depth_gt = depth_gt[mask]
    depth_pred_s = F.interpolate(depth_pred,[args.origin_size[0],args.origin_size[1]],mode='bilinear',align_corners=False)
    depth_pred_s = depth_pred_s[mask]
    s = torch.median(depth_gt) / torch.median(depth_pred_s) + 1e-6
    return depth_pred * s

def output2depth(args, outputs):
    disp = outputs[("disp", 0, 0)]
    disp = F.interpolate(
        disp, [args.height, args.width], mode="bilinear", align_corners=False)
    _, depth_pred = disp_to_depth(disp, args.min_depth, args.max_depth)
    return depth_pred

def debug_print(args,str):
    print('debug: rank=',args.rank,str)

def wandb_save_code(args):
    if args.rank==0:
        code_artifact = wandb.Artifact('FSDE_L',type='code')

        for file in args.save_code_files:
            print('file=',file)
            code_artifact.add_file(file)
        wandb.run.log_artifact(code_artifact)
        print('########  code has been saved  ########')

#def my_loss_fun(x,alpha,scale,approximate=False,epslion=1e-6):
def robust_loss_fun(x,alpha,scale):
    if(alpha==0):
        squared_scaled_x = (x / scale)**2
        x = 0.5 * squared_scaled_x
        x = torch.as_tensor(x)
        return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))

def SaveCode_Local(args):
    log_path = args.log_path
    save_code_path = os.path.join(log_path,'code')
    if os.path.exists(save_code_path):
        shutil.rmtree(save_code_path)
    os.makedirs(save_code_path)
    #mkdir
    for file_i in args.makedirs:
        os.makedirs(save_code_path+'/'+file_i)
    #files
    for file_i in args.save_local_files:
        shutil.copy(file_i,save_code_path+'/'+file_i)
    #folders
    for file_i in args.save_local_folders:
        shutil.copytree(file_i,save_code_path+'/'+file_i)
    print('saved local code to',save_code_path)

def write_text2img(img,text):
    H,W,_ = img.shape #(H,W,3)
    cv2.putText(img,text,(W//2,H//2),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,0,0),1,cv2.LINE_AA)
    return img  #(H,W,3)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(0)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__=='__main__':
    x = torch.randn((5,5))
    y = torch.randn((5,5))
    l1 = torch.nn.L1Loss()
    l1_loss = l1(x,y)
    abs = torch.abs(x-y)
    b = robust_loss_fun(abs,0,0.1)
    print('abs',abs)
    print('robust_loss', b)
    print('l1_loss',l1_loss)
    print('robust_loss.mean',torch.mean(b))