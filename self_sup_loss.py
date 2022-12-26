import torch
from Camera import Project_depth,BackprojectDepth
from utils import output2depth_use_scale,robust_loss_fun
from networks.layers import *
def compute_reprojection_loss(args, pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if args.no_ssim:
        reprojection_loss = l1_loss
    else:
        ssim = SSIM()
        ssim_loss = ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def oneCamera_photometricLoss(args,inputs, outputs, losses, camera, disp_type,refine_time=None):
    if disp_type=='init': disp_name='disp'
    else: disp_name = 'disp_refine_'+str(refine_time)

    total_loss = 0
    for scale in args.scales:
        loss = 0
        reprojection_losses = []
        source_scale = 0
        disp = outputs[(disp_name, camera, scale)]
        color = inputs[("color",camera, 0, scale)]
        target = inputs[("color",camera, 0, source_scale)]

        self_mask = inputs['self_mask',camera,0]
        self_mask = torch.cat([self_mask,self_mask,self_mask],dim=1)
        zero = torch.zeros_like(target)
        target = torch.where(self_mask<0.1,zero,target)
        for frame_id in args.frame_ids[1:]:
            pred = outputs[("color_"+disp_name, camera, frame_id, scale)]
            pred = torch.where(self_mask<0.1,zero,pred)
            reprojection_losses.append(compute_reprojection_loss(args, pred, target))

        reprojection_loss = torch.cat(reprojection_losses, 1)

        identity_reprojection_losses = []
        for frame_id in args.frame_ids[1:]:
            pred = inputs[("color",camera, frame_id, source_scale)]
            pred = torch.where(self_mask<0.1,zero,pred)
            identity_reprojection_losses.append(
                compute_reprojection_loss(args, pred, target))

        identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)

        # add random numbers to break ties
        identity_reprojection_loss += torch.randn(
            identity_reprojection_loss.shape).cuda() * 0.00001

        combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

        to_optimise, idxs = torch.min(combined, dim=1)

        #outputs["identity_selection/{}".format(scale),camera] = (
        #            idxs > identity_reprojection_loss.shape[1] - 1).float()

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)
        #'''
        #debug
        if args.robust_loss:
            robust_loss = robust_loss_fun(to_optimise, 0, args.robust_loss_scale)
            loss = robust_loss.mean()
        #'''
        loss = loss + args.disparity_smoothness * smooth_loss / (2 ** scale)
        total_loss += loss
        #losses["loss/{}".format(scale)] = loss

    total_loss /= args.num_scales
    losses['loss_item/forward_back_self_supervised_loss_'+str(camera)+'_'+str(disp_name)] = total_loss
    losses["loss"] = losses['loss'] + total_loss
    #return losses, outputs
    return  losses

def cross_cam_photometric_loss(args, inputs, outputs, losses):
    """Compute the reprojection and smoothness losses for a minibatch
    """
    total_loss = 0

    for scale in args.scales:
        loss = 0
        reprojection_losses = []
        source_scale = 0

        disp = outputs[("disp", 0, scale)]
        color = inputs[("color", 0, scale)]
        target = inputs[("color", 0, source_scale)]
        self_mask = inputs['self_mask', 0, 0]
        self_mask = torch.cat([self_mask,self_mask,self_mask],dim=1)
        zero = torch.zeros_like(self_mask)

        warp_img_l = outputs[("color", 'l', scale)]
        warp_img_r = outputs[('color','r',scale)]
        warp_img_o = warp_img_l + warp_img_r

        #print('size=',self_mask.shape,zero.shape,warp_img_o.shape)

        warp_img_o = torch.where(self_mask<0.1, zero, warp_img_o)
        target = torch.where(self_mask<0.1,zero,target)
        target = torch.where(warp_img_o<0.1,zero,target)
        outputs['warp_img_o'] = warp_img_o
        loss += compute_reprojection_loss(args, warp_img_o, target).mean()

        zero = torch.zeros_like(disp)
        warp_img_o = F.interpolate(warp_img_o,disp.shape[-2:], mode="bilinear", align_corners=False)
        disp = torch.where(warp_img_o[:,0:1,:,:]<0.1, zero, disp)

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, warp_img_o)

        loss += args.disparity_smoothness * smooth_loss / (2 ** scale)
        total_loss += loss
        #losses["loss/{}".format(scale)] = loss

    total_loss /= args.num_scales
    losses['cross_camera_photometric'] = total_loss
    losses["loss"] += total_loss
    return losses, outputs


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def depth_consistency_loss(args,inputs,outputs,losses,disp_type,refine_time=None):
    add_name = '' if disp_type=='init' else '_refine_'+str(refine_time)
    l1_loss = torch.nn.L1Loss()

    for camera in ['l','f']:
        warp_depth = outputs['warp_depth'+add_name,camera]
        depth = outputs['depth_scale'+add_name, camera, 0]
        self_mask = inputs['self_mask',camera,0]
        zero = torch.zeros_like(depth)
        depth = torch.where(self_mask<0.1,zero,depth)
        mask = torch.logical_and(warp_depth > 0., depth > 0.)
        loss = l1_loss(warp_depth[mask], depth[mask]) * args.weight_depth_consistency_loss#.item()

        losses['loss_item/depth_consistency'+add_name+'_'+str(camera)] = loss
        losses['loss'] = losses['loss'] + loss

    return losses

if __name__=='__main__':
    depth_consistency_loss(1,1,1)
