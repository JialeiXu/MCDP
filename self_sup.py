from utils import output2depth_use_scale
from Camera import Project_depth,BackprojectDepth
import torch.nn.functional as F
import torch
from networks.layers import *
from Camera import transformation_from_parameters,BackprojectDepth,Project3D

def predict_poses(args, inputs,models,outputs, camera):

    pose_feats = {f_i: inputs["color_aug", camera, f_i, 0] for f_i in args.frame_ids} #[0, -1, 1])

    for f_i in args.frame_ids[1:]:
        if f_i != "s":
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_input = [models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = models["pose"](pose_input)
            outputs[("axisangle", camera, f_i)] = axisangle
            outputs[("translation", camera, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", camera, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
    return outputs

def generate_images_pred_forward_back(args, inputs, outputs, camera,disp_type,refine_time=None):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    if refine_time!=None: refine_time='_'+str(refine_time)
    if disp_type=='init': disp_name='disp'
    else: disp_name = 'disp_refine' + refine_time
    backproject_depth = {}
    project_3d = {}
    for scale in args.scales:
        h = args.height // (2 ** scale)
        w = args.width // (2 ** scale)

        backproject_depth[scale] = BackprojectDepth(args.batch_size, h, w).cuda()
        project_3d[scale] = Project3D(args.batch_size, h, w).cuda()

    for scale in args.scales:
        #if args.code_num > 1 and args.depth_code:
        #    disp = outputs[("disp_refine",camera, scale)]
        #else:
        disp = outputs[(disp_name, camera, scale)]
        disp = F.interpolate(
            disp, [args.height, args.width], mode="bilinear", align_corners=False)
        source_scale = 0

        _, depth = disp_to_depth(disp, args.min_depth, args.max_depth)

        outputs[("depth_"+disp_name, camera, scale)] = depth

        for i, frame_id in enumerate(args.frame_ids[1:]):#default=[0, -1, 1])

            T = outputs[("cam_T_cam", camera, frame_id)]

            cam_points = backproject_depth[source_scale](
                depth, inputs[("inv_K", camera,0,source_scale)])
            pix_coords = project_3d[source_scale](
                cam_points, inputs[("K",camera,0, source_scale)], T)

            #outputs[("sample", camera, frame_id, scale)] = pix_coords

            outputs[("color_"+disp_name, camera, frame_id, scale)] = F.grid_sample(
                inputs[("color",camera, frame_id, source_scale)],
                #outputs[("sample", camera, frame_id, scale)],
                pix_coords,
                padding_mode="border")

            #outputs[("color_identity",camera, frame_id, scale)] = \
            #    inputs[("color", camera, frame_id, source_scale)]
    return outputs

def generate_images_pred_l_r(args, inputs, outputs):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    backproject_depth = {}
    project_3d = {}
    for scale in args.scales:
        h = args.height // (2 ** scale)
        w = args.width // (2 ** scale)

        backproject_depth[scale] = BackprojectDepth(args.batch_size, h, w)
        backproject_depth[scale] = backproject_depth[scale].cuda()

        project_3d[scale] = Project3D(args.batch_size, h, w)
        project_3d[scale] = project_3d[scale].cuda()

    ###add mask
    color_l = inputs['color','l',0]
    self_mask = inputs['self_mask','l',0]
    zero = torch.zeros_like(color_l)
    color_l = torch.where(self_mask<0.1,zero,color_l)
    color_r = inputs['color','r',0]
    self_mask = inputs['self_mask','r',0]
    color_r = torch.where(self_mask<0.1,zero,color_r)


    for scale in args.scales:
        depth = output2depth_use_scale(args,inputs,outputs,frame=0,scale=scale)
        #disp = outputs[("disp", 0, scale)]
        #disp = F.interpolate(
        #    disp, [args.height, args.width], mode="bilinear", align_corners=False)
        source_scale = 0

        #_, depth = disp_to_depth(disp, args.min_depth, args.max_depth)

        outputs[("depth_scale", 0, scale)] = depth
        cam_points = backproject_depth[source_scale](
            depth, inputs[("inv_K", 0, source_scale)], inputs['extrinsics',0].float())#, source_scale)
        pix_coords_l = project_3d[source_scale](
            cam_points, inputs[("K",'l', source_scale)], inputs['extrinsics_inv','l'].float())
        pix_coords_r = project_3d[source_scale](
            cam_points, inputs[("K", 'r', source_scale)], inputs['extrinsics_inv','r'].float())

        outputs[("color", 'l', scale)] = F.grid_sample(
            color_l, pix_coords_l, padding_mode='zeros', align_corners=True)
        outputs[("color", 'r', scale)] = F.grid_sample(
            color_r, pix_coords_r, padding_mode='zeros', align_corners=True)

    return outputs

def generate_cross_camera_project_depth(args,inputs,outputs,disp_type,refine_time=None):
    add_name = '' if disp_type=='init' else '_refine_'+str(refine_time)
    for camera in ['l','f']:
        outputs['depth_scale'+add_name,camera,0] = output2depth_use_scale(args,inputs,outputs,camera,disp_type,refine_time=refine_time)
    self_mask = inputs['self_mask','l',0]
    zero = torch.zeros_like(outputs['depth_scale'+add_name,camera,0])
    outputs['depth_scale'+add_name,'l',0] = torch.where(self_mask<0.1,zero,outputs['depth_scale'+add_name,'l',0])
    self_mask = inputs['self_mask','f',0]
    outputs['depth_scale'+add_name,'f',0] = torch.where(self_mask<0.1,zero,outputs['depth_scale'+add_name,'f',0])

    B,_,H,W = inputs['color','f',0,0].shape
    depth2point = BackprojectDepth(B,H,W)
    point2depth = Project_depth(B,H,W)
    cam_points_l = depth2point(outputs['depth_scale'+add_name,'l',0], inputs['inv_K', 'l', 0, 0],
                               inputs['extrinsics', 'l', 0].float())
    cam_points_f = depth2point(outputs['depth_scale'+add_name,'f',0], inputs['inv_K', 'f', 0, 0],
                               inputs['extrinsics', 'f', 0].float())

    warp_depth_l2f = point2depth(cam_points_l, inputs['K', 'f',0, 0], inputs['extrinsics_inv', 'f',0].float())
    warp_depth_f2l = point2depth(cam_points_f, inputs['K', 'l',0 , 0], inputs['extrinsics_inv', 'l', 0].float())
    outputs['warp_depth'+add_name,'f'] = warp_depth_l2f
    outputs['warp_depth'+add_name,'l'] = warp_depth_f2l
    return outputs

