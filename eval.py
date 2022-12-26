import torch.nn
from utils import *

def val(args,eval_models,writers,val_loader,val_whole):

    models = {}
    for k, v in eval_models.items():
        models[k] = v.module

    error = AverageMeter(i=len(args.depth_metric_names))
    models = set_eval(models)
    for val_loader_i in val_loader:
        len_val = len(val_loader_i)
        for batch_i,inputs in enumerate(val_loader_i):
            if (not val_whole) and batch_i%5!=0: continue
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
                        outputs, _  = models['GRU'](args,inputs,outputs, None, is_train=False)
                else:
                    for refine_time_i in range(args.refine_times):
                        with torch.no_grad():
                            outputs = models['refine_net_'+str(refine_time_i)](args,inputs,outputs,refine_time_i)

            losses = []
            losses = compute_depth_losses(args, inputs, outputs, losses, 'init')
            if args.depth_code and args.code_num > 1:
                if args.GRU:
                    losses = compute_depth_losses(args, inputs, outputs,losses, 'refine',refine_time=0)
                else:
                    for refine_time_i in range(args.refine_times):
                        losses = compute_depth_losses(args, inputs, outputs,losses, 'refine',refine_time_i)
            error.update(losses,n=inputs['color','f',0,0].shape[0])

            if (not val_whole) :
                break
    if writers!=None:
        save_model(args,models,'last')
        log(args, inputs, outputs, losses, writers, val_whole)
        num = 8 if args.code_num>1 and args.depth_code else 0
        if error.avg[num] < args.best_rel:
            args.best_rel = error.avg[num]
            save_model(args,models)
    return error
