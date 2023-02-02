import os, time, torch, imageio, csv
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter
import utils, shutil
import model as architecture
import data.common as common
from data.dataloader import dataloader as DatasetLoader
from option import args
from data import data
import src.degradation as degradation
from tqdm import tqdm
from src.cal_complexity import profile_origin
import warnings
# from src.SSIM import SSIM
warnings.filterwarnings("ignore")
from src.img_postprocessing import USMSharp
from pytorch_wavelets import DWTForward, DWTInverse
# from src.bicubic import bicubic
def main():
    global opt, scaling
    opt = utils.print_args(args)
    opt.normalize_mean = torch.from_numpy(np.array([0.466, 0.448, 0.403])).float().view(1, 3, 1, 1).cuda()
    opt.normalize_std = torch.from_numpy(np.array([0.242, 0.234, 0.246])).float().view(1, 3, 1, 1).cuda()
    # if opt.n_colors == 3:
    #     if opt.data_train == 'DF2K':
    #         # DF2K data normalize
    #         normalize_mean = torch.from_numpy(np.array([0.466, 0.448, 0.403])).float().view(1, 3, 1, 1)
    #         normalize_std = torch.from_numpy(np.array([0.242, 0.234, 0.246])).float().view(1, 3, 1, 1)
    #     elif opt.data_train == 'ImageNet':
    #         # imagenet data normalize
    #         normalize_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().view(1, 3, 1, 1)
    #         normalize_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().view(1, 3, 1, 1)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print('===> Building SR_Model')
    print("===> Building model")
    model = {
        'SR': architecture.Generator(opt.n_colors, opt.net_type, opt.n_channels,
                                     opt.n_blocks, opt.n_units, opt.n_layers, opt.growth_rate,
                                     opt.act, opt.use_CoMo, opt.scale)
    }

    optimizer = {
        'SR': None
    }

    scheduler = {
        'SR': None
    }

    if opt.train.lower() == 'train':
        # model['SR'] = utils.load_checkpoint(opt.resume_SR, model['SR'], opt.cuda, opt.n_GPUs)
        if os.path.isfile(opt.resume_SR):
            print('===> Loading Checkpoint from {}'.format(opt.resume_SR))
            ckp = torch.load(opt.resume_SR)
            CKP_BEST = torch.load(opt.resume_SR)
            model['SR'].load_state_dict(ckp, strict=False)
        else:
            print('===> No Checkpoint in {}'.format(opt.resume_SR))
            CKP_BEST = None
            Avg_PSNR_BEST = 0

        print('===> Calculating NumParams & FLOPs')
        input = torch.FloatTensor(1, opt.n_colors, 480 // opt.scale, 360 // opt.scale)
        macs, params = profile_origin(model['SR'], inputs=(input,), verbose=False)
        print('-------------SR Model-------------')
        print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(params * 1e-3, macs * 1e-9, input[0].shape))

        torch.cuda.empty_cache()

        print("===> Setting GPU")
        for item in model:
            if opt.n_GPUs > 1 and opt.cuda:
                model[item] = torch.nn.DataParallel(model[item]).cuda()
                para = filter(lambda x: x.requires_grad, model[item].module.parameters())
            else:
                model[item] = model[item].cuda() if opt.cuda else model[item]
                para = filter(lambda x: x.requires_grad, model[item].parameters())
            optimizer[item] = opt.optimizer[item]([{'params': para, 'initial_lr': opt.lr[item]}], lr=opt.lr[item])
            if opt.n_epochs > opt.lr_gamma_1[item]:
                scheduler[item] = optim.lr_scheduler.StepLR(optimizer[item],
                                                            last_epoch=opt.start_epoch,
                                                            step_size=opt.lr_gamma_1[item],
                                                            gamma=opt.lr_gamma_2[item])
            else:
                scheduler[item] = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer[item],
                                                                                 last_epoch=opt.start_epoch,
                                                                                 T_0=opt.lr_gamma_1[item],
                                                                                 T_mult=1,
                                                                                 eta_min=opt.lr_gamma_2[item])

            model[item].train()
        start_epoch = opt.start_epoch if opt.start_epoch >= 0 else 0

        if os.path.exists(opt.model_path + '/' + 'runs'):
            shutil.rmtree(opt.model_path + '/' + 'runs')
        writer = SummaryWriter(opt.model_path + '/runs')

        print('===> Testing')
        for valid_set in opt.data_valid:
            print('\t on {} datasets'.format(valid_set))
            result_path = opt.model_path + '/Results/{}'.format(valid_set)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            PSNR, SSIM, Time = validation(valid_set, result_path, model)
            if valid_set == 'Urban100':
                Avg_PSNR_BEST = PSNR
            writer.add_scalar('PSNR/{}'.format(valid_set), PSNR, start_epoch)
            writer.add_scalar('SSIM/{}'.format(valid_set), SSIM, start_epoch)

        # # Avg_PSNR_BEST = 0
        print('===> Building Training dataloader')
        trainset = DatasetLoader(opt)
        train_dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                      num_workers=opt.threads, pin_memory=False)

        for epoch in range(start_epoch + 1, opt.n_epochs + 1):
            # if (epoch - 1) % 1 == 0:
            #     print('===> Building Training dataloader')
            #     trainset = DatasetLoader(opt)
            #     train_dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
            #                                   num_workers=opt.threads, pin_memory=False)

            torch.cuda.empty_cache()
            print('===> Training')
            train(train_dataloader, optimizer, model, epoch, writer)
            scheduler['SR'].step()

            torch.cuda.empty_cache()
            print('===> Testing')
            PSNR_ALL = {}
            SSIM_ALL = {}
            print('\t on {} datasets'.format(opt.data_valid[0]))
            result_path = opt.model_path + '/Results/{}'.format(opt.data_valid[0])
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            Avg_PSNR, SSIM, Time = validation(opt.data_valid[0], result_path, model)
            PSNR_ALL[opt.data_valid[0]] = Avg_PSNR
            SSIM_ALL[opt.data_valid[0]] = SSIM
            torch.cuda.empty_cache()

            if ((Avg_PSNR_BEST - Avg_PSNR) / Avg_PSNR_BEST > 0.005) or np.isnan(Avg_PSNR):
                model['SR'].load_state_dict(CKP_BEST)
                print('Poor Updating !!!')
            elif Avg_PSNR_BEST < Avg_PSNR:
                for valid_set in opt.data_valid[1:]:
                    print('\t on {} datasets'.format(valid_set))
                    result_path = opt.model_path + '/Results/{}'.format(valid_set)
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                    PSNR, SSIM, Time = validation(valid_set, result_path, model)
                    PSNR_ALL[valid_set] = PSNR
                    SSIM_ALL[valid_set] = SSIM
                torch.cuda.empty_cache()

                print('Best Updating !!!')
                Avg_PSNR_BEST = Avg_PSNR
                model_path= opt.model_path + '/Checkpoints/model_epoch_best.pth'
                torch.save(model['SR'].state_dict(), model_path)
                print('Checkpoint saved to {}'.format(model_path))
                CKP_BEST = torch.load(model_path)
                for valid_set in opt.data_valid:
                    writer.add_scalar('PSNR/{}'.format(valid_set), PSNR_ALL[valid_set], epoch)
                    writer.add_scalar('SSIM/{}'.format(valid_set), SSIM_ALL[valid_set], epoch)
            else:

                for valid_set in opt.data_valid[1:]:
                    print('\t on {} datasets'.format(valid_set))
                    result_path = opt.model_path + '/Results/{}'.format(valid_set)
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                    PSNR, SSIM, Time = validation(valid_set, result_path, model)
                    PSNR_ALL[valid_set] = PSNR
                    SSIM_ALL[valid_set] = SSIM
                torch.cuda.empty_cache()

                print('General Updating !!!')
                model_path = opt.model_path + '/Checkpoints/model_epoch_last.pth'
                torch.save(model['SR'].state_dict(), model_path)
                print('Checkpoint saved to {}'.format(model_path))
                # CKP_BEST = model['SR'].state_dict().copy()
                CKP_BEST = torch.load(model_path)
                for valid_set in opt.data_valid:
                    writer.add_scalar('PSNR/{}'.format(valid_set), PSNR_ALL[valid_set], epoch)
                    writer.add_scalar('SSIM/{}'.format(valid_set), SSIM_ALL[valid_set], epoch)
        writer.close()

    elif opt.train == 'Test':
        model['SR'] = utils.load_checkpoint(opt.model_path + '.pth', model['SR'], opt.cuda, opt.n_GPUs)

        print("===> Setting GPU")
        for item in model:
            if opt.n_GPUs > 1 and opt.cuda:
                model[item] = torch.nn.DataParallel(model[item]).cuda()
            else:
                model[item] = model[item].cuda() if opt.cuda else model[item]
            model[item].eval()

        for i in range(len(opt.data_valid)):
            valid_path = opt.dir_data + 'Test/' + opt.data_valid[i]
            with open(opt.model_path + '/SR Results/' + opt.data_valid[i] + '_x{:d}.csv'.format(opt.scale), 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(['image_name', 'PSNR', 'SSIM', 'Time'])
                validation(valid_path, model, scale=opt.scale, f_csv=f_csv)
        torch.cuda.empty_cache()
    else:
        raise InterruptedError


def train(training_dataloader, optimizer, model, epoch, writer):
    criterion_MAE = nn.L1Loss(reduction='mean').cuda()
    criterion_LInf = architecture.LInf_Loss(reduction='mean').cuda()
    # coef_RGB2Y = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1).cuda()
    # criterion_PSNR_Y = architecture.PSNRLoss(toY=True).cuda()
    # criterion_Char = architecture.L1_Charbonnier_loss().cuda()
    # criterion_MAPE = architecture.MAPELoss(reduction='mean').cuda()
    # criterion_SSIM = SSIM(size_average=False).cuda()
    # xfm = DWTForward(J=1, wave='haar', mode='zero').cuda()
    # ifm = DWTInverse(wave='haar', mode='zero').cuda()
    for item in model:
        model[item].train()
    # Bic_Resize = bicubic().cuda()
    scaler = torch.cuda.amp.GradScaler()
    lambda_ext = 0.02

    # sharpen = USMSharp(radius=21, sigma=2.0).cuda()

    with tqdm(total=len(training_dataloader), ncols=140) as pbar:
        for iteration, (LR_img, HR_img) in enumerate(training_dataloader):
            if HR_img.shape[0] == opt.batch_size:
                niter = (epoch - 1) * len(training_dataloader) + iteration

                HR_img = Variable(HR_img, volatile=False)
                LR_img = Variable(LR_img)

                if opt.cuda:
                    HR_img = HR_img.cuda()
                    LR_img = LR_img.cuda()
                HR_img = (HR_img - opt.normalize_mean) / opt.normalize_std
                LR_img = (LR_img - opt.normalize_mean) / opt.normalize_std

                optimizer['SR'].zero_grad()
                with torch.cuda.amp.autocast():
                    SR_img = model['SR'](LR_img)
                    # loss = -criterion_SSIM(SR_img, HR_img)
                    loss_basic = criterion_MAE(SR_img, HR_img)
                    # loss = criterion_MAE(torch.log(SR_img + 0.5),
                    #                         Variable(torch.log(HR_img + 0.5), requires_grad=False))
                    # sharpen = USMSharp(radius=21, sigma=np.random.uniform(1.0, 3.0)).cuda()
                    # loss_extension = criterion_MAE(sharpen(SR_img) - SR_img,
                    #                                Variable(sharpen(HR_img) - SR_img, requires_grad=False))
                    # loss_DN = criterion_PSNR(img_denoised, Variable(img_clean, requires_grad=False))
                    # loss_extension = 3 * criterion_MAE(torch.sum(SR_img * coef_RGB2Y, dim=1) / 255.,
                    #                                torch.sum(HR_img * coef_RGB2Y, dim=1) / 255.)
                    # x_L, x_H = xfm(SR_img)
                    loss_extension = criterion_LInf(SR_img, HR_img)
                    # loss_extension = 0

                scaler.scale(loss_basic + lambda_ext*loss_extension).backward()
                if opt.use_CoMo:
                    for group in optimizer['SR'].param_groups:
                        a = np.zeros(len(group["params"]))
                        i = 0
                        param_group_copy = ['' for _ in range(len(group["params"]))]
                        for param in group["params"]:
                            if param.grad != None:
                                if param.grad.shape != param.shape:
                                    param_group_copy[i] = param.grad.data
                                    ref_batch_grad = param.grad.data.view(opt.batch_size,
                                                                          param.data.shape[0], param.data.shape[1],
                                                                          param.data.shape[2], param.data.shape[3])
                                    param.grad.data = torch.mean(ref_batch_grad, dim=0)
                                    a[i] = 1
                            i += 1
                # scaler.unscale_(optimizer['SR'])
                # torch.nn.utils.clip_grad_value_(model['SR'].parameters(), 0.1)
                    scaler.step(optimizer['SR'])
                    scaler.update()

                    for group in optimizer['SR'].param_groups:
                        for i in range(len(a)):
                            if a[i] == 1:
                                para = group["params"][i]
                                para.grad.data = param_group_copy[i]
                else:
                    # scaler.unscale_(optimizer['SR'])
                    # torch.nn.utils.clip_grad_value_(model['SR'].parameters(), 0.1)
                    scaler.step(optimizer['SR'])
                    scaler.update()

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(Epoch=epoch,
                                 LeRate='{:.6f}'.format(optimizer['SR'].param_groups[0]['lr']),
                                 Loss='{:.3f}+{}*{:.3f}'.format(loss_basic, lambda_ext, loss_extension))

                if (iteration + 1) % 50 == 0:
                    if 'loss_basic' in locals().keys():
                        writer.add_scalar('Loss/loss_basic', loss_basic, niter)
                    if 'loss_extension' in locals().keys():
                        writer.add_scalar('Loss/loss_extension', loss_extension, niter)
                    torch.cuda.empty_cache()
                    if np.isnan(loss_basic.data.cpu().numpy()):
                        break

def validation(valid_set, result_path, model):
    for item in model:
        model[item].eval()
    count = 0
    Avg_PSNR = 0
    Avg_SSIM = 0
    Avg_Time = 0

    path_lr = opt.dir_data + '/Test/{}/LR_bicubic/X{}'.format(valid_set, opt.scale)
    path_hr = opt.dir_data + '/Test/{}/HR'.format(valid_set)
    file_lr = os.listdir(path_lr)
    file_hr = os.listdir(path_hr)
    file_lr.sort()
    file_hr.sort()
    length = file_lr.__len__()

    if opt.cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        Time = 0

    torch.cuda.empty_cache()
    with torch.no_grad():
        with tqdm(total=length, ncols=140) as pbar:
            for idx_img in range(length):
                torch.cuda.empty_cache()
                img_name, ext = os.path.splitext(file_lr[idx_img])
                LR_img = imageio.imread(os.path.join(path_lr, file_lr[idx_img]))
                HR_img = imageio.imread(os.path.join(path_hr, file_hr[idx_img]))

                LR_img, HR_img = common.set_channel([LR_img, HR_img], opt.n_colors)
                LR_img = common.np2Tensor(LR_img, opt.value_range)
                HR_img = common.np2Tensor(HR_img, opt.value_range)

                LR_img = Variable(LR_img).unsqueeze(0)

                if opt.cuda:
                    LR_img = LR_img.cuda()

                start.record()
                LR_img = (LR_img - opt.normalize_mean) / opt.normalize_std
                with torch.cuda.amp.autocast():
                    SR_img = model['SR'].forward(LR_img.half())
                SR_img = SR_img * opt.normalize_std + opt.normalize_mean
                end.record()
                torch.cuda.synchronize()
                Time = start.elapsed_time(end) * 1e-3
                SR_img = SR_img.data[0].cpu()

                PSNR = utils.calc_PSNR(SR_img, HR_img, opt.value_range, shave=opt.scale)
                SSIM = utils.calc_SSIM(SR_img, HR_img, opt.value_range, shave=opt.scale)

                Avg_PSNR += PSNR
                Avg_SSIM += SSIM
                Avg_Time += Time
                count = count + 1
                # if opt.n_colors > 1:
                #     SR_img = SR_img.mul(255).clamp(0, 255).round()
                #     SR_img = SR_img.numpy().astype(np.uint8)
                #     SR_img = SR_img.transpose((1, 2, 0))
                #     SR_img = Image.fromarray(SR_img)
                # else:
                #     SR_img = SR_img[0, :, :].mul(opt.value_range).clamp(0, opt.value_range).round().numpy().astype(
                #         np.uint8)
                #     SR_img = Image.fromarray(SR_img).convert('L')
                #
                # SR_img.save(result_path + '/' + img_name + '.png')

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(Eval='|{:.3f}|{:.5f}|'.format(Avg_PSNR / count, Avg_SSIM / count),
                                 Time='{:.3f}ms'.format(Avg_Time / count * 1000))

    return Avg_PSNR / count, Avg_SSIM / count, Avg_Time / count

if __name__ == '__main__':
    main()