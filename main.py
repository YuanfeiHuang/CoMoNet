import os, time, torch, imageio, csv
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter
import utils
import model as architecture
import data.common as common
from option import args
from data import data
import src.degradation as degradation
from tqdm import tqdm
from src.cal_complexity import profile_origin
import warnings
warnings.filterwarnings("ignore")

def main():
    global opt, normalize_mean, normalize_std, bicubic
    opt = utils.print_args(args)

    if opt.n_colors == 3:
        if opt.data_train == 'DF2K':
            # DF2K data normalize
            normalize_mean = torch.from_numpy(np.array([0.466, 0.448, 0.403])).float().view(1, 3, 1, 1)
            normalize_std = torch.from_numpy(np.array([0.242, 0.234, 0.246])).float().view(1, 3, 1, 1)
        elif opt.data_train == 'ImageNet':
            # imagenet data normalize
            normalize_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().view(1, 3, 1, 1)
            normalize_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().view(1, 3, 1, 1)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    bicubic = degradation.bicubic()
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        # os.environ['CUDA_VISIBLE_DEVICES'] = opt.GPU_ID
        bicubic = bicubic.cuda()
        normalize_mean, normalize_std = normalize_mean.cuda(), normalize_std.cuda()

    cudnn.benchmark = True

    print('===> Building SR_Model')
    print("===> Building model")
    model = {
        'SR': architecture.Generator(opt.n_colors, opt.n_channels, opt.n_blocks, opt.n_units, opt.growth_rate, opt.groups,
                                     opt.act, opt.use_Att, opt.scale)
    }

    optimizer = {
        'SR': None
    }

    scheduler = {
        'SR': None
    }

    print('===> Calculating NumParams & FLOPs')
    input = torch.FloatTensor(1, opt.n_colors, 480 // opt.scale, 360 // opt.scale)
    macs, params = profile_origin(model['SR'], inputs=(input,), verbose=False)
    print('-------------SR Model-------------')
    print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(params * 1e-3, macs * 1e-9, input[0].shape))

    torch.cuda.empty_cache()

    if opt.train == 'Train':
        model['SR'] = utils.load_checkpoint(opt.resume_SR, model['SR'], opt.cuda, opt.n_GPUs)
        if opt.start_epoch > 0:
            epoch_init = opt.start_epoch
        else:
            epoch_init = 1

        print("===> Setting GPU")
        for item in model:
            if opt.n_GPUs > 1 and opt.cuda:
                model[item] = torch.nn.DataParallel(model[item]).cuda()
                para = filter(lambda x: x.requires_grad, model[item].module.parameters())
            else:
                model[item] = model[item].cuda() if opt.cuda else model[item]
                para = filter(lambda x: x.requires_grad, model[item].parameters())
            optimizer[item] = optim.Adam(params=para, lr=opt.lr)
            scheduler[item] = optim.lr_scheduler.StepLR(optimizer[item],
                                                        step_size=opt.lr_step_size,
                                                        gamma=opt.lr_gamma)
            model[item].train()

        writer = SummaryWriter(opt.model_path + '/runs')

        print('===> Validation')
        for i in range(len(opt.data_valid)-4):
            valid_path = opt.dir_data + 'Test/' + opt.data_valid[i]
            validation(valid_path, model, scale=opt.scale, f_csv=None)

        print('===> Loading Training Dataset')
        train_dataloader = data(opt).get_loader()

        for epoch in range(epoch_init, opt.n_epochs + 1):
            print('===> Training')
            train(train_dataloader, optimizer, model, epoch, writer)
            utils.save_checkpoint(model['SR'], epoch, opt.model_path + '/SR Models')
            print('===> Validation')
            for i in range(len(opt.data_valid)):
                valid_path = opt.dir_data + 'Test/' + opt.data_valid[i]
                PSNR, SSIM = validation(valid_path, model, scale=opt.scale, f_csv=None)
                writer.add_scalar('Testing/PSNR_' + opt.data_valid[i], PSNR, epoch)
                writer.add_scalar('Testing/SSIM_' + opt.data_valid[i], SSIM, epoch)
            torch.cuda.empty_cache()
            scheduler['SR'].step()
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
    criterion_MAE =nn.L1Loss(reduction='mean').cuda()

    for item in model:
        model[item].train()
        
    prepro = degradation.bicubic()

    with tqdm(total=len(training_dataloader), ncols=140) as pbar:
        for iteration, HR_img in enumerate(training_dataloader):
            if HR_img.shape[0] == opt.batch_size:
                niter = (epoch - 1) * len(training_dataloader) + iteration

                HR_img = Variable(HR_img, volatile=False)
                LR_img = prepro(HR_img, [1 / opt.scale])
                LR_img = Variable(LR_img)

                if opt.cuda:
                    HR_img = HR_img.cuda()
                    LR_img = LR_img.cuda()

                LR_img = (LR_img - normalize_mean) / normalize_std
                HR_img = (HR_img - normalize_mean) / normalize_std

                optimizer['SR'].zero_grad()
                SR_img = model['SR'](LR_img)
                loss_SR = criterion_MAE(SR_img, HR_img)
                loss_SR.backward()

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

                optimizer['SR'].step()

                for group in optimizer['SR'].param_groups:
                    for i in range(len(a)):
                        if a[i] == 1:
                            para = group["params"][i]
                            para.grad.data = param_group_copy[i]

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(_E=epoch,
                                 _Lr=optimizer['SR'].param_groups[0]['lr'],
                                 l_SR='{:.3f}'.format(loss_SR) if 'loss_SR' in locals().keys() else '')

                if (iteration + 1) % 10 == 0:
                    if 'loss_SR' in locals().keys():
                        writer.add_scalar('Loss/loss_SR', loss_SR, niter)


def validation(valid_path, model, scale, f_csv):
    for item in model:
        model[item].eval()
    count = 0
    Avg_PSNR = 0
    Avg_SSIM = 0
    Avg_Time = 0
    file = os.listdir(valid_path)
    file.sort()
    length = file.__len__()

    prepro = degradation.bicubic()

    if opt.cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        Time = 0
    with torch.no_grad():
        with tqdm(total=length, ncols=140) as pbar:
            for idx_img in range(length):
                torch.cuda.empty_cache()
                img_name, ext = os.path.splitext(file[idx_img])
                HR_img = imageio.imread(valid_path + '/' + img_name + ext)
                HR_img = common.set_channel(HR_img, opt.n_colors)
                HR_img = common.np2Tensor(HR_img, opt.value_range)
                HR_img = Variable(HR_img).view(1, HR_img.shape[0], HR_img.shape[1], HR_img.shape[2])
                LR_img = prepro(HR_img, scale=[1 / scale])

                if opt.cuda:
                    HR_img = HR_img.cuda()
                    LR_img = LR_img.cuda()

                start.record()
                LR_img = (LR_img - normalize_mean) / normalize_std
                SR_img = model['SR'](LR_img)
                end.record()
                torch.cuda.synchronize()
                Time = start.elapsed_time(end) * 1e-3
                SR_img = SR_img * normalize_std + normalize_mean
                SR_img = SR_img.data[0].cpu()

                PSNR = utils.calc_PSNR(SR_img, HR_img.data[0].cpu(), opt.value_range, shave=scale)
                SSIM = utils.calc_SSIM(SR_img, HR_img.data[0].cpu(), opt.value_range, shave=scale)

                if f_csv:
                    f_csv.writerow([img_name, PSNR, SSIM, Time])

                Avg_PSNR += PSNR
                Avg_SSIM += SSIM
                Avg_Time += Time
                count = count + 1
                if opt.n_colors > 1:
                    SR_img = SR_img.mul(255).clamp(0, 255).round()
                    SR_img = SR_img.numpy().astype(np.uint8)
                    SR_img = SR_img.transpose((1, 2, 0))
                    SR_img = Image.fromarray(SR_img)
                else:
                    SR_img = SR_img[0, :, :].mul(opt.value_range).clamp(0, opt.value_range).round().numpy().astype(
                        np.uint8)
                    SR_img = Image.fromarray(SR_img).convert('L')

                SR_path = opt.model_path + '/SR Results/' + valid_path.split('Test/')[1] + '/x{:d}'.format(scale)
                if not os.path.exists(SR_path):
                    os.makedirs(SR_path)
                SR_img.save(SR_path + '/' + img_name + '.png')

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(Eval='|{:.2f}|{:.4f}|'.format(Avg_PSNR / count, Avg_SSIM / count),
                                 Deg='x{:d}'.format(scale),
                                 Time='{:.3f}ms'.format(Avg_Time / count * 1000))
    if f_csv:
        f_csv.writerow(['Avg', Avg_PSNR / count, Avg_SSIM / count, Avg_Time / count])

    return Avg_PSNR / count, Avg_SSIM / count

if __name__ == '__main__':
    main()
