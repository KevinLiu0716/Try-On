import shutil
from pathlib import Path

import cupy
import torch
import torchvision as tv
from thop import profile as ops_profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.viton_dataset import LoadVITONDataset
from pipelines import DMVTONPipeline
from opt.test_opt import TestOptions
from utils.general import Profile, print_log, warm_up
from utils.metrics import calculate_fid_given_paths, calculate_lpips_given_paths
from utils.torch_utils import select_device


import zmq
import sys
import cv2

import time
from PIL import Image
import torchvision.transforms as transforms


def get_transform(train, method=Image.BICUBIC, normalize=True):
    transform_list = []

    base = float(2**4)
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if train:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, 0)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    try:
        ow, oh = img.size  # PIL
    except Exception:
        oh, ow = img.shape  # numpy
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def run_test_pf(
    pipeline, device, img_dir, save_dir, log_path, save_img=True, frame=None
):
    metrics = {}

    # result_dir = Path(save_dir) / 'results'
    result_dir = Path(save_dir)
    tryon_dir = result_dir / 'tryon'
    # visualize_dir = result_dir / 'visualize'
    tryon_dir.mkdir(parents=True, exist_ok=True)
    # visualize_dir.mkdir(parents=True, exist_ok=True)

    # start_time = time.time()  # 记录开始时间
    # # Warm-up gpu
    # dummy_input = {
    #     'person': torch.randn(1, 3, 256, 192).to(device),
    #     'clothes': torch.randn(1, 3, 256, 192).to(device),
    #     'clothes_edge': torch.randn(1, 1, 256, 192).to(device),
    # }

    # end_time = time.time()  # 记录结束时间
    # elapsed_time = end_time - start_time  # 计算时间差
    # print(f"Warm-up gpu input:{elapsed_time} seconds")

    # start_time = time.time()  # 记录开始时间
    # with cupy.cuda.Device(int(device.split(':')[-1])):
    #     warm_up(pipeline, **dummy_input)
    # end_time = time.time()  # 记录结束时间
    # elapsed_time = end_time - start_time  # 计算时间差
    # print(f"Run Warm-up gpu :{elapsed_time} seconds")

    with torch.no_grad():
        start_time = time.time() * 1000
        count = 0
        seen, dt = 0, Profile(device=device)

        # Person image
        # img = Image.open(Path(self.dataroot) / f'{self.phase}_img' / im_name).convert('RGB')
        # img = img.resize((self.width, self.height))
        img = Image.open(Path("./screenshot/test_img").joinpath(frame)).convert('RGB')
        img_tensor = get_transform(train=True)(img)  # [-1,1]

        # Clothing image
        cloth = Image.open(Path("./screenshot/test_color").joinpath("000560_1.jpg")).convert('RGB')

        # cloth = cloth.resize((self.width, self.height))
        cloth_tensor = get_transform(train=True)(cloth)  # [-1,1]

        # Clothing edge
        # Clothing edge
        cloth_edge = Image.open(Path("./screenshot/test_edge").joinpath("000560_1.jpg")).convert('L')
        cloth_edge_tensor = get_transform(train=True, method=Image.NEAREST, normalize=False)(cloth_edge)  # [-1,1]



        # Prepare data
        real_image = img_tensor.unsqueeze(0).to(device)
        clothes = cloth_tensor.unsqueeze(0).to(device)
        edge = cloth_edge_tensor.unsqueeze(0).to(device)

        with cupy.cuda.Device(int(device.split(':')[-1])):
            with dt:
                p_tryon, warped_cloth = pipeline(real_image, clothes, edge, phase="test")

        # Save images
        p_name = frame
        img_path = tryon_dir / p_name

        tv.utils.save_image(
            p_tryon,
            tryon_dir / p_name,
            nrow=int(1),
            normalize=True,
            value_range=(-1, 1),
        )






        # start_for_time = time.time()  # 记录开始时间
        # for idx, data in enumerate(data_loader):
        #     count += 1  # 每次循环迭代，计数器加一
        #     start_infor_time = time.time()  # 记录开始时间
        #     # Prepare data
        #     real_image = data['image'].to(device)
        #     clothes = data['color'].to(device)
        #     edge = data['edge'].to(device)

        #     with cupy.cuda.Device(int(device.split(':')[-1])):
        #         with dt:
        #             p_tryon, warped_cloth = pipeline(real_image, clothes, edge, phase="test")

        #     seen += len(p_tryon)

        #     # Save images
        #     for j in range(len(data['p_name'])):
        #         p_name = data['p_name'][j]
        #         img_path = tryon_dir / p_name

        #         tv.utils.save_image(
        #             p_tryon[j],
        #             tryon_dir / p_name,
        #             nrow=int(1),
        #             normalize=True,
        #             value_range=(-1, 1),
        #         )
                
        #     #     # Display the result
        #     for img_path in data['p_name']:
        #         img_path = tryon_dir / img_path
        #         img = cv2.imread(str(img_path))
        #         cv2.imshow('Result', img)
        #         cv2.waitKey(1)
                
        #     end_infor_time = time.time()  # 记录结束时间
        #     elapsed_time = end_infor_time - start_infor_time  # 计算时间差
        #     print(f"Iteration {idx + 1} elapsed time: {elapsed_time} seconds")
        # # print(f"The for loop executed {count} times.")  # 打印循环执行次数
        # end_for_time = time.time()  # 记录结束时间
        # elapsed_time = end_for_time - start_for_time  # 计算时间差
        # print(f"For loop time: {elapsed_time} seconds")

    # fid = calculate_fid_given_paths(
    #     paths=[str(img_dir), str(tryon_dir)],
    #     batch_size=50,
    #     device=device,
    # )
    # lpips = calculate_lpips_given_paths(paths=[str(img_dir), str(tryon_dir)], device=device)

    # # FID
    # metrics['fid'] = fid
    # metrics['lpips'] = lpips

    # Speed
    # t = dt.t / 1 * 1e3  # speeds per image
    # metrics['fps'] = 1000 / t
    # print_log(
    #     log_path,
    #     f'Speed: %.1fms per image {real_image.size()}'
    #     % t,
    # )

    # # Memory
    # mem_params = sum([param.nelement()*param.element_size() for param in pipeline.parameters()])
    # mem_bufs = sum([buf.nelement()*buf.element_size() for buf in pipeline.buffers()])
    # metrics['mem'] = mem_params + mem_bufs # in bytes

    # ops, params = ops_profile(pipeline, (*dummy_input.values(), ), verbose=False)
    # metrics['ops'] = ops
    # metrics['params'] = params

    # # Log
    # metrics_str = 'Metric, {}'.format(', '.join([f'{k}: {v}' for k, v in metrics.items()]))
    # print_log(log_path, metrics_str)

    # Remove results if not save


    end_time = time.time() * 1000
    elapsed_time = end_time - start_time
    fps = 1000/elapsed_time
    print(f"tryon time: {elapsed_time} ms")
    print(f"tryon time: {fps} fps")


    if not save_img:
        shutil.rmtree(result_dir)
    else:
        print_log(log_path, f'Results are saved at {result_dir}')

    return metrics


def main(opt):
    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)
    log_path = Path(opt.save_dir) / 'log.txt'

    # Inference Pipeline
    pipeline = DMVTONPipeline(
        align_corners=opt.align_corners,
        checkpoints={
            'warp': opt.pf_warp_checkpoint,
            'gen': opt.pf_gen_checkpoint,
        },
    ).to(device)
    pipeline.eval()
    print_log(log_path, f'Load pretrained parser-free warp from {opt.pf_warp_checkpoint}')
    print_log(log_path, f'Load pretrained parser-free gen from {opt.pf_gen_checkpoint}')

    # # Dataloader
    # test_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192))
    # data_loader = DataLoader(
    #     test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers
    # )

    # print("save_dirsave_dirsave_dirsave_dirsave_dirsave_dirsave_dirsave_dir",opt. save_dir)


    # 初始化 OpenCV 視窗
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)


    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://localhost:5001")

    while True:
        start_time = time.time() * 1000  # 记录开始时间

        print("111111111111")
        response = socket.recv().decode("utf-8")
        # print(response)

        # 记录重新创建data_loader之前的时间点
        before_data_loader_time = time.time()

        # 重新创建data_loader
        # test_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192), frame=response)
        # data_loader = DataLoader(
        #     test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers
        # )

        # 记录重新创建data_loader之后，执行run_test_pf之前的时间点
        before_run_test_pf_time = time.time()

        # 考慮用multiprocess
        run_test_pf(
            pipeline=pipeline,
            # data_loader=data_loader,
            device=device,
            log_path=log_path,
            save_dir=opt.save_dir,
            img_dir=Path(opt.dataroot) / 'test_img',
            save_img=True,
            frame=response,
        )

        end_time = time.time() * 1000  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算整个循环的时间差
        # 顯示結果
        tryon_image = cv2.imread(str(Path(opt.save_dir) / 'tryon' / response))  # 讀取保存的 try-on 圖像

        # elapsed_time_text = f"{elapsed_time:.2f} ms"
        elapsed_time_text = f"{1000/elapsed_time:.2f} fps"
        # 在圖像上顯示文字
        cv2.putText(tryon_image, elapsed_time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imshow('Result', tryon_image)
        cv2.waitKey(1)  # 更新視窗

        
        # data_loader_time = before_run_test_pf_time - before_data_loader_time  # 计算重新创建data_loader的时间差
        # run_test_pf_time = end_time - before_run_test_pf_time  # 计算执行run_test_pf的时间差

        # print(f"Time spent on recreating data_loader: {data_loader_time} seconds")  # 打印重新创建data_loader的时间差
        # print(f"Time spent on running run_test_pf: {run_test_pf_time} seconds")  # 打印执行run_test_pf的时间差
        # print(f"Total elapsed time for this iteration: {elapsed_time} seconds")  # 打印整个循环的时间差


if __name__ == "__main__":
    opt = TestOptions().parse_opt()
    main(opt)