# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import platform
import yaml
import time
import datetime
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import paddle.nn.functional as F
from ppocr.utils.stats import TrainingStats
from ppocr.utils.save_load import save_model
from ppocr.utils.utility import print_dict, AverageMeter
from ppocr.utils.logging import get_logger
from ppocr.utils.loggers import VDLLogger, WandbLogger, Loggers
from ppocr.utils import profiler
from ppocr.data import build_dataloader
from ppocr.data.imaug import transform, create_operators
from ppocr.postprocess.db_postprocess import DBPostProcess
class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config",default='/tmp/Experiments/PaddleOCR/domain/domain_config/det_r50_dbnet++_icdar13_new_mp_500epochs_baseline.yaml',help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

def generate_pseudo_label(data,post_result,name=None):
        # transforms = 
    # DetLabelEncode: null       标签是通过这几个生成的的的
    # - MakeShrinkMap: shrink_ratio: 0.4  min_text_size: 8
    # - MakeBorderMap: shrink_ratio: 0.4  thresh_min: 0.3
    transforms = list()     
    transforms.append({'MakeShrinkMap': {'shrink_ratio': 0.4, 'min_text_size': 8}})   
    transforms.append({'MakeBorderMap': {'shrink_ratio': 0.4, 'thresh_min': 0.3, 'thresh_max': 0.7}})
    # transforms.append({'KeepKeys': {'keep_keys': ['image','strong_image','threshold_map','threshold_mask','shrink_map','shrink_mask']}})  
    transforms.append({'KeepKeys': {'keep_keys': ['threshold_map','threshold_mask','shrink_map','shrink_mask']}})  
    ops = create_operators(transforms, None)
    # image = []
    # strong_image = []
    shrink_map = []
    shrink_mask = []
    border_map = []
    border_mask = []
    import cv2
    for i in range(len(post_result)):
        data1 = dict()
        boxes = post_result[i]['points']
        data1['image'] = data[0][i].transpose((1,2,0))
        data1['strong_image'] = data[1][i]
        texts = list()
        ignore_tags = list()
        for _ in range(len(boxes)):
            texts.append(' ')
            ignore_tags.append(False)
        data1['polys'] = boxes
        data1['texts'] = texts
        data1['ignore_tags'] = ignore_tags
        outs = transform(data1, ops)
        # 在这里展示一下 生成的几个图片
        # image.append(outs[0].transpose((2,0,1)).unsqueeze(0))
        # strong_image.append(outs[1].unsqueeze(0))
        border_map.append(paddle.to_tensor(outs[0]).unsqueeze(0))
        border_mask.append(paddle.to_tensor(outs[1]).unsqueeze(0))
        shrink_map.append(paddle.to_tensor(outs[2]).unsqueeze(0))
        shrink_mask.append(paddle.to_tensor(outs[3]).unsqueeze(0))
        
        # cv2.imwrite("/tmp/Experiments/PaddleOCR/border_map.jpg",outs[2] * 200)
        # cv2.imwrite("/tmp/Experiments/PaddleOCR/border_mask.jpg",outs[3] * 200)
        # cv2.imwrite("/tmp/Experiments/PaddleOCR/shrink_map.jpg",outs[4] * 200)
        # cv2.imwrite("/tmp/Experiments/PaddleOCR/shrink_mask.jpg",outs[5] * 200)


        # 2,3,4,5 to tensor
    # image =  paddle.concat(image,axis=0) 
    # strong_image =  paddle.concat(strong_image,axis=0)
    border_map =  paddle.concat(border_map,axis=0)
    border_mask =  paddle.concat(border_mask,axis=0)
    shrink_map =  paddle.concat(shrink_map,axis=0)
    shrink_mask =  paddle.concat(shrink_mask,axis=0)
    pseudo_label = [1]
    # # pseudo_label.append(image)
    # # pseudo_label.append(strong_image)
    pseudo_label.append(border_map)
    pseudo_label.append(border_mask)
    pseudo_label.append(shrink_map)
    pseudo_label.append(shrink_mask)
    return pseudo_label


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def check_device(use_gpu, use_xpu=False, use_npu=False, use_mlu=False):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config {} cannot be set as true while your paddle " \
          "is not compiled with {} ! \nPlease try: \n" \
          "\t1. Install paddlepaddle to run model on {} \n" \
          "\t2. Set {} as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and use_xpu:
            print("use_xpu and use_gpu can not both be ture.")
        if use_gpu and not paddle.is_compiled_with_cuda():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
        if use_xpu and not paddle.device.is_compiled_with_xpu():
            print(err.format("use_xpu", "xpu", "xpu", "use_xpu"))
            sys.exit(1)
        if use_npu and not paddle.device.is_compiled_with_npu():
            print(err.format("use_npu", "npu", "npu", "use_npu"))
            sys.exit(1)
        if use_mlu and not paddle.device.is_compiled_with_mlu():
            print(err.format("use_mlu", "mlu", "mlu", "use_mlu"))
            sys.exit(1)
    except Exception as e:
        pass


def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            if isinstance(preds[k], dict) or isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, list):
        for k in range(len(preds)):
            if isinstance(preds[k], dict):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, paddle.Tensor):
        preds = preds.astype(paddle.float32)
    return preds


def train(config,
          train_dataloader,
          target_dataloader,
          valid_dataloader,
          device,
          model,
          teacher_model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          log_writer=None,
          scaler=None,
          amp_level='O2',
          amp_custom_black_list=[]):
    cal_metric_during_train = config['Global'].get('cal_metric_during_train',
                                                   False)
    calc_epoch_interval = config['Global'].get('calc_epoch_interval', 1)
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']
    profiler_options = config['profiler_options']

    total_step =config['Global']['MAX_SREP']
    burn_up_step = config['Global']['BURN_UP_STEP']

    global_step = 0
    if 'global_step' in pre_best_model_dict:
        global_step = pre_best_model_dict['global_step']
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                'No Images in eval dataset, evaluation during training ' \
                'will be disabled'
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, " \
            "an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step))
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    model_average = False
    model.train()

    use_srn = config['Architecture']['algorithm'] == "SRN"
    extra_input_models = [
        "SRN", "NRTR", "SAR", "SEED", "SVTR", "SPIN", "VisionLAN",
        "RobustScanner", "RFL", 'DRRG'
    ]
    extra_input = False
    if config['Architecture']['algorithm'] == 'Distillation':
        for key in config['Architecture']["Models"]:
            extra_input = extra_input or config['Architecture']['Models'][key][
                'algorithm'] in extra_input_models
    else:
        extra_input = config['Architecture']['algorithm'] in extra_input_models
    try:
        model_type = config['Architecture']['model_type']
    except:
        model_type = None

    algorithm = config['Architecture']['algorithm']

    start_epoch = best_model_dict[
        'start_epoch'] if 'start_epoch' in best_model_dict else 1

    if start_epoch==1:
        start_step=1
    else:
        start_step = int(start_epoch * len(train_dataloader))
    
    total_samples = 0
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    reader_start = time.time()

    max_iter = len(train_dataloader) - 1 if platform.system(
    ) == "Windows" else len(train_dataloader)

    start_time = time.time()
    # train_dataloader_iter = iter(train_dataloader)
    # target_dataloader_iter = iter(target_dataloader)
    
    # # copy the student model to teacher model
    # student_model_dict = model.state_dict()
    # new_state_dict = {}
    # for key, value in student_model_dict.items():
    #     new_state_dict[key] = value
    # teacher_model.set_state_dict(new_state_dict)   
    
    # PostProcess = DBPostProcess(0.5)

    # cur_metric = eval(
    # model,
    # target_dataloader,
    # post_process_class,
    # eval_class,
    # model_type,
    # extra_input=extra_input,
    # scaler=scaler,
    # amp_level=amp_level,
    # amp_custom_black_list=amp_custom_black_list)
    # cur_metric_str = 'cur metric, {}'.format(', '.join(
    #     ['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
    # logger.info(cur_metric_str)


    # model.eval()
    # with paddle.no_grad():    
    #     for idx,(target_data,name) in enumerate(target_dataloader):
    #         preds = model(target_data[0])
    #         # img_list = [[736,1280,1,1]] 
    #         img_list = [[640,640,1,1],[640,640,1,1],[640,640,1,1],[640,640,1,1]]  
    #         post_result = post_process_class(preds, img_list)

    #             # 把预测的boxes画到 图上。
    #         # add code
    #         for i in range(len(post_result)):
    #             boxes = post_result[i]['points']
    #             import cv2
    #             img_name = os.path.join("/tmp/Experiments/PaddleOCR/checkpoints/imgs/",name[i])
    #             src_im = cv2.imread(img_name)
    #             for box in boxes:
    #                 box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
    #                 cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    #             save_path = os.path.join("/tmp/Experiments/PaddleOCR/checkpoints/imgs_poly/", name[i])
    #             cv2.imwrite(save_path, src_im)
    #             # add code
    #         # 生成伪标签
    #         pseudo_label = generate_pseudo_label(target_data,post_result,name)
          
    init_model = True
    for step in range(start_step, total_step + 1):
        profiler.add_profiler_step(profiler_options)
        lr = optimizer.get_lr()    
        try:
            train_data = next(train_dataloader_iter) 
        except:
            train_dataloader_iter = iter(train_dataloader) 
            train_data = next(train_dataloader_iter)
      
        
        s_domain_loss,t_domain_loss = 0,0
        source_data = paddle.concat((train_data[0],train_data[1]),axis=0)  # source weak and strong data into student model 
        source_preds = model(source_data)
        batch = []
        batch.append(source_data)
        batch.append(paddle.concat((train_data[2],train_data[2]),axis=0))
        batch.append(paddle.concat((train_data[3],train_data[3]),axis=0))
        batch.append(paddle.concat((train_data[4],train_data[4]),axis=0))
        batch.append(paddle.concat((train_data[5],train_data[5]),axis=0))
        # 去掉域分类器
        # source_domain_cls = source_preds['domain_cls']
        
        # source_domain_cls = paddle.reshape(source_domain_cls,shape=[source_domain_cls.shape[0],-1])
        # bce_loss1 = paddle.nn.BCELoss()
        # aa = paddle.zeros_like(source_domain_cls.shape,dtype=paddle.float32)  ?????
        # s_domain_loss = bce_loss1(source_domain_cls,aa)
        # aa = np.zeros(source_domain_cls.shape,dtype=np.float32)
        # s_domain_loss = bce_loss1(source_domain_cls,paddle.to_tensor(aa))
        
        #  new loss
        # s_gt = paddle.zeros_like(source_domain_cls)
        # s_domain_loss = F.binary_cross_entropy(source_domain_cls, s_gt, reduction='none')
        # mask = batch[2].unsqueeze(1)
        # s_domain_loss = (s_domain_loss * mask).sum() / (mask.sum() + 1e-6)  # loss * batch[2](label_threshold_mask)  means only focus on text region
        # s_domain_loss = paddle.mean(s_domain_loss)     
  
        source_loss = loss_class(source_preds, batch)
        source_avg_loss = source_loss['loss']
        print_loss = {}
        for key,value in source_loss.items():
            print_loss['s_'+key] = value
        # print_loss['s_cls_loss'] = s_domain_loss  
        if step < burn_up_step:
            
            # total_loss = source_avg_loss + s_domain_loss * 0.001
            total_loss = source_avg_loss


        else:
            if init_model:    # 由于波动比较大，个人认为这里应该按照最好的结果来进行初始化。。不然很浪费。那最后是不是要保存这个的最好结果呢？不是的。如果我用70的进行弱监督。
                # 使用best初始化
                params = paddle.load(config['Global']['save_model_dir']  + 'best_accuracy.pdparams')
                new_state_dict = {}
                for k1 in params.keys():
                    new_state_dict[k1] = params[k1]
                teacher_model.set_state_dict(new_state_dict)

                # 使用last初始化
                # copy the student model to teacher model
                # student_model_dict = model.state_dict()
                # new_state_dict = {}
                # for key, value in student_model_dict.items():
                #     new_state_dict[key] = value
                # teacher_model.set_state_dict(new_state_dict)
                init_model = False
                # 这里使用 教师模型预测一下。
            else :
                # EMA update the teacher model 
                student_model_dict = model.state_dict()
                new_state_dict = {}
                for key, value in teacher_model.state_dict().items():
                    new_state_dict[key] = (student_model_dict[key] * 0.005 ) + value * 0.995
                teacher_model.set_state_dict(new_state_dict)
                
            try:
                target_data = next(target_dataloader_iter) 
            except:
                target_dataloader_iter = iter(target_dataloader)
                target_data = next(target_dataloader_iter)
            
            teacher_model.eval()
            with paddle.no_grad():    
                preds = teacher_model(target_data[0])
            # model.eval()
            # with paddle.no_grad():    
            #     preds = model(target_data[0])
            # model.train()
            img_list = []
            for _ in range(8):
                img_list.append([640,640,1,1])
            
            # if step < total_step * 0.6:
            #     post_process =  DBPostProcess(0.4,0.5,1000,1.5)
            # else: post_process =  DBPostProcess(0.5,0.5,1000,1.5)
            # post_process_class
            post_result = post_process_class(preds, img_list)
            
            # 这里把预测的结果 画到裁剪的图像上。

            pseudo_label = generate_pseudo_label(target_data,post_result)

            # 弱增强生成为标签。强增强进入学生模型得到预测结果。算loss
            
            target_preds = model(target_data[1])  # strong target data into student model
            
            # target_domain_cls = target_preds['domain_cls']
            # target_domain_cls = paddle.reshape(target_domain_cls,shape=[target_domain_cls.shape[0],-1])
            # bce_loss2 = paddle.nn.BCELoss()
            # bb = np.ones(target_domain_cls.shape,dtype=np.float32)
            # t_domain_loss = bce_loss2(target_domain_cls,paddle.to_tensor(bb))

            # new domain cls
            # t_gt = paddle.ones_like(target_domain_cls)
            # t_domain_loss = F.binary_cross_entropy(target_domain_cls, t_gt, reduction='none')
            
            # # list(1,border_map,border_mask,shrink_map,shrink_mask)
            # mask = pseudo_label[2].unsqueeze(1)
            # t_domain_loss = (t_domain_loss * mask).sum() / (mask.sum() + 1e-6)  # loss * label_threshold_mask  means only focus on text region
            # t_domain_loss = paddle.mean(t_domain_loss)   

            target_loss = loss_class(target_preds, pseudo_label)
            target_avg_loss = target_loss['loss']
            

            for key,value in target_loss.items():
                print_loss['t_'+key] = value
            # print_loss['t_cls_loss'] = t_domain_loss
            # print("loss_shrink_maps:  {}, loss_threshold_maps: {}, loss_binary_maps: {}"\
            #       .format(np.array(print_loss['t_loss_shrink_maps']),np.array(print_loss["t_loss_threshold_maps"]),np.array(print_loss['t_loss_binary_maps'])))
            total_loss = source_avg_loss + target_avg_loss * (step / float(total_step)) * 0.5 
            # total_loss = source_avg_loss + target_avg_loss * 0.5 +  (s_domain_loss + t_domain_loss) * 0.005
            # total_loss = source_avg_loss + target_avg_loss * 0.1

        
        total_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
#         发生异常: AssertionError  明天再看。 模型的问题，换了一个模型就没报错了
# Optimizer set error, conv2d_0.w_0_velocity_0 should in state dict
        global_step += 1
        if not isinstance(lr_scheduler, float):
            lr_scheduler.step()

        # logger and visualdl
        stats = {k: v.numpy().mean() for k, v in print_loss.items()}
        stats['lr'] = lr
        train_stats.update(stats)
        
        if log_writer is not None and dist.get_rank() == 0:
            log_writer.log_metrics(metrics=train_stats.get(), prefix="TRAIN", step=global_step)
        
        if dist.get_rank() == 0 and (global_step > 0 and global_step % print_batch_step == 0):
            logs = train_stats.log()

            eta_sec = ((time.time() - start_time) / global_step) * (total_step - step)
            eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
            strs = 'step: [{}/{}], global_step: {}, {}, eta: {}'.format(
                step, total_step, global_step, logs, eta_sec_format)
            logger.info(strs)
    
        # eval
        if global_step > start_eval_step and \
                (global_step - start_eval_step) % eval_batch_step == 0 \
                and dist.get_rank() == 0:
            if model_average:
                Model_Average = paddle.incubate.optimizer.ModelAverage(
                    0.15,
                    parameters=model.parameters(),
                    min_average_window=10000,
                    max_average_window=15625)
                Model_Average.apply()
            cur_metric = eval(
                model,
                valid_dataloader,
                post_process_class,
                eval_class,
                model_type,
                extra_input=extra_input,
                scaler=scaler,
                amp_level=amp_level,
                amp_custom_black_list=amp_custom_black_list)
            cur_metric_str = 'cur metric, {}'.format(', '.join(
                ['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
            logger.info(cur_metric_str)

            # logger metric
            if log_writer is not None:
                log_writer.log_metrics(
                    metrics=cur_metric, prefix="EVAL", step=global_step)

            if cur_metric[main_indicator] >= best_model_dict[
                    main_indicator]:
                best_model_dict.update(cur_metric)
                best_model_dict['best_epoch'] = step//len(train_dataloader)
                save_model(
                    model,
                    optimizer,
                    save_model_dir,
                    logger,
                    config,
                    is_best=True,
                    prefix='best_accuracy',
                    best_model_dict=best_model_dict,
                    epoch=step//len(train_dataloader),
                    global_step=global_step)
            best_str = 'best metric, {}'.format(', '.join([
                '{}: {}'.format(k, v) for k, v in best_model_dict.items()
            ]))
            logger.info(best_str)
            # logger best metric
            if log_writer is not None:
                log_writer.log_metrics(
                    metrics={
                        "best_{}".format(main_indicator):
                        best_model_dict[main_indicator]
                    },
                    prefix="EVAL",
                    step=global_step)

                log_writer.log_model(
                    is_best=True,
                    prefix="best_accuracy",
                    metadata=best_model_dict)
    
        epoch = step / len(train_dataloader)

        if dist.get_rank() == 0  and epoch % 1 == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                config,
                is_best=False,
                prefix='latest',
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step)

            if log_writer is not None:
                log_writer.log_model(is_best=False, prefix="latest")

        if dist.get_rank() == 0 and epoch > 0 and epoch % save_epoch_step == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                config,
                is_best=False,
                prefix='iter_epoch_{}'.format(epoch),
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step)
            if log_writer is not None:
                log_writer.log_model(
                    is_best=False, prefix='iter_epoch_{}'.format(epoch))

    best_str = 'best metric, {}'.format(', '.join(
        ['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    if dist.get_rank() == 0 and log_writer is not None:
        log_writer.close()
    return


def eval(model,
         valid_dataloader,
         post_process_class,
         eval_class,
         model_type=None,
         extra_input=False,
         scaler=None,
         amp_level='O2',
         amp_custom_black_list=[]):
    model.eval()
    with paddle.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader),
            desc='eval model:',
            position=0,
            leave=True)
        max_iter = len(valid_dataloader) - 1 if platform.system(
        ) == "Windows" else len(valid_dataloader)
        sum_images = 0
        for idx, batch in enumerate(valid_dataloader):
            if idx >= max_iter:
                break
            images = batch[0]

            start = time.time()

            # use amp
            if scaler:
                with paddle.amp.auto_cast(
                        level=amp_level,
                        custom_black_list=amp_custom_black_list):
                    if model_type == 'table' or extra_input:
                        preds = model(images, data=batch[1:])
                    elif model_type in ["kie"]:
                        preds = model(batch)
                    elif model_type in ['can']:
                        preds = model(batch[:3])
                    elif model_type in ['sr']:
                        preds = model(batch)
                        sr_img = preds["sr_img"]
                        lr_img = preds["lr_img"]
                    else:
                        preds = model(images)
                preds = to_float32(preds)
            else:
                if model_type == 'table' or extra_input:
                    preds = model(images, data=batch[1:])
                elif model_type in ["kie"]:
                    preds = model(batch)
                elif model_type in ['can']:
                    preds = model(batch[:3])
                elif model_type in ['sr']:
                    preds = model(batch)
                    sr_img = preds["sr_img"]
                    lr_img = preds["lr_img"]
                else:
                    preds = model(images)

            batch_numpy = []
            for item in batch:
                if isinstance(item, paddle.Tensor):
                    batch_numpy.append(item.numpy())
                else:
                    batch_numpy.append(item)
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            if model_type in ['table', 'kie']:
                if post_process_class is None:
                    eval_class(preds, batch_numpy)
                else:
                    post_result = post_process_class(preds, batch_numpy)
                    eval_class(post_result, batch_numpy)
            elif model_type in ['sr']:
                eval_class(preds, batch_numpy)
            elif model_type in ['can']:
                eval_class(preds[0], batch_numpy[2:], epoch_reset=(idx == 0))
            else:
                # post_result = post_process_class(preds, [[640,640,1,1]])
                post_result = post_process_class(preds, batch_numpy[1])
               
                eval_class(post_result, batch_numpy)

            pbar.update(1)
            total_frame += len(images)
            sum_images += 1
        # Get final metric锛宔g. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric


def update_center(char_center, post_result, preds):
    result, label = post_result
    feats, logits = preds
    logits = paddle.argmax(logits, axis=-1)
    feats = feats.numpy()
    logits = logits.numpy()

    for idx_sample in range(len(label)):
        if result[idx_sample][0] == label[idx_sample][0]:
            feat = feats[idx_sample]
            logit = logits[idx_sample]
            for idx_time in range(len(logit)):
                index = logit[idx_time]
                if index in char_center.keys():
                    char_center[index][0] = (
                        char_center[index][0] * char_center[index][1] +
                        feat[idx_time]) / (char_center[index][1] + 1)
                    char_center[index][1] += 1
                else:
                    char_center[index] = [feat[idx_time], 1]
    return char_center


def get_center(model, eval_dataloader, post_process_class):
    pbar = tqdm(total=len(eval_dataloader), desc='get center:')
    max_iter = len(eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(eval_dataloader)
    char_center = dict()
    for idx, batch in enumerate(eval_dataloader):
        if idx >= max_iter:
            break
        images = batch[0]
        start = time.time()
        preds = model(images)

        batch = [item.numpy() for item in batch]
        # Obtain usable results from post-processing methods
        post_result = post_process_class(preds, batch[1])

        #update char_center
        char_center = update_center(char_center, post_result, preds)
        pbar.update(1)

    pbar.close()
    for key in char_center.keys():
        char_center[key] = char_center[key][0]
    return char_center


def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profile_dic)

    if is_train:
        # save_config
        save_model_dir = config['Global']['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(log_file=log_file)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global'].get('use_gpu', False)
    use_xpu = config['Global'].get('use_xpu', False)
    use_npu = config['Global'].get('use_npu', False)
    use_mlu = config['Global'].get('use_mlu', False)

    alg = config['Architecture']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN',
        'CLS', 'PGNet', 'Distillation', 'NRTR', 'TableAttn', 'SAR', 'PSE',
        'SEED', 'SDMGR', 'LayoutXLM', 'LayoutLM', 'LayoutLMv2', 'PREN', 'FCE',
        'SVTR', 'ViTSTR', 'ABINet', 'DB++', 'TableMaster', 'SPIN', 'VisionLAN',
        'Gestalt', 'SLANet', 'RobustScanner', 'CT', 'RFL', 'DRRG', 'CAN',
        'Telescope'
    ]

    if use_xpu:
        device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
    elif use_npu:
        device = 'npu:{0}'.format(os.getenv('FLAGS_selected_npus', 0))
    elif use_mlu:
        device = 'mlu:{0}'.format(os.getenv('FLAGS_selected_mlus', 0))
    else:
        device = 'gpu:{}'.format(dist.ParallelEnv()
                                 .dev_id) if use_gpu else 'cpu'
    check_device(use_gpu, use_xpu, use_npu, use_mlu)

    device = paddle.set_device(device)

    config['Global']['distributed'] = dist.get_world_size() != 1

    loggers = []

    if 'use_visualdl' in config['Global'] and config['Global']['use_visualdl']:
        save_model_dir = config['Global']['save_model_dir']
        vdl_writer_path = '{}/vdl/'.format(save_model_dir)
        log_writer = VDLLogger(vdl_writer_path)
        loggers.append(log_writer)
    if ('use_wandb' in config['Global'] and
            config['Global']['use_wandb']) or 'wandb' in config:
        save_dir = config['Global']['save_model_dir']
        wandb_writer_path = "{}/wandb".format(save_dir)
        if "wandb" in config:
            wandb_params = config['wandb']
        else:
            wandb_params = dict()
        wandb_params.update({'save_dir': save_model_dir})
        log_writer = WandbLogger(**wandb_params, config=config)
        loggers.append(log_writer)
    else:
        log_writer = None
    print_dict(config, logger)

    if loggers:
        log_writer = Loggers(loggers)
    else:
        log_writer = None

    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    return config, device, logger, log_writer



    # maps = preds['maps']
    # label_threshold_map = maps[:,0,:,:]   
    # label_threshold_map = label_threshold_map * (label_threshold_map > 0.3)
    # label_threshold_mask = maps[:,1,:,:]
    # bitmap = label_threshold_mask > 0.3
    # bitmap = bitmap[0].numpy() * 255  # (1, H, W)
    # outs = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST,
    #                             cv2.CHAIN_APPROX_SIMPLE)
    # if len(outs) == 3:
    #         _, contours, _ = outs[0], outs[1], outs[2]
    # elif len(outs) == 2:
    #         contours, _ = outs[0], outs[1]
    # xmin = 1000
    # xmax = -1
    # ymin = 1000
    # ymax = -1
    # for i in range(len(contours)):
    #     xmin = min(xmin, contours[i][:,:, 0].min())
    #     xmax = max(xmax, contours[i][:,:, 0].max())
    #     ymin = min(ymin, contours[i][:,:, 1].min())
    #     ymax = max(ymax, contours[i][:,:, 1].max())
    # label_threshold_mask = paddle.zeros_like(label_shrink_mask)
    # label_threshold_mask[xmin:xmax,ymax,ymax] = 1
    # label_shrink_map = maps[:,2,:,:]  
    # label_shrink_map =paddle.ones(1 * (label_shrink_map > 0.3))
    # label_shrink_mask = paddle.ones((4,1,640,640),dtype=np.float32)