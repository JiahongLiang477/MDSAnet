import os
import time
import torch
import random
import argparse
import numpy as np
from torchvision import transforms
from utils.cls_visual import pred_cls_map_dl
import utils.data_load_operate as data_load_operate
from utils.data_load_operate import applyPCA
from utils.evaluation import Evaluator, evaluate_OA
from sklearn import metrics
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from model.MDSAnet import MDSANet



time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())


def vis_a_image(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=False):
    visualize_predict(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=only_vis_label)
    visualize_predict(gt_vis,pred_vis,save_single_predict_path.replace('.png','_mask.png'),save_single_gt_path,only_vis_label=True)


# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int,default=0)
    parser.add_argument('--model_name', type=str, default='MDSAnet') 
    parser.add_argument('--data_set_path',type=str,default='./dataset')
    parser.add_argument('--work_dir',type=str,default='./')
    parser.add_argument('--exp_name', type=str, default='Results_MDSAnet')


    parser.add_argument("--data_train", type=int, default=0) 
    parser.add_argument('--train_samples', type=int, default=120)
    parser.add_argument('--val_samples', type=int, default=40)
    parser.add_argument("--train_ratio", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.01)

    parser.add_argument("--model_type_flag", type=int, default=1)
    parser.add_argument('--model_3D_spa_flag', type=int, default=0, help="0 or 1")
    parser.add_argument("--patch_size", type=int, default=11)
    parser.add_argument('--in_channels', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()
    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_parser()

exp_name = args.exp_name
seed_list = [2501]  #

num_list = [args.train_samples, args.val_samples] 
ratio_list = [args.train_ratio, args.val_ratio]  

dataset_index = args.dataset_index

max_epoch = args.max_epoch
batch_size = args.batch_size
learning_rate = args.lr

net_name = args.model_name

patchsize = args.patch_size

paras_dict = {'net_name':net_name,'dataset_index':dataset_index,'num_list':num_list,
              'lr':learning_rate,'seed_list':seed_list}

                     # 0        1         2         3       
data_set_name_list = ['UP', 'Houston', 'HongHu', 'HanChuan']
data_set_name = data_set_name_list[dataset_index]



transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[123.6750, 116.2800, 103.5300], std=[58.395, 57.120, 57.3750]),
])


if __name__ == '__main__':
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    setting_name = 'tr{}val{}'.format(str(args.train_samples),str(args.val_samples)) + '_lr{}'.format(str(learning_rate))

    dataset_name = data_set_name

    exp_name = args.exp_name

    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder,'train_tr{}_val{}.log'.format(num_list[0],num_list[1]))
    logger = setup_logger(name='{}'.format(dataset_name),logfile=save_log_path)
    torch.cuda.empty_cache()

    logger.info(save_folder)

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)
    height, width, channels = data.shape
    gt_reshape = gt.reshape(-1)
    class_count = max(np.unique(gt))
    patch_size = args.patch_size
    patch_length = patch_size // 2
    
    data_padded = data_load_operate.data_pad_zero(data, patch_length)
    data_padded = applyPCA(data_padded, args.in_channels)
    height_patched, width_patched, channels_patched = data_padded.shape

    

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    evaluator = Evaluator(num_class=class_count)
    
    data_total_index = np.arange(data.shape[0] * data.shape[1])  # For total sample cls_map.

    for exp_idx,curr_seed in enumerate(seed_list):

        setup_seed(curr_seed)
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                                       num_list,
                                                                                                       gt_reshape,
                                                                                                       class_count,
                                                                                                       args.data_train)
        index = (train_data_index, val_data_index, test_data_index)
        train_iter, val_iter, test_iter = data_load_operate.generate_iter_1(data_padded, height, width, gt_reshape, index, patch_length, batch_size, args.model_type_flag,
                                                                               model_3D_spa_flag=args.model_3D_spa_flag, last_batch_flag=0)
         # load data for the cls map of all the labed samples
        all_iter = data_load_operate.generate_iter_2(data_padded, height, width, gt_reshape, all_data_index,
                                                     patch_length,
                                                     batch_size, args.model_type_flag, model_3D_spa_flag=args.model_3D_spa_flag)
        # load data for the cls map of the total samples
        total_iter = data_load_operate.generate_iter_2(data_padded,height, width, gt_reshape, data_total_index, patch_length,
                     patch_size, args.model_type_flag, model_3D_spa_flag=args.model_3D_spa_flag)
        

        net = MDSANet(input_channels=channels_patched, num_classes=class_count, patch_size=patch_size, device=device)
        logger.info(paras_dict)
        logger.info(net)


        # ############################################
        # val_label = test_label
        # ############################################

        net.to(device)

        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]

        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
        logger.info(optimizer)
        best_loss = 99999

        best_val_acc = 0
        # Event record time begin
        # train_start_event = torch.cuda.Event(enable_timing=True)
        # train_end_event = torch.cuda.Event(enable_timing=True)
        # train_start_event.record()

        # Time record time begin
        torch.cuda.synchronize(device)
        train_ts = time.perf_counter()
        for epoch in range(max_epoch):
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            loss_dict = {}

            net.train()
            
            if args.model_type_flag == 1 :   # data for single spatial net 
                for X_spa, y in train_iter:
                    X_spa, y = X_spa.to(device), y.to(device)
                    y_pred = net(X_spa)

                    ls = loss_func(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0
            elif args.model_type_flag == 2:  # data for single spectral net
                for X_spe, y in train_iter:
                    X_spe, y = X_spe.to(device), y.to(device)
                    y_pred = net(X_spe)

                    ls = loss_func(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0
            
            elif args.model_type_flag == 3:  # data for spectral-sptial net
                for X_spa, X_spe, y in train_iter:
                    X_spa, X_sp, y = X_spa.to(device), X_spe.to(device), y.to(device)
                    y_pred = net(X_spa, X_spe)

                    ls = loss_func(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    batch_counter += 1
                    epoch_first_iter = 0

            val_acc, val_loss = evaluate_OA(val_iter, net, loss_func, device, args.model_type_flag)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            if val_acc>=best_val_acc:
                best_val_acc = val_acc
                torch.save(net.state_dict(), save_weight_path)
                print(save_weight_path)
            

            train_loss_list.append(train_loss_sum)
            train_acc_list.append(train_acc_sum / trained_samples_counter)
            
            logger.info('epoch: %d, training_sampler_num: %d, batch_count: %.2f, train loss: %.6f, tarin loss sum: %.6f, '
             'train acc: %.3f, val_acc: %.3f, train_acc_sum: %.1f', 
             epoch + 1, trained_samples_counter, batch_counter, train_loss_sum / batch_counter, train_loss_sum,
             train_acc_sum / trained_samples_counter, val_acc, train_acc_sum)
            
            torch.cuda.empty_cache()

        torch.cuda.synchronize(device)
        train_to = time.perf_counter()
        Train_Time_ALL.append(train_to-train_ts)

        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        pred_test = []

        load_weight_path = save_weight_path
        net.update_params = None
        # best_net = copy.deepcopy(net)
        # best_net = BaseNet(input_channels=channels_patched, n_classes=class_count, patch_size=patch_size)
        best_net = MDSANet(input_channels=channels_patched, num_classes=class_count, patch_size=patch_size, device=device)
        
        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        test_evaluator = Evaluator(num_class=class_count)

        torch.cuda.synchronize(device)
        test_ts = time.perf_counter()

        with torch.no_grad():
            train_acc_sum, samples_num_counter = 0.0, 0

            if args.model_type_flag == 1:  # data for single spatial net
                for X_spa, y in test_iter:
                    X_spa = X_spa.to(device)
                    y = y.to(device)

                    y_pred = best_net(X_spa)

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))

            elif args.model_type_flag == 2:  # data for single spectral net
                for X_spe, y in test_iter:
                    X_spe = X_spe.to(device)
                    y = y.to(device)
                    y_pred = best_net(X_spe)

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))

            elif args.model_type_flag == 3:  # data for spectral-spatial net
                for X_spa, X_spe, y in test_iter:
                    X_spa = X_spa.to(device)
                    X_spe = X_spe.to(device)
                    y = y.to(device)
                    y_pred = best_net(X_spa, X_spe)

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))

            torch.cuda.synchronize(device)
            test_to = time.perf_counter()
            Test_Time_ALL.append(test_to - test_ts)

            y_gt = gt_reshape[test_data_index] - 1
            confusion_matrix = metrics.confusion_matrix(pred_test, y_gt)
            print(confusion_matrix)
            test_evaluator.reset()
            test_evaluator.add_confusion_matrix(confusion_matrix=confusion_matrix)
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = test_evaluator.Kappa()
            logger.info('Test:|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test,
                                                                                    Acc_test))

            # Visualization for all the labeled samples and total the samples
            sample_list1 = [all_iter, all_data_index]
            sample_list2 = [total_iter]

            pred_cls_map_dl(sample_list1,best_net,gt, predict_save_path,args.model_type_flag, device)
            pred_cls_map_dl(sample_list2, best_net, gt,  predict_save_path,args.model_type_flag, device)    


        

        # Output infors
        if args.data_train == 1:
            train_data = str(args.train_samples)
            val_data = str(args.val_samples)
        else:
            train_data = str(ratio_list[0]) + str('%')
            val_data   = str(ratio_list[1]) + str('%')
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train data=" + str(train_data) \
                      + " val data=" + str(val_data) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) \
                      + "\ntrain_time:" + str(Train_Time_ALL[exp_idx]) \
                      + "\ntest_time:" + str(Test_Time_ALL[exp_idx])+ "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA:', str(list(OA_ALL)))
    logger.info('List of AA:', str(list(AA_ALL)))
    logger.info('List of KPP:', str(list(KPP_ALL)))
    logger.info('OA=', str(round(np.mean(OA_ALL) * 100, 2)), '±', str(round(np.std(OA_ALL) * 100, 2)))
    logger.info('AA=', str(round(np.mean(AA_ALL) * 100, 2)), '±', str(round(np.std(AA_ALL) * 100, 2)))
    logger.info('Kpp=', str(round(np.mean(KPP_ALL) * 100, 2)), '±', str(round(np.std(KPP_ALL) * 100, 2)))
    logger.info('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '±',
                np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))

    logger.info("Average training time=", str(round(np.mean(Train_Time_ALL), 2)), '±', str(round(np.std(Train_Time_ALL), 3)))
    logger.info("Average testing time=", str(round(np.mean(Test_Time_ALL) * 1000, 2)), '±',
          str(round(np.std(Test_Time_ALL) * 1000, 3)))
    del net
