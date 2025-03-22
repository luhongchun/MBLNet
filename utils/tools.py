import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json
import random
from utils.dataset_fine import label2onehot
from skimage.measure import regionprops
import csv
import time
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

#MBLNet
def compute_result_MBLNet(config, data_loader, model, device):
    model.eval()
    with torch.no_grad():
        for batch_cnt_val, (inputs, labels, _) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            feature_map, outputs, outputs_codes, border = model(inputs,labels)
            b, c, w, h = feature_map.shape
            border = torch.sum(feature_map.detach(), dim=1, keepdim=True)
            a = torch.mean(border, dim=[2, 3], keepdim=True)
            M = (border > a).float().detach()
            M = M.squeeze(1).cpu().numpy()
            bboxs = []
            for i in range(len(M)):
                prop = regionprops(M[i].astype(int))
                if len(prop) == 0:
                    bbox = [0, 0, w, w]
                    print('there is one img no intersection')
                else:
                    bbox = prop[0].bbox
                    bboxs.append(bbox)
            i_b, i_c, i_w, i_h = inputs.size()
            local_imgs = torch.zeros([i_b, i_c, i_w, i_h]).to(config['device'])  # [N, 3, 448, 448]
            sign = i_w // w
            for i in range(b):
                [x0, y0, x1, y1] = bboxs[i]
                x0 = x0 * sign - 1
                y0 = y0 * sign - 1
                x1 = x1 * sign - 1
                y1 = y1 * sign - 1
                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0
                local_imgs[i:i + 1] = torch.nn.functional.interpolate(inputs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],
                                                                size=(i_w, i_h), mode='bilinear', align_corners=True)
            _, outputs_object, outputs_codes1, _ = model(local_imgs.detach(), labels)
            #------------------------------------------------------------------------------------------
            if batch_cnt_val == 0:
                ground = labels
                pred_out = outputs_codes1
            else:
                ground = torch.cat((ground,labels))
                pred_out = torch.cat((pred_out,outputs_codes1))

    return torch.sign(pred_out).cpu(), ground.cpu()

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH



# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk=None):
    if topk is None:
        topk = retrievalL.shape[0]
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    # print(num_query)
    # print(num_gallery)
    # print(queryL.shape)
    # print(retrievalL.shape)
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall

# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset, epoch_loss):
    device = config["device"]
    # print("calculating test binary code......")
    trn_binary, trn_label = compute_result_MBLNet(config, dataset_loader, net, device=device)
    # print("calculating dataset binary code.......")
    start_time = time.perf_counter()
    tst_binary, tst_label = compute_result_MBLNet(config, test_loader, net, device=device)

    tst_label = label2onehot(tst_label.cpu(), config["n_class"])
    trn_label = label2onehot(trn_label.cpu(), config["n_class"])
    mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                 trn_binary.numpy(), trn_label.numpy(),
                                                 config["topK"])
    if mAP > Best_mAP:
        Best_mAP = mAP
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
            np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
            np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
            np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
            torch.save(net.state_dict(), os.path.join(save_path, "model.pt"))
        # ----------pr recall----------------------------------
        index_range = num_dataset // 300
        index = [i * 300 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 300
        index = [1] + index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]

        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }
        os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
        with open(config["pr_curve_path"], 'w') as f:
            f.write(json.dumps(pr_data))
        print("pr curve save to ", config["pr_curve_path"])
        # --------------------------------------------------

    print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    # print(config)
    f = open(os.path.join('./result/csv_val/' + config["info"] + '_' + config["dataset"] + '_' + str(bit) + '.csv'), 'a',
             encoding='utf-8', newline='')
    wr = csv.writer(f)
    #wr.writerow(['models', 'mAP'])
    wr.writerow([config["info"], epoch+1, mAP, epoch_loss])
    f.close()
    return Best_mAP
