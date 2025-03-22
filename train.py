from utils.tools1 import *
from network1 import *
from torch.optim import lr_scheduler
import torch
import torch.optim as optim
import time
import csv
from utils.dataset_fine1 import get_data_fine, calc_train_codes
from skimage.measure import regionprops
from sklearn.metrics import hamming_loss
torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "alpha": 0.1,
        "lr": 0.01,
        "info": "[MBLNet]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": ResNet18_MBLNet,
        "dataset": "aircraft",
        # "dataset": "Stanford_Cars",
        # "dataset": "cub_bird",
        "epoch": 1,
        "test_map": 1,
        "save_path": "result/pth/MBLNet",
        "device": torch.device("cuda:0"),
        "bit_list": [12, 24, 36, 48],
        "set_seed": 111,
        "class_mask": 0.7,
        "cam_mask_rate": 0.3,
    }
    # config = config_dataset(config)
    return config

def train_val(config, bit):
    device = config["device"]
    dataloader, num_dataset = get_data_fine(config)
    model = config["net"](bit, config["n_class"], config["class_mask"]).to(device)
    # ---------------------------- loss and opt ------------------------------------
    criterion = nn.CrossEntropyLoss()
    criterion_hash = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), config["lr"], momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # ---------------------------- log and train ------------------------------------
    # log_file = open(config["info"] + '_' + config["dataset"] + '_' + str(bit) + '.log', 'a')
    # log_file.write(str(config))
    # log_file.write('\n')
    print('training start ...')
    # log_file.close()
    Best_mAP = 0
    train_codes = calc_train_codes(dataloader, bit, config["n_class"])
    for epoch in range(config["epoch"]):
        model.train()
        ce_loss = 0.0
        current_time = time.perf_counter()
        for batch_cnt, (inputs, labels, item) in enumerate(dataloader['train']):

            codes = torch.tensor(train_codes[item, :]).float().cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            feature_map, outputs_class, outputs_codes, border, _, x_b_1 = model(inputs, labels)
            # --------------------Branch 1-Erasing the high-response part prompts the network to learn finer-grained features--------------------------------------------
            #----channel filtering-----------------------------
            c, w, h = feature_map.shape[1:]
            nonzeros = (feature_map != 0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6  # Percentage of non-zeroes per channel map
            nonzeros_mean = torch.mean(nonzeros, dim=1, keepdim=True)  # Average of the percentage of non-zeroes for all channels
            Mask = (nonzeros > nonzeros_mean).float().detach()  # The number of non-zeroes per channel is assigned a value of 1 if it is greater than the average value, otherwise it is assigned a value of 0. Filtering the useless channel noise and retaining only the highly responsive channel features
            feature_map1 = feature_map * Mask.unsqueeze(-1).unsqueeze(-1)  # Get the filtered feature map
            # ---spatial erase--------------------------------
            attention = torch.sum(feature_map1.detach(), dim=1, keepdim=True)
            attention = nn.functional.interpolate(attention, size=(224, 224), mode='bilinear', align_corners=True)
            masks = []
            for i in range(labels.size()[0]):
                threshold = random.uniform(0.9, 1.0)
                mask = (attention[i] < threshold * attention[i].max()).float()
                masks.append(mask)
            masks = torch.stack(masks)
            hide_imgs = inputs * masks
            _, outputs_hide, _, _, _, _ = model(hide_imgs,labels)
            # ----------------------Branch 2 - Extracting Global Features - Zooming in on the target in the input image by attentively retrieving the target edges-----------------------------------
            b, c, w, h = feature_map.shape
            border = torch.sum(feature_map.detach(), dim=1, keepdim=True)
            a = torch.mean(border, dim=[2, 3], keepdim=True)
            M = (border > a).float().detach()
            M = M.squeeze(1).cpu().numpy()
            bboxs = []
            for i in range(len(M)):
                prop = regionprops(M[i].astype(int))  # seek a boundary
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
                local_imgs[i:i + 1] = nn.functional.interpolate(inputs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(i_w, i_h), mode='bilinear', align_corners=True)
            _, outputs_object, outputs_codes1, _, MC_loss, x_b_2 = model(local_imgs.detach(), labels)
            #--------------------end------------------------------------------------
            loss_class = criterion(outputs_class, labels) #Original Branch Loss
            loss_class_hide = criterion(outputs_hide, labels) #Erase Loss
            loss_class_object = criterion(outputs_object, labels)#Border Loss
            loss_class_object = loss_class_object + 0.005 * (MC_loss[0] + 10.0 * MC_loss[1])
            loss_class_rival = hamming_loss(x_b_1.cpu().numpy(), x_b_2.cpu().numpy())  # Against Loss, Hamming Loss 公式见https://blog.csdn.net/Bit_Coders/article/details/124961880
            loss_codes = criterion_hash(outputs_codes1, codes)#Hashloss
            loss = loss_class + loss_codes + loss_class_hide + 2*loss_class_object + 10*loss_class_rival
            loss.backward()
            optimizer.step()
            ce_loss += loss.item() * inputs.size(0)

        epoch_loss = ce_loss / dataloader['train'].total_item_len
        exp_lr_scheduler.step()

        print("epoch:", epoch+1)
        print("\b loss:%.3f" % (epoch_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, dataloader['val'], dataloader['base'], model, bit, epoch, num_dataset, epoch_loss)

    f = open(os.path.join('./result/csv/'+config["info"]+'_'+config["dataset"]+'_' + str(bit) + '.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['models', 'Best_mAP'])
    wr.writerow([config["info"],  Best_mAP])
    f.close()

    return Best_mAP

if __name__ == "__main__":
    config = get_config()
    print(config)
    set_seed(config["set_seed"])
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"result/json/resnet18/MBLNet_{config['dataset']}_{bit}.json"
        Best_mAP = train_val(config, bit)

