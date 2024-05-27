import os
import sys

import mindspore.context as context

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import presets
import transforms
import mindspore_hub as mshub
from mindspore.experimental import optim
import mindspore as ms
from mindspore import nn, ops
import logging
import numpy as np
from PIL import Image
from mindspore.dataset.transforms import RandomChoice
import random
import pickle as pkl
import pandas as pd

CFG = {
    'root_dir': '/data/dataset',
    'seed': 42,
    'resize_size': 256,  # val
    'crop_size': 224,  # train
    'epochs': 1,
    'warmup_epochs': 5,
    'train_bs': 64,
    'valid_bs': 64,
    'lr': 0.5,
    'weight_decay': 2e-5,
    'lr_warmup_decay': 0.01,
    'num_workers': 4,
    'accum_iter': 1,
    'verbose_step': 5,
    'device': 'GPU',
    'num_classes': 1000,
    'model_name': 'mindspore/1.9/resnet50_imagenet2012',  # swin_tiny_patch4_window7_224,resnet50
    'pkl_pth': 'eq_1000_1000.pkl',
    'info': 'EBV_ResNet_dim1000_SGD_epoch105',
    'ifval': False,
    'model_path': 'EBV_ResNet_dim1000_SGD_epoch105.ckpt'
}
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("./" + CFG['info'] + ".txt")
handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(stream_handler)


def seed_everything(seed):
    random.seed(seed)
    ms.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_img(path):
    img = Image.open(path).convert('RGB')
    return img


class ImageNetDataset:
    def __init__(self, root, part='train'):
        self.part = part
        self.images = []
        self.labels = []
        if part == 'train':
            mycsv = pd.read_csv('imagenet_train.csv')
        else:
            mycsv = pd.read_csv('imagenet_val.csv')
        for i in range(len(mycsv['image_id'])):
            self.images.append(root + mycsv['image_id'][i])
            self.labels.append(int(mycsv['label'][i]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = get_img(self.images[index])
        return image, self.labels[index]


def forward_fn(data, label, d):
    logits = model(data)
    loss = loss_fn((logits @ d.t() / 0.07), label)
    return loss


def train_one_epoch(epoch, model, grad_fn, optimizer, train_loader,
                    d, scheduler=None, schd_batch_update=False):
    model.set_train()

    running_loss = None

    train_loader = train_loader.create_tuple_iterator()

    for step, (imgs, image_labels) in enumerate(train_loader):

        imgs = imgs.float()

        loss, grads = grad_fn(imgs, image_labels, d)

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
            optimizer(grads)
            if scheduler is not None and schd_batch_update:
                scheduler.step()

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) % CFG['verbose_step'] == 0):
            description = f'epoch {epoch} batch {step + 1} loss: {running_loss:.4f}'
            logger.info(description)

    logger.info(' Epoch: ' + str(epoch) + ' Final Train Loss: {:.4f}'.format(running_loss))

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(epoch, model, val_loader, d):
    model.set_train(False)

    total_num = 0
    correct_num = 0

    val_loader = val_loader.create_tuple_iterator()

    for imgs, image_labels in val_loader:
        imgs = imgs.float()

        image_preds = model(imgs)  # batch_size * 50

        image_preds = image_preds @ d.t()  # .abs()  #bs *200

        image_preds = ops.argmax(image_preds, 1).numpy()
        image_targets = image_labels.numpy()

        total_num += image_targets.shape[0]
        correct_num += (image_preds == image_targets).sum()

    ans = correct_num / total_num
    logger.info(' Epoch: ' + str(epoch) + ' validation accuracy = {:.6f}'.format(ans))
    return ans


def test_one_epoch(model, val_loader, d):
    model.set_train(False)

    image_preds_all = []
    image_targets_all = []

    val_loader = val_loader.create_tuple_iterator()

    for imgs, image_labels in val_loader:
        imgs = imgs.float()

        image_preds = model(imgs)  # batch_size * 50

        image_preds = image_preds @ d.t()  # .abs()  #bs *200

        image_preds_all += [ops.argmax(image_preds, 1).numpy()]
        image_targets_all += [image_labels.numpy()]

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    ans = (image_preds_all == image_targets_all).mean()
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))

    return ans


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = mshub.load(name=CFG['model_name'], class_num=CFG['num_classes'], pretrained=False)

    def construct(self, x):
        x = self.backbone(x)
        x = ops.L2Normalize()(x)
        return x


if __name__ == '__main__':

    context.set_context(device_target=CFG['device'])

    seed_everything(CFG['seed'])

    model = Net()

    ##############################################

    mixupcutmix = RandomChoice([
        transforms.RandomMixup(num_classes=CFG['num_classes'], p=1.0, alpha=0.2),
        transforms.RandomCutmix(num_classes=CFG['num_classes'], p=1.0, alpha=1.0)
    ])

    train_transforms = presets.classification_preset_train(crop_size=CFG['crop_size'],
                                                           auto_augment_policy="ta_wide",
                                                           random_erase_prob=0.1)

    train_dataset = ImageNetDataset(root=CFG['root_dir'], part='train')

    collate_fn = lambda img, label, info: mixupcutmix(img, label)

    train_loader = ms.dataset.GeneratorDataset(source=train_dataset,
                                               column_names=["image", "label"],
                                               num_parallel_workers=CFG['num_workers'],
                                               shuffle=True)

    train_loader = train_loader.map(train_transforms, 'image')

    train_loader = train_loader.batch(batch_size=CFG['train_bs'], input_columns=['image', 'label'],
                                      per_batch_map=collate_fn)

    val_dataset = ImageNetDataset(root=CFG['root_dir'], part='val')

    val_transforms = presets.classification_preset_eval(crop_size=CFG['resize_size'], resize_size=CFG['resize_size'])

    val_loader = ms.dataset.GeneratorDataset(source=val_dataset, column_names=["image", "label"],
                                             num_parallel_workers=CFG['num_workers'])
    val_loader = val_loader.map(val_transforms, 'image')
    val_loader = val_loader.batch(batch_size=CFG['valid_bs'])


    ##############################################

    optimizer = optim.SGD(model.trainable_params(), lr=CFG['lr'], weight_decay=CFG['weight_decay'], momentum=0.9)

    grad_fn = ms.value_and_grad(forward_fn, None, model.trainable_params(), has_aux=False)

    main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs'] - CFG['warmup_epochs']  # , eta_min = 1e-5
    )
    warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=CFG['lr_warmup_decay'], total_iters=CFG['warmup_epochs']
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[CFG['warmup_epochs']]
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    d = pkl.load(open(CFG['pkl_pth'], 'rb')).data
    d = ops.L2Normalize()(d)

    if CFG['ifval'] == True:
        ms.load_param_into_net(model, ms.load_checkpoint(CFG['model_path']))
        answer = test_one_epoch(model, val_loader, d)
        exit()

    best_answer = 0.0
    for epoch in range(CFG['epochs']):
        train_one_epoch(epoch, model, grad_fn, optimizer, train_loader,
                        d, scheduler=scheduler, schd_batch_update=False)
        answer = 0.0

        if epoch % 1 == 0:
            answer = valid_one_epoch(epoch, model, val_loader, d)
        if answer > best_answer:
            ms.save_checkpoint(model, CFG['info'] + '.ckpt'.format(epoch))
        if answer > best_answer:
            best_answer = answer

    del model, optimizer, train_loader, val_loader, scheduler
    logger.info('BEST-TEST-ACC: ' + str(best_answer))
