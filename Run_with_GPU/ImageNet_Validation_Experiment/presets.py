import mindspore as ms
from mindspore.dataset import vision
from mindspore.dataset.vision.transforms import Inter


def classification_preset_train(crop_size,
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225),
                                interpolation=Inter.BILINEAR,
                                hflip_prob=0.5,
                                auto_augment_policy=None,
                                random_erase_prob=0.0):
    trans = [vision.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0:
        trans.append(vision.RandomHorizontalFlip(hflip_prob))
    if auto_augment_policy is not None:
        if auto_augment_policy == "ra":
            trans.append(vision.RandAugment(interpolation=interpolation))
        elif auto_augment_policy == "ta_wide":
            trans.append(vision.TrivialAugmentWide(interpolation=interpolation))
        else:
            aa_policy = vision.AutoAugmentPolicy(auto_augment_policy)
            trans.append(vision.AutoAugment(policy=aa_policy, interpolation=interpolation))
    trans.extend(
        [
            vision.ToTensor(),
            vision.ToType(ms.float32),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
        ]
    )
    if random_erase_prob > 0:
        trans.append(vision.RandomErasing(prob=random_erase_prob))

    return trans


def classification_preset_eval(crop_size,
                               resize_size=256,
                               mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225),
                               interpolation=Inter.BILINEAR):
    trans = [
        vision.Resize(resize_size, interpolation=interpolation),
        vision.CenterCrop(crop_size),
        vision.ToTensor(),
        vision.ToType(ms.float32),
        vision.Normalize(mean=mean, std=std, is_hwc=False),
    ]
    return trans
