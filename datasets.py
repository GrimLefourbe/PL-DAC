import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
from pycocotools.coco import COCO
import torch.serialization
import pickle as pkl
import os
from PIL import Image
import numpy as np

PLDAC_data_folder = 'C:/Coding/PLDAC-data'

val_img_folder = PLDAC_data_folder + '/val2017'
capValFile = PLDAC_data_folder + '/annotations/captions_val2017.json'
instValFile = PLDAC_data_folder + '/annotations/instances_val2017.json'
keypValFile = PLDAC_data_folder + '/annotations/person_keypoints_val2017.json'
ValstorageFolder = PLDAC_data_folder + '/val_objects'

Train_img_folder = PLDAC_data_folder + '/train2017'
instTrainFile = PLDAC_data_folder + '/annotations/instances_train2017.json'
TrainStorageFolder = PLDAC_data_folder + '/train_objects'

def getimgfromann(tensor_image, ann, mode='bbox'):
    if mode=='bbox':
        x, y, width, height = ann['bbox']
        return tensor_image[:, round(y):round(y+height), round(x):round(x+width)]

def get_objects_from_cat(dset, cats):
    res = []
    for img, anns in dset:
        for ann in anns:
            if ann['category_id'] in cats:
                res.append((getimgfromann(img, ann, mode='bbox'), ann))
    return res

def save_objects(folder, objects):
    toPIL = transforms.ToPILImage()
    for i, obj in enumerate(objects):
        torch.serialization.save(obj, open(os.path.join(folder,'{}.pkl'.format(i)), 'wb'))
        toPIL(obj[0]).save(os.path.join(folder,'{}.jpg'.format(i)))

def sort_into_cats(cats, dset, base_folder=ValstorageFolder):
    catsdict = {cat['id']:cat['name'] for cat in cats}

    for catid, catname in catsdict.items():
        os.makedirs(os.path.join(base_folder, f'{catid}_{catname}'),exist_ok=True)

    toPIL = transforms.ToPILImage()
    annByCats = {cat:[] for cat in catsdict}

    for img, anns in dset:
        for ann in anns:
            try:
                img_ann = getimgfromann(img, ann, mode='bbox')
            except ValueError:
                print(ann)
                continue
            catid = ann['category_id']
            catname = catsdict[catid]
            annByCats[catid].append(ann)
#            torch.serialization.save((img_ann,ann), open(os.path.join(base_folder,f'{catid}_{catname}',f'{ann["id"]}.pkl'),'wb'))
            pkl.dump(ann, open(os.path.join(base_folder, f'{catid}_{catname}',f'{ann["id"]}.pkl'), 'wb'))
            toPIL(img_ann).save(os.path.join(base_folder,f'{catid}_{catname}',f'{ann["id"]}.jpg'))
            print(f'Saved 1 image in {catname}')

    for catid, catname in catsdict.items():
        pkl.dump(annByCats[catid], open(os.path.join(base_folder, f'{catid}_{catname}', f'all.pkl'), 'wb'))

def sort_into_cats2(cats, dset, base_folder=ValstorageFolder):
    catsdict = {cat['id']:cat['name'] for cat in cats}
    for catid, catname in catsdict.items():
        os.makedirs(os.path.join(base_folder,f'{catid}_{catname}'), exist_ok=True)
        objs = get_objects_from_cat(dset, {catid})
        torch.serialization.save(objs,open(os.path.join(base_folder,f'{catid}_{catname}','all.pkl'), 'wb'))
        del objs
        print(f'Saved {catname}')


class ObjectsDataset(torch.utils.data.Dataset):
    def __init__(self, category, nbperobj=10, shuffled=False, obj_format=(8, 8), bg_format=(64, 64), maxsize=-1,
                 obj_transform=transforms.ToTensor(), bg_transform=transforms.ToTensor(),
                 objects_folder=ValstorageFolder, background_folder=val_img_folder, annFile=instValFile):
        super().__init__()

        self.objects_folder = objects_folder
        self.bg_folder = background_folder

        self.obj_transform = transforms.Compose([transforms.Resize(obj_format), obj_transform])
        self.bg_transform = transforms.Compose([transforms.Resize(bg_format), bg_transform])

        coco = COCO(annFile)
        self.bg_ids = list(coco.imgs.keys())

        self.category = category
        self.nbperobj = min(nbperobj, len(self.bg_ids))

        self.anns = pkl.load(open(os.path.join(f'{objects_folder}', f'{category["id"]}_{category["name"]}', 'all.pkl'), 'rb'))
        print(f'Catégorie sélectionnée : {category["name"]}')
        print(f'{len(self.anns)} objets, {len(self.bg_ids)} backgrounds')

        maxsize = min(maxsize, len(self.anns)*self.nbperobj)

        if shuffled:
            self.indices = np.random.choice(len(self.anns)*self.nbperobj, size=maxsize, replace=True)
            self.bg_ids = np.random.choice(self.bg_ids, size=self.nbperobj, replace=True)
        else:
            self.indices = np.arange(maxsize)
            self.bg_ids = self.bg_ids[:nbperobj].copy()

        self.coords = np.hstack((np.random.randint(bg_format[0] - obj_format[0], size=(maxsize, 1)),
                                 np.random.randint(bg_format[1] - obj_format[1], size=(maxsize, 1))))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]

        obj_id = self.anns[idx//self.nbperobj]['id']
        obj = Image.open(os.path.join(self.objects_folder,f'{self.category["id"]}_{self.category["name"]}', f'{obj_id}.jpg'))

        bg_id = self.bg_ids[idx % self.nbperobj]
        bg = Image.open(os.path.join(self.bg_folder, f'{bg_id:0>12}.jpg')).convert('RGB')

        obj = self.obj_transform(obj)
        bg = self.bg_transform(bg)

        return obj, bg, torch.IntTensor(self.coords[item])

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

def get_cat_as_tensors(cat, storage=ValstorageFolder):
    return np.array(torch.serialization.load(open(os.path.join(storage, f"{cat['id']}_{cat['name']}", 'all.pkl'),'rb')))

if __name__ == '__main__':
    # cap = dset.CocoCaptions(root=img_folder, annFile=capFile, transform=transforms.ToTensor())
    inst = dset.CocoDetection(root=Train_img_folder, annFile=instTrainFile, transform=transforms.ToTensor())
    # keyp = dset.CocoDetection(root=img_folder, annFile=keypFile, transform=transforms.ToTensor())
    coco = COCO(instTrainFile)
    sort_into_cats(coco.loadCats(coco.getCatIds()), inst, base_folder=TrainStorageFolder)
#    t = ObjectsDataset({'id':2, 'name':'bicycle'})
