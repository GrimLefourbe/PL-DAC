import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
from pycocotools.coco import COCO
import torch.serialization
import pickle as pkl
import os
from PIL import Image
import numpy as np

img_folder ='/home/vivien/Documents/PLDAC-data/val2017'
capFile='/home/vivien/Documents/PLDAC-data/annotations/captions_val2017.json'
instFile='/home/vivien/Documents/PLDAC-data/annotations/instances_val2017.json'
keypFile='/home/vivien/Documents/PLDAC-data/annotations/person_keypoints_val2017.json'
storageFolder='/home/vivien/Documents/PLDAC-data/objects'

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

def sort_into_cats(cats, dset, base_folder=storageFolder):
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

def sort_into_cats2(cats, dset, base_folder=storageFolder):
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
                 objects_folder=storageFolder, background_folder=img_folder, annFile=instFile):
        super().__init__()
        self.objects_folder = objects_folder
        self.bg_folder = background_folder

        self.obj_transform = transforms.Compose([transforms.Resize(obj_format), obj_transform])
        self.bg_transform = transforms.Compose([transforms.Resize(bg_format), bg_transform])

        self.bg_dset = dset.CocoDetection(root=background_folder, annFile=annFile, transform=self.bg_transform)

        self.category = category
        self.nbperobj = min(nbperobj, self.bg_dset.__len__())

        self.anns = pkl.load(open(os.path.join(f'{objects_folder}', f'{category["id"]}_{category["name"]}', 'all.pkl'), 'rb'))
        self.indices = np.arange(len(self.anns)*self.nbperobj)
        self.coords = np.hstack((np.random.randint(bg_format[0] - obj_format[0], size=(len(self.indices), 1)),
                                 np.random.randint(bg_format[1] - obj_format[1], size=(len(self.indices), 1))))

        if shuffled:
            np.random.shuffle(self.indices)
        if maxsize > 0:
            self.indices = self.indices[:maxsize]

    def __len__(self):
        return len(self.indices)


    def __getitem__(self, item):
        idx = self.indices[item]

        obj_id = self.anns[idx//self.nbperobj]['id']
        obj = Image.open(os.path.join(f'{self.objects_folder}',f'{self.category["id"]}_{self.category["name"]}', f'{obj_id}.jpg'))

        return self.obj_transform(obj), self.bg_dset[idx % self.nbperobj][0], torch.IntTensor(self.coords[idx])

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

def get_cat_as_tensors(cat, storage=storageFolder):
    return np.array(torch.serialization.load(open(os.path.join(storage, f"{cat['id']}_{cat['name']}", 'all.pkl'),'rb')))

if __name__ == '__main__':
    # cap = dset.CocoCaptions(root=img_folder, annFile=capFile, transform=transforms.ToTensor())
    # inst = dset.CocoDetection(root=img_folder, annFile=instFile, transform=transforms.ToTensor())
    # keyp = dset.CocoDetection(root=img_folder, annFile=keypFile, transform=transforms.ToTensor())
    # coco = COCO(instFile)
    # sort_into_cats(coco.loadCats(coco.getCatIds()), inst)
    t = ObjectsDataset({'id':2, 'name':'bicycle'})
