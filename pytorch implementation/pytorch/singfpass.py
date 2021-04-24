import os
import torch

classdict = {
    0: "airplane",
    1: "bathtub",
    2: "bed",
    3: "bench",
    4: "bookshelf",
    5: "bottle",
    6: "bowl",
    7: "car",
    8: "chair",
    9: "cone",
    10: "cup",
    11: "curtain",
    12: "desk",
    13: "door",
    14: "dresser",
    15: "flower_pot",
    16: "glass_box",
    17: "guitar",
    18: "keyboard",
    19: "lamp",
    20: "laptop",
    21: "mantel",
    22: "monitor",
    23: "night_stand",
    24: "person",
    25: "piano",
    26: "plant",
    27: "radio",
    28: "range_hood",
    29: "sink",
    30: "sofa",
    31: "stairs",
    32: "stool",
    33: "table",
    34: "tent",
    35: "toilet",
    36: "tv_stand",
    37: "vase",
    38: "wardrobe",
    39: "xbox",
    }


def classify_folder(model):
    path = 'to_classify/'
    filenames = [f for f in os.listdir(path) if f.endswith('.pt')]
    file_paths = []
    
    for each in filenames:
        file_paths.append(path + each)
    print(file_paths)

    for file in file_paths:
        single_data = (torch.load(file)).tolist()
        single_filename = os.path.splitext(os.path.basename(str(file)))[0]
        all_data = []
        all_data.append(single_data)
        adt = torch.tensor(all_data)
        adt = adt.permute(0, 2, 1)
        #print(adt)
        logits = model(adt)
        guess = (logits.max(dim=1)[1]).item()
        print('filename: ', single_filename)
        print('predicted class: Nr.: ', guess)
        print('predicted class: Name: ', classdict[guess], '\n')
        print('Logits: ', logits, '\n')
        loadfilename = "classfiles/" + classdict[guess] + ".txt"
        savefilename = "output/" + single_filename + ".xml"
        openfile = open(loadfilename, 'r')

        newfile = open(savefilename, "w")
        nfcontent = openfile.read()
        newfile.write(nfcontent)

