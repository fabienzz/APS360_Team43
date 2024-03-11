import numpy as np
import os
import math
from matplotlib.pyplot import imsave,show
from skimage.io import imread
import cv2
import xml.etree.ElementTree as ET

ROOT_PATH = "data"
SCALE = 255

def loadData(path,type,width,height,size):

    data = np.zeros([height,width,size])
    new_path = os.path.join(ROOT_PATH,type+"_fixedSize")
    os.makedirs(new_path, exist_ok=True)
    # print(new_path)
    for i in range(size):
        img = imread(os.path.join(path,str(i)+'.png'),as_gray=True)

        img_resized = cv2.resize(img, (width, height))

        original_height, original_width = img.shape
        aspect_ratio = original_width / original_height
        target_width = int(min(width, height * aspect_ratio))
        target_height = int(min(height, width / aspect_ratio))

        pad_width = max(0, width - target_width)
        pad_height = max(0, height - target_height)

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        resized_img = cv2.resize(padded_img, (width, height), interpolation=cv2.INTER_LINEAR)
        imsave(new_path + '\\'+str(i)+'.png',resized_img)
        data[:,:,i] = resized_img


    return data


# Extract label from inkml file in the form of "output#img_name#ground_truth", "#" is chosen as the delimiter to avoid conflict with the ground truth and path
def extractLabel(input_path):
    xmlns='{http://www.w3.org/2003/InkML}'

    tree = ET.parse(input_path)
    doc_namespace = xmlns
    root = tree.getroot()
    ground_truth = ""
    for ele in root.findall(doc_namespace + 'annotation'):
        if ele.get('type') == 'truth':
            ground_truth = ele.text
            break

    assert(ground_truth != "")
    if ground_truth[0] == '$' and ground_truth[-1] == '$':
        ground_truth = ground_truth[1:-1]
    ret = ground_truth + '\n'
    return ret

# Extract single character from previously extracted ground truth label and create one-to-one mapping between character and integer value i.e. <key,value>
# Return the maximum length of the ground truth
def getTokenDict():
    dataset = ['train','test','val']

    special_char = ['START','END','PAD']
    val = 0
    token_dict = {}

    # first add special characters
    for s_char in special_char:
        token_dict[s_char] = val
        val += 1

    # find the maximum length of the ground truth to determine the normalized length
    max_length = 0

    # Loop through all data sets to find the maximum length of the ground truth and distinct characters
    for name in dataset:

        f = open(ROOT_PATH + "\\"+name  + '_Label_Normalized.txt', 'r')
        lines = f.readlines()

        for ground_truth in lines:

            max_length = max(max_length,len(ground_truth.split(' ')))

            for char in ground_truth.split(' '):
                if char not in token_dict:
                    token_dict[char] = val
                    val += 1

        f.close()
    
    # store the key value pair in a file for future use
    token_file_path = 'data\\token.txt'

    print("Generating token file...")
    
    f = open(token_file_path, 'w')
    for key,value in token_dict.items():
        f.write(str(key) + ',' + str(value) + '\n')

    print(f"Token file generated, containing {len(token_dict)} entries")

    return max_length, token_dict

# Tokenize the ground truth and pad the ground truth to the maximum length, then store the tokenized ground truth in a new file
def tokenize(max_length,token_dict):
    dataset = ['train','test','val']

    # Loop through all data sets to tokenize the ground truth
    for name in dataset:

        inputf = open(ROOT_PATH + "\\" + name  + '_Label_Normalized.txt', 'r')
        outputf = open(ROOT_PATH + "\\" + name  + '_Label_Tokenized.txt', 'w')

        inputlines = inputf.readlines()
        for ground_truth in inputlines:
            tokenized_label = []
            padding_length = max_length - len(ground_truth.split(' '))

            tokenized_label.append(token_dict['START'])
            for char in ground_truth.split(' '):
                tokenized_label.append(token_dict[char])
            tokenized_label.append(token_dict['END'])

            for i in range(padding_length):
                tokenized_label.append(token_dict['PAD'])

            outputf.write(','.join(str(x) for x in tokenized_label) + '\n')
        inputf.close()
        outputf.close()
    

# untokenize the ground truth to the original(normalized) form
def untokenize(tensor,token_dict,max_length):
    dataset = ['train','test','val']
    special_char = ['START','END','PAD']

    untokenize_ground_truth = ""
    # Loop through all elements to utokenize the ground truth
    for ele in tensor:
        untokenize_ground_truth += token_dict[ele] + ' '
    
    return untokenize_ground_truth


def getDatasetSizes():
    dataset = ['Training','Test','Validation']

    sizes = []
    for name in dataset:
        size = len(os.listdir(ROOT_PATH + "\\" + name + "_small"))
        sizes.append(size-1)

    return sizes

if __name__ == "__main__":

    from inkml2img import inkml2img

    data_path = "data\\SmallDataset"

    test_path = "data\\Test_small"
    train_path = "data\\Training_small"
    val_path = "data\\Validation_small"
    
    # os.makedirs(train_path, exist_ok=True)
    # os.makedirs(test_path, exist_ok=True)
    # os.makedirs(val_path, exist_ok=True)

    # size = len(os.listdir(data_path))
    # count,train_count,val_count,test_count = 0,0,0,0

    # f1 = open(train_path + '/'+'label.txt', 'w')
    # f2 = open(val_path + '/'+'label.txt', 'w')
    # f3 = open(test_path + '/'+'label.txt', 'w')

    # # Split the data into training, validation set
    # for file in os.listdir(data_path):
        # print(file)
        # if file.endswith('.inkml') :
    #         label = extractLabel(os.path.join(data_path,file))
            # if count < size * 0.7:
    #             inkml2img(os.path.join(data_path,file), os.path.join(train_path,str(train_count)+'.png'))
    #             f1.write(label)
                # train_count += 1
            # elif count < size * 0.85:
    #             inkml2img(os.path.join(data_path,file), os.path.join(val_path,str(val_count)+'.png'))
    #             f2.write(label)
                # val_count += 1
            # else:
    #             inkml2img(os.path.join(data_path,file), os.path.join(test_path,str(test_count)+'.png'))
    #             f3.write(label)
                # test_count += 1
            # count += 1
    # f1.close()
    # f2.close()
    # f3.close()
    
    # max_length,token_dict = getTokenDict()
    # tokenize(max_length,token_dict)

    train_size,val_size,test_size = getDatasetSizes()
    # train_dataset  = loadData('data\\Training_small',"train",300,100,train_size)
    val_dataset  = loadData('data\\Validation_small',"val",300,100,val_size)
    test_dataset  = loadData('data\\Test_small',"test",300,100,test_size)
