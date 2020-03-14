import base64
import json
import os
import os.path as osp
import glob
import imgviz
import PIL.Image

from shutil import copyfile
from labelme import utils
"""
time: 2020-3-9
use: clean the datasets,put the png photo and json file in the same path for coverting to coco format
developer: FZH
"""

input_dir='/media/ubuntu/ed46c97a-790d-4295-841f-5780494caf97/datasets2coco/datasets_with_json'

out_file=None

json_files = glob.glob(osp.join(input_dir, '*.json'))
for image_id, json_file in enumerate(json_files):
    if out_file is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(out_file, out_dir)
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    num_json=osp.basename(json_file).split('.')[0]
    copyfile(json_file,'/media/ubuntu/ed46c97a-790d-4295-841f-5780494caf97/datasets2coco/'+'json_all/'+str(num_json)+'.json')

    data = json.load(open(json_file))
    imageData = data.get('imageData')

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data['shapes'], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
    )

    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
    # utils.lblsave(osp.join('/media/ubuntu/ed46c97a-790d-4295-841f-5780494caf97/datasets2coco', str(num_json)+'.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')

json_dirs=glob.glob(osp.join(input_dir, '*_json'))
for json_num,json_dir in enumerate(json_dirs):
    num_png = osp.basename(json_dir).split('_')[0]
    copyfile(json_dir+'/img.png','/media/ubuntu/ed46c97a-790d-4295-841f-5780494caf97/datasets2coco/'+'json_all/'+str(num_png)+'.png')


# 重新写json中文件path
def rewrite_json_file(filepath,json_data):
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=4, sort_keys=False)
    f.close()


json_path_corr=glob.glob(osp.join('/media/ubuntu/ed46c97a-790d-4295-841f-5780494caf97/datasets2coco/'+'json_all', '*.json'))
for json_num, json_dir in enumerate(json_path_corr):
    num_png = osp.basename(json_dir).split('.')[0]
    data = json.load(open(json_dir))
    data['imagePath']=str(num_png)+'.png'
    rewrite_json_file(json_dir, data)
    
    
#    这里，我把生成的png文件和原来的json文件（一一对应的）放到了一个新的目录下，目录下的文件为：
# 1.json、1.png、2.json、2.png…………等等。可以注意到后两段是我自己加的，因为我用的是师兄的旧的数据集，
# 有的json文件中的ImagePath的链接已经对应不上，而正好我在生成我的dataset时，我是按照文件前缀命名的
# ，所以，我更改了json文件中的ImagePath内容，使其能对应。如果你的数据不存在这样的问题，注释掉80-92行
# 即可
