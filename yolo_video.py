import sys
import argparse
import os
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import json

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

def get_image_files(dir):
    imgs = []
    file_dir = os.listdir(dir)
    file_dir.sort(key=lambda x:int(x[:-4]))
    #print(file_dir)
    for file in file_dir:
        #print(file[:-4])
        file_lower = file.lower()
        if file_lower.endswith(".png") or file_lower.endswith(".jpg"):
            imgs.append(os.path.join(dir, file))
    return imgs

def detect_imgdir(yolo, dir, output_txt=False):
    sub_dict = {"bbox":[], "label":[], "score":[]}
    submission = []
    speed_set = []
    img_files = get_image_files(dir)
    save_dir = os.path.join('output')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    for img in img_files:
        try:
            image = Image.open(img)
        except:
            print('Open Error! {}'.format(img))
            continue
        else:
        

            fullpath = os.path.join(save_dir, os.path.basename(img))
            
            detections = list()
            bounding_box_cord = list()
            label_set = list()
            score_set = list()
            speed = list()
            
            
            r_image = yolo.detect_image(image, single_image=False, output=detections, bbox_set=bounding_box_cord, label_set=label_set, score_set=score_set, speed=speed)
            

            
            
            sub_dict["bbox"] = bounding_box_cord
            sub_dict['label'] = label_set
            sub_dict['score'] = score_set
            submission.append(dict(sub_dict))
            speed_set.append(speed[0])
            if not output_txt:
                r_image.save(fullpath,"PNG")
                print('save {}'.format(fullpath))
            else:
                basename = os.path.basename(img)  # eg. 123.jpg
                txt_file = os.path.splitext(basename)[0]+'.txt'  # eg. 0001
                txt_fullpath = os.path.join(save_dir, txt_file)
                with open(txt_fullpath, 'w+') as the_file:
                    full_text = '\n'.join(detections)
                    the_file.write(full_text)

    print("Image Count: " +str(len(speed_set)))
    print("Average: " + str(1000*sum(speed_set)/ len(speed_set)) +" ms per image")
    txt_fullpath = os.path.join(save_dir, '0856621_2.json')
    with open(txt_fullpath, 'w+') as json_file:
        json.dump(submission, json_file, cls=MyEncoder)
        
    yolo.close_session()

FLAGS = None


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )

    parser.add_argument(
        '--imgdir', type=str, default='',
        help='Image dir detection mode, will ignore all positional arguments'
    )

    parser.add_argument(
        '--txt', default=False, action="store_true",
        help='Image dir detection will output txt files'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))

    elif os.path.isdir(FLAGS.imgdir):
        print("Image directory mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_imgdir(YOLO(**vars(FLAGS)), FLAGS.imgdir, FLAGS.txt)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")