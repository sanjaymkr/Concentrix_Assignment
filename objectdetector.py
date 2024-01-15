import math
import os
from collections import defaultdict
import traceback
from typing import List, Tuple
from .intitalizer import FUNC_NAME
import ast
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self,input_width = 640, input_height = 640, score_threshold = 0.2, nms_threshold = 0.4, confidence_threshold = 0.4, logger=None, model_path : list = None, output_folder : str = None):
        self.class_list = ['part', 'table_borders','table_no_borders','user_photo'] #class names
        self.NUM_THREAD_CPU = cv2.getNumThreads()
        self.NUM_THREAD_GPU = cv2.getNumThreads()
        self.INPUT_WIDTH = input_width
        self.INPUT_HEIGHT = input_height
        self.SCORE_THRESHOLD = score_threshold
        self.NMS_THRESHOLD = nms_threshold
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.OUTPUT_FOLDER = output_folder
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.is_cuda = self.detect_cuda()
        self.model_path = model_path
        self.net = self.build_model()
        self.check_ouput_folders()
        self.TABLE_PATH = os.path.join(self.OUTPUT_FOLDER, "tables")
        self.OUTPUT_PATH = os.path.join(self.OUTPUT_FOLDER, "output")
        self.PART_PATH = os.path.join(self.OUTPUT_FOLDER, "part")
        self.PHOTO_PATH = os.path.join(self.OUTPUT_FOLDER, "userphoto")


    def check_ouput_folders(self):
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER, mode=0o777, exist_ok=True)

        sub_dirs = ["tables", "output", "part","userphoto"]
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(self.OUTPUT_FOLDER, sub_dir)
            if not os.path.exists(sub_dir_path):
                os.makedirs(sub_dir_path, mode=0o777, exist_ok=True)

    def detect_cuda(self):
        cv2.setUseOptimized(True)
        if cv2.useOptimized() and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Use GPU
            cv2.cuda.setDevice(0)
            cv2.setNumThreads(self.NUM_THREAD_GPU)
            return True
        else:
            # Use CPU
            cv2.setNumThreads(self.NUM_THREAD_CPU)
            return False

    def build_model(self):
        models = []
        for model_path in self.model_path:
            net = cv2.dnn.readNet(str(model_path))
            if self.is_cuda:
                print("Attempting to use CUDA")
            else:
                print("Running on CPU")
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            models.append(net)
        return models

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0,
                                     (self.INPUT_WIDTH, self.INPUT_HEIGHT),
                                     mean=0,
                                     swapRB=True, crop=False)
        output=[]
        for i in self.net:
            i.setInput(blob)
            preds = i.forward()
            has_positive_inf = np.any(np.isposinf(preds))
            has_negative_inf = np.any(np.isneginf(preds))
            has_inf = has_positive_inf or has_negative_inf
            logger.info(f"[{__class__.__name__}] ({FUNC_NAME()}) has_inf :{has_inf}")
            if has_inf:
                continue
            output.append(preds)
        if output:
            final_output=np.concatenate(output, axis=1)
            return final_output
        else:
            return []

    def wrap_detection(self, input_image, output_data):
        result_class_ids = []
        result_confidences = []
        result_boxes = []

        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]
        image_width, image_height, _ = input_image.shape
        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT
        for row_idx in range(rows):
            row = output_data[row_idx]
            confidence = row[4]
            if any(math.isinf(value) for value in row):
                continue
            elif confidence >= 0.4:
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):
                    try:
                        class_ids.append(class_id)
                        x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                        if any(math.isinf(value) for value in [x, y, w, h]):
                            continue
                        else:
                            left = int((x - 0.5 * w) * x_factor) if int((x - 0.5 * w) * x_factor) > 0 else 0 #bugfix for negative dimension
                            top = int((y - 0.5 * h) * y_factor) if int((y - 0.5 * h) * y_factor)  > 0 else 0
                            width = int(w * x_factor)
                            height = int(h * y_factor)
                            box = np.array([left, top, width, height])
                            boxes.append(box)
                            confidences.append(confidence)
                    except:
                        e = traceback.format_exc()
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

        result_size = len(indexes)
        result_confidences = [0.0] * result_size
        result_class_ids = [0] * result_size
        result_boxes = [None] * result_size

        for i, index in enumerate(indexes):
            result_confidences[i] = confidences[index]
            result_class_ids[i] = class_ids[index]
            result_boxes[i] = boxes[index]
        return result_class_ids, result_confidences, result_boxes


    def format_yolov5(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def check_folders(self, image_path):
        save_folder_name = os.path.splitext(os.path.basename(image_path))[0]
        table_path = os.path.join(self.TABLE_PATH, save_folder_name)
        output_path = os.path.join(self.OUTPUT_PATH, save_folder_name)
        part_path = os.path.join(self.PART_PATH, save_folder_name)
        photo_path= os.path.join(self.PHOTO_PATH, save_folder_name)
        if not os.path.exists(table_path):
            os.mkdir(table_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(part_path):
            os.mkdir(part_path)
        if not os.path.exists(photo_path):
            os.mkdir(photo_path)
        return output_path, table_path, part_path, photo_path

    def process_image_to_format(self, image_path: str) -> Tuple[str, List[np.ndarray], np.ndarray]:
        # File Name
        file_name=os.path.basename(image_path).split('.')[0]
        # Frames
        if not isinstance(image_path,str):
            image_path = str(image_path)
        frames = cv2.imread(image_path)
        height, width = frames.shape[:2]
        yoloImage = self.format_yolov5(frames)
        return file_name, frames, yoloImage, height , width


    def check_overlap(self,rect, bbox):
        bbox_tuple = tuple(bbox)

        if ((rect[0] >= bbox_tuple[0] - 10) and (rect[1] >= bbox_tuple[1] - 10) and \
                (rect[0] + rect[2] <= bbox_tuple[0] + bbox_tuple[2] + 10) and \
                (rect[1] + rect[3] <= bbox_tuple[1] + bbox_tuple[3] + 10)):
                return True
        return False

    def find_overlapping_indices(self,boxes):
        overlapped_indices = set()
        for i, rect in enumerate(boxes):
            for j, bbox in enumerate(boxes):
                if i != j and self.check_overlap(rect, bbox):
                    overlapped_indices.add(i)
                    break
        return overlapped_indices

    def check_overlap_boxes(self,class_ids, confidences, boxes):
        result_class_ids = []
        result_confidences = []
        result_boxes = []

        # Convert the list of boxes to numpy array
        boxes = np.array(boxes)

        overlapped_indices = self.find_overlapping_indices(boxes)

        for i, (class_id, confidence, box) in enumerate(zip(class_ids, confidences, boxes)):
            if i not in overlapped_indices:
                result_class_ids.append(class_id)
                result_confidences.append(confidence)
                result_boxes.append(box.tolist())

        return result_class_ids, result_confidences, result_boxes
    def find_intersection(self,rect1, rect2):
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
        y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
        return x1, y1, x2, y2

    def calculate_area(self,rect):
            return rect[2] * rect[3]

    def calculate_overlap(self,rect1, rect2):

        x1, y1, x2, y2 = self.find_intersection(rect1, rect2)

        if x1 >= x2 or y1 >= y2:
            return 0

        intersection_area = (x2 - x1) * (y2 - y1)
        rect1_area = self.calculate_area(rect1)
        rect2_area = self.calculate_area(rect2)

        overlap_percentage = (intersection_area / min(rect1_area, rect2_area)) * 100
        return overlap_percentage

    def remove_overlap(self,class_ids, confidences, boxes):
        coord_rem = set()
        coord_add=[]

        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes[i + 1:], start=i + 1):
                overlap_per = self.calculate_overlap(box1, box2)
                if overlap_per > 10:
                    coord_rem.add(i)
                    coord_rem.add(j)
                    new_cord= [
                            min(box1[0],box2[0]),
                            min(box1[1],box2[1]),
                            max(box1[0]+box1[2],box2[0]+box2[2]),
                            max(box1[1]+box1[3],box2[1]+box2[3]),
                            ]
                    fin_coord=[
                        new_cord[0],new_cord[1],
                        new_cord[2]-new_cord[0],
                        new_cord[3]-new_cord[1]
                    ]
                    coord_add.append(fin_coord)

        new_class = []
        new_conf = []
        new_box = []
        new_class = [class_ids[i] for i in range(len(boxes)) if i not in coord_rem]
        new_conf = [confidences[i] for i in range(len(boxes)) if i not in coord_rem]
        new_box = [boxes[i] for i in range(len(boxes)) if i not in coord_rem]

        if len(coord_add)>0:
            for i in coord_add:
                new_box.append(i)
                new_class.append(0)
                new_conf.append(0.75)

        return new_class, new_conf, new_box
    def predict(self, image_path, write_bounding_box=False):
        crop_file_paths = photo_file_path = []
        bbox_results = {}
        photo_file_path=""
        crop_file_paths  = []
        file_name, frames, yoloImage,im_h,im_w = self.process_image_to_format(image_path)
        output_path, table_path, part_path, photo_path = self.check_folders(image_path)

        detection = self.detect(yoloImage)
        try:
            if detection.size > 0:
                class_ids, confidences, boxes = self.wrap_detection(yoloImage, detection[0])
                class_ids, confidences, boxes = self.check_overlap_boxes(class_ids,confidences,boxes)
                class_ids, confidences, boxes = self.remove_overlap(class_ids,confidences,boxes)
                bbox_path = os.path.join(self.OUTPUT_FOLDER,f"{file_name}.txt")
                if write_bounding_box:
                # if True:
                    with open(bbox_path, "w") as fp:
                        fp.writelines("\n".join(map(str, boxes)))
                i=0

                bbox_results = defaultdict(list)
     
                for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                    color = self.colors[int(classid) % len(self.colors)]
                    cv2.rectangle(frames, box, color, 2)
                    cv2.rectangle(frames, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                    crop=frames[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
                    if crop.size > 0:
                        if classid==1: # Table with borders
                            file_path = os.path.join(table_path, file_name+self.class_list[classid]+str(i)+".jpg")
                            box[1]=im_h-box[1]
                            cv2.imwrite(file_path,crop)
                            crop_file_paths.append(file_path)
                            bbox_results['table_borders'].append(box)
                        elif classid==2: # Table without borders
                            file_path = os.path.join(table_path, file_name+self.class_list[classid]+str(i)+".jpg")
                            cv2.imwrite(file_path,crop)
                            crop_file_paths.append(file_path)
                            bbox_results['table_no_borders'].append(box)
                        elif classid==3: # Userphoto
                            photo_file_path = os.path.join(photo_path, file_name+self.class_list[classid]+str(i)+".jpg")
                            cv2.imwrite(photo_file_path,crop)
                        else: # Part
                            file_path = os.path.join(part_path,file_name+self.class_list[classid]+str(i)+".jpg")
                            cv2.imwrite(file_path,crop)
                            crop_file_paths.append(file_path)
                            bbox_results['part'].append(box)

                    i+=1
                    cv2.putText(frames, f"{self.class_list[classid]} - {','.join(map(str,box))}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
                cv2.imwrite( os.path.join(output_path, file_name+".jpg"),frames)
        except:
            e = traceback.format_exc()

        return crop_file_paths, bbox_results, photo_file_path
