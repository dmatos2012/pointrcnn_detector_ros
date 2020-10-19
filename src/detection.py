#!/usr/bin/env python
import sys
import os
import rospy
import numpy as np
import torch
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import cv2
from torch.utils.data import DataLoader
from sensor_msgs.msg import PointCloud2
import torch.nn.functional as F
import ros_numpy
import random
from prcnn_ros.msg import Detection, DetectionArray
import argparse

#PointRCNN imports
import PointRCNN.tools._init_path
from PointRCNN.lib.net.point_rcnn import PointRCNN
from PointRCNN.lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from PointRCNN.lib.config import cfg, cfg_from_file
import PointRCNN.lib.utils.calibration as calibration
from PointRCNN.lib.utils.bbox_transform import decode_bbox_target
import PointRCNN.lib.utils.kitti_utils as kitti_utils
import PointRCNN.lib.utils.iou3d.iou3d_utils as iou3d_utils


class BoundingBoxNode():
    def __init__(self, calib, model):
        self.calib = calib
        self.model = model
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/kitti/camera_color_left/image_raw", Image, self.img_callback, queue_size=1)
        self.image_pub = rospy.Publisher("image_boxes", Image, queue_size = 1)
        self.velo_sub = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.velo_callback, queue_size=1, buff_size = 2**24)
        self.img_shape = [375,1242,3] # Hardcoded. img shape initialization 
        self.line_strip_pub = rospy.Publisher("draw_boxes", MarkerArray, queue_size = 1)
        self.detection_pub = rospy.Publisher("det_boxes", DetectionArray, queue_size = 1)
        self.cv_image = None
      

    def img_callback(self, data):
        height = data.height
        width = data.width
        self.img_shape = list([height,width, 3])
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") # due to cv using bgr not rgb
    
    def extract_data(self, data):
        random_select = True
        #pcl_msg = pc2.read_points(data, skip_nans=False)
        #pts_lidar = np.array(list(pcl_msg), dtype=np.float32)
        pts_lidar = ros_numpy.point_cloud2.pointcloud2_to_xyzi_array(data)
        mode = 'TEST'
        # get valid point (projected points should be in image)
        pts_rect = self.calib.lidar_to_rect(pts_lidar[:, 0:3])
        pts_intensity = pts_lidar[:, 3]

        pts_img, pts_rect_depth = self.calib.rect_to_img(pts_rect)
        pts_valid_flag = KittiRCNNDataset.get_valid_flag(pts_rect, pts_img, pts_rect_depth, self.img_shape)
        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]

        if mode == 'TRAIN' or random_select:
            if cfg.RPN.NUM_POINTS < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, cfg.RPN.NUM_POINTS - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if cfg.RPN.NUM_POINTS > len(pts_rect):
                    extra_choice = np.random.choice(choice, cfg.RPN.NUM_POINTS - len(pts_rect), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]
        if mode == 'TEST':
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect

        return pts_input

    def velo_callback(self, data):
        pts_input = self.extract_data(data)
        np.random.seed(666)
        with torch.no_grad():
            MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
            self.model.eval()
            inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
            inputs = torch.unsqueeze(inputs, 0)
            input_data = {'pts_input': inputs}
            # model inference
            ret_dict = self.model(input_data)
            batch_size = 1 
            roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
            roi_boxes3d = ret_dict['rois']  # (B, M, 7)
            seg_result = ret_dict['seg_result'].long()  # (B, N)

            rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
            rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

            # bounding box regression
            anchor_size = MEAN_SIZE
            if cfg.RCNN.SIZE_RES_ON_ROI:
                assert False

            pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                            anchor_size=anchor_size,
                                            loc_scope=cfg.RCNN.LOC_SCOPE,
                                            loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                            num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                            get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                            loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                            get_ry_fine=True).view(batch_size, -1, 7)

            # scoring
            if rcnn_cls.shape[2] == 1:
                raw_scores = rcnn_cls  # (B, M, 1)

                norm_scores = torch.sigmoid(raw_scores)
                pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
            else:
                pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
                cls_norm_scores = F.softmax(rcnn_cls, dim=1)
                raw_scores = rcnn_cls[:, pred_classes]
                norm_scores = cls_norm_scores[:, pred_classes]
            # refine results
            roi_boxes3d_np = roi_boxes3d.cpu().numpy()
            pred_boxes3d_np = pred_boxes3d.cpu().numpy()
            roi_scores_raw_np = roi_scores_raw.cpu().numpy()
            raw_scores_np = raw_scores.cpu().numpy()

            rpn_cls_np = ret_dict['rpn_cls'].cpu().numpy()
            rpn_xyz_np = ret_dict['backbone_xyz'].cpu().numpy()
            seg_result_np = seg_result.cpu().numpy()
            output_data = np.concatenate((rpn_xyz_np, rpn_cls_np.reshape(batch_size, -1, 1),
                                        seg_result_np.reshape(batch_size, -1, 1)), axis=2)
            # scores threshold
            inds = norm_scores > cfg.RCNN.SCORE_THRESH
            for k in range(batch_size):
                cur_inds = inds[k].view(-1)
                if cur_inds.sum() == 0:
                    continue
                pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
                raw_scores_selected = raw_scores[k, cur_inds]
                norm_scores_selected = norm_scores[k, cur_inds]
                # NMS thresh
                # rotated nms
                boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
                keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
                pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
                scores_selected = raw_scores_selected[keep_idx]
                try:
                    pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()
                    pred_boxes3d_selected, scores_selected = threshold_score(pred_boxes3d_selected, scores_selected, 1.7)
                    self.visualize_image_plane(self.cv_image, self.calib, pred_boxes3d_selected,scores_selected, self.img_shape)
                    self.visualize_lidar_plane(data, self.calib, pred_boxes3d_selected,scores_selected, self.img_shape)
                    self.dets_to_tracker(self.calib, pred_boxes3d_selected, scores_selected, self.img_shape) # used to send to tracker
                    print("Number of detections pr msg: ", pred_boxes3d_selected.shape[0])
                except TypeError:
                    print("Empty detections")


    def dets_to_tracker(self, calib, bbox3d, scores, img_shape):
        '''
        Sends coordinates as a marker array for the AB3DMOT tracker to use
        '''
        bboxes_arr = DetectionArray()
        #arr_bbox.header.frame_id = "camera_color_left"
        car_type_int = 2.0
        corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d) #(N,8,3)
        # Project 3D bb into image plane
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

        for k in range(bbox3d.shape[0]):
            bbox = Detection()
            # bbox.header.frame_id = "camera_color_left"
            # #bbox.header.stamp = rospy.Time.now()
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry
            #h,w,l,x,y,z,rot_y
            bbox.bbox3d = list([bbox3d[k,3],bbox3d[k,4],bbox3d[k,5], bbox3d[k,0],bbox3d[k,1],bbox3d[k,2], 
            bbox3d[k,6]])
            bbox.other_info = list([alpha, car_type_int, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2],
             img_boxes[k, 3], scores[k]])
            bboxes_arr.bboxes.append(bbox)
        self.detection_pub.publish(bboxes_arr)
        bboxes_arr.bboxes = []

    


    def visualize_image_plane(self, img, calib,bbox3d, scores, img_shape):
        img2 = np.copy(img)
        corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d) #(N,8,3)
        # Project 3D bb into image plane
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

        for k in range(bbox3d.shape[0]):

            if box_valid_mask[k] == 0:
                continue
            corners2d = kitti_utils.project_to_image(corners3d[k], calib.P2)
            colors = get_color()
            img2 = kitti_utils.draw_projected_box3d(img2, corners2d,colors[k]) 
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(img2, "bgr8"))
    
    def visualize_lidar_plane(self, data, calib, bbox3d, scores, img_shape):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = data.header.frame_id
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.header.stamp = rospy.Time.now()

        # marker scale (scale y and z not used due to being linelist)
        marker.scale.x = 0.08
        # marker color
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.points = []        
        corner_for_box_list = [0,1,0,3,2,3,2,1,4,5,4,7,6,7,6,5,3,7,0,4,1,5,2,6]
        corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d) #(N,8,3)
        for box_nr in range(corners3d.shape[0]):  #corners3d.shape[0]
            for corner in corner_for_box_list: 
                box3d_pts_3d_velo = calib.project_rect_to_velo(corners3d[box_nr])
                p = Point()
                p.x = box3d_pts_3d_velo[corner,0]
                p.y = box3d_pts_3d_velo[corner,1]
                p.z = box3d_pts_3d_velo[corner,2]
                marker.points.append(p)
        marker_array.markers.append(marker)

        id = 0
        for m in marker_array.markers:
            m.id = id
            id += 1
        self.line_strip_pub.publish(marker_array)        
        marker_array.markers = []
        

# Load checkpoint
def load_checkpoint(model=None, optimizer=None, filename='checkpoint'):
    if os.path.isfile(filename):
        #logger.info("==> Loading from checkpoint '{}'".format(filename))
        print("loading from checkpoint", filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        #logger.info("==> Done")
        print("Done")
    else:
        raise FileNotFoundError

    return it, epoch


# Convert ROS Image Msg into PIL
def get_calib(calib_file):
    assert os.path.exists(calib_file)
    return calibration.Calibration(calib_file)
    
def threshold_score(bbox3d, scores, value):
    thr_score = []
    scores = np.squeeze(scores) # make sure they are 1D
    idx_kept = []
    for i, score in enumerate(scores):
        if score > value:
            idx_kept.append(i)
            thr_score.append(score)
    thr_bbox3d = np.zeros((len(idx_kept),7))
    for idx, value in enumerate(idx_kept):
        thr_bbox3d[idx,:] = bbox3d[value,:]
    thr_score = np.expand_dims(thr_score, axis = 1) # return it back to original shape
    return thr_bbox3d, thr_score



def get_color():
    max_colors = 30
    colors = []
    for color in range(max_colors):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        rgb = (r,g,b)
        colors.append(rgb)
    return colors



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path for trained prcnn model")
    parser.add_argument("--calib", type=str, help="Location of calib file")
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    calib = get_calib(args.calib)
    ckpt_filename = args.model

    with torch.no_grad():
        model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST') # hardcoded for car, so (background, car) so 2 classes. 
        model.cuda()
        load_checkpoint(model=model, optimizer=None, filename=ckpt_filename)

    rospy.init_node('BoundingBoxNode', anonymous=True)
    BoundingBoxNode(calib, model)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")

if __name__=="__main__":
    main()
    
    