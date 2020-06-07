import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ImageUtils:
    
    @staticmethod
    def show_image(img, size=(6, 12), caption='Untitled', subplot=None):
        if subplot==None:
            _, (subplot) = plt.subplots(1, 1)
        subplot.axis('off')
        subplot.imshow(img, cmap='gray')
        plt.title(caption)  
    
    @staticmethod
    def image_for_extraction(raw_image):
        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def image_for_segmentation(raw_image):
        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        # thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((1,1),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.erode(thresh, (2,2),iterations=2)
        return thresh

    @staticmethod
    def reverseXY(img_h, img_w, bbox):
        x0, y0, x1, y1 = bbox
        x1 = int(x1 * img_w)
        y1 = int(y1 * img_h)
        x0 = int(x0 * img_w)
        y0 = int(y0 * img_h)
        return [x0, y0, x1, y1]

    @staticmethod
    def image_converter(input_img, out_img, out_format ='TIFF', quality = 100):
        im = Image.open(input_img) 
        page = im.convert('RGB')
        page.save(out_img, out_format, quality=quality)       
    
        return
    
    @staticmethod
    def crop_image(bw_image, axis):
        """remove empty pixels as value = 0"""
        # horizonal 
        H, W = bw_image.shape[:2]
        h_th = 1  
        hori_hist = np.mean(bw_image, axis=1)
        hori_hist = [0 if i < h_th else 1 for i in hori_hist]
        if 1 in hori_hist:
            t_b = hori_hist.index(1)
            b_b = H - hori_hist[::-1].index(1)
        else:
            t_b = 0
            b_b = H
              
        # vertical 
        v_th = 1
        vert_hist = np.mean(bw_image, axis=0)
        vert_hist = [0 if i < v_th else 1 for i in vert_hist ]         
        if 1 in vert_hist:
            l_b = vert_hist.index(1)
            r_b = W - vert_hist[::-1].index(1)
        else:
            l_b = 0
            r_b = W
        
        if axis ==0:
            return bw_image[t_b:b_b, :]
        
        elif axis ==1:
            return bw_image[:, l_b:r_b]
        
        elif axis ==2:
            return bw_image[t_b:b_b:, l_b:r_b]     
        
        else:
            return bw_image    
    
    @staticmethod
    def remove_lines(img_gray, horizontalsize=120, verticalsize=100 ):  # in selected ROI, which might be easy
        """
        The larger horizontalsize/verticalsize, the less lines removed. However, the small horizontalsize/verticalsize
        will remove lots of details even they are not real lines
        """

        _, img_thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)     
        rows, cols = img_thresh.shape
    
        # removing horizontal lines
        horizontal = img_thresh.copy()
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
        horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
        horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    
        # inverse the image, so that lines are black for masking
        horizontal_inv = cv2.bitwise_not(horizontal)
    
        # perform bitwise_and to mask the lines with provided mask
        masked_img = cv2.bitwise_and(img_thresh, img_thresh, mask=horizontal_inv)
    
        # removing horizontal lines
        vertical = img_thresh.copy()
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
        vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
        vertical_inv = cv2.bitwise_not(vertical)    
        masked_img = cv2.bitwise_and(masked_img, masked_img, mask=vertical_inv)

        return 255 - masked_img
    
    @staticmethod
    def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
    
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
    
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
    
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    
        return buf
    
    @staticmethod
    def remove_noise_and_smooth(gray_img, thresh_img, kernel=(2, 2)):
        filtered = np.max(thresh_img) - thresh_img
        kernel = np.ones(kernel, np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # img = image_smoothening(img)
        clean_img = cv2.bitwise_not(gray_img, closing)
        G_blur = cv2.GaussianBlur(clean_img, (9,9), 0)

        return G_blur