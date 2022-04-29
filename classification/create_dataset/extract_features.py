import cv2
import numpy as np

def binarizeImage(image):

  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  histg = cv2.calcHist([image],[0],None,[256],[0,256]) 
  thr = np.argmax(histg) - 15
  _, bina_image = cv2.threshold(image, thresh=thr, maxval=255, 
                                type=cv2.THRESH_BINARY_INV)

  return bina_image.astype(np.uint8)

def extract_area(img):
    binarize_img = binarizeImage(img)
    area = np.sum(binarize_img) / 255
    binarize_img = cv2.copyMakeBorder(binarize_img, 10, 10, 10, 10, 
                                      cv2.BORDER_CONSTANT, None, value = 0)
    return binarize_img, area

def extract_length(img):
    binarize_img = binarizeImage(img)
    binarize_img = cv2.copyMakeBorder(binarize_img, 10, 10, 10, 10, 
                                      cv2.BORDER_CONSTANT, None, value = 0)
    skeleton = cv2.ximgproc.thinning(binarize_img)
    length = np.sum(skeleton) / 255
    return skeleton, length


#Load the image in grayscale
img_path = 'F:/chromosome/chromosomes/chromosomes/4_02/chr-3-1.jpg'
img = cv2.imread(img_path)

binarize_img, area = extract_area(img)
cv2.imwrite('binary.jpg', binarize_img)

skeleton, length = extract_length(img)
cv2.imwrite('skel.jpg', skeleton)
