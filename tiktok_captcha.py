import cv2
import PIL
import numpy as np
import requests
import keras
import tflite_runtime.interpreter as tflite

batch_size = 32
img_height = 64
img_width = 48
class_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'a_l', 'a_u', 'b_u', 'c', 'cone', 'cubic', 'cylinder', 'd_u', 'e_l', 'e_u', 'f_u', 'flower', 'g_l', 'g_u', 'globular', 'h_l', 'h_u', 'hexic', 'k_u', 'l_u', 'm_l', 'm_u', 'n_l', 'p_u', 'q_u', 'r_l', 'r_u', 's_l', 't_l', 't_u', 'u_u', 'v', 'w_u', 'x', 'y_l', 'y_u', 'z']
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 4000
TF_MODEL_FILE_PATH = 'models/model.tflite'

def soft_max(x):
    exponents = []
    for element in x:
        exponents.append(np.exp(element))
    summ = sum(exponents)
    for i in range(len(exponents)):
        exponents[i] = exponents[i] / summ 
    return exponents 

# image = cv2.imread("captcha/4.png")
def checkInner(a, b):
    xa, ya, wa,ha = a
    xb, yb, wb,hb = b
    area_a = wa * ha
    area_b = wb * hb
    if area_a >= area_b:
        return False
    
    # if area_a > area_b:
    #     if xb >= xa and xb <= xa + wa and yb >= ya and yb <= ya + ha:
    #         return -1
    #     return 0
    if xa >= xb and xa <= xb + wb and ya >= yb and ya <= yb + hb:
        return True
    return False


def removeInner(bounding_arr):
    results = []
    looper = range(0, len(bounding_arr))
    for i in looper:
        isInner = False
        for j in looper:
            isInner = checkInner(bounding_arr[i], bounding_arr[j])
            if isInner:
                break
        
        if isInner:
            continue
        results.append(bounding_arr[i])
    return results


def splitImg(image):
    img2 = cv2.convertScaleAbs(image, alpha=1.35, beta=2)
    gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, black_white = cv2.threshold(gray_img, 250, 255, cv2.THRESH_BINARY)
    
    Contours, _ = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rect_arr = []
    images = []
    for contour in Contours:
        area = cv2.contourArea(contour)

        if area > MIN_CONTOUR_AREA and area < MAX_CONTOUR_AREA:
            rect_arr.append(cv2.boundingRect(contour))

    rect_arr = removeInner(rect_arr)
    for i, rect in enumerate(rect_arr):
        x, y, w, h = rect
        cropped_image = black_white[y : y + h, x : x + w]
        cropped_image = cv2.resize(cropped_image, (img_width, img_height))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
        # filename = "working/"+str(i)+".jpg"
        # cv2.imwrite(filename, cropped_image)
        images.append(cropped_image)
        

    return [rect_arr, images]

# model = create_model()

# model.summary()

# model.load_weights("checkpoints/cp.ckpt")
interpreter = tflite.Interpreter(model_path=TF_MODEL_FILE_PATH)
classify_lite = interpreter.get_signature_runner('serving_default')
print("model loaded")


def get_classify(img):
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img, 0) # Create a batch

    prediction = classify_lite(rescaling_1_input=img_array)['dense_1']

    score = soft_max(prediction)
    if(np.max(score) < 0.5):
        return None
    return class_names[np.argmax(score)]

def get_the_same(arr):
    max_index = len(arr)
    for i in range(max_index):
        for j in range(i + 1, max_index):
            print (str(arr[i]) + ":"+ str(arr[j]))
            if arr[i] == arr[j]:
                return [i,j]
    return [None, None]

def get_center_point(rect):
    return [rect[0] + rect[2]/2, rect[1] + rect[3]/2]


def bypass(img_url):
    resp = requests.get(img_url)
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)
    rect_array, images = splitImg(image)
    if len(images)  !=7 :
        return []
    res = [get_classify(img) for img in images]
    if res == None:
        return []
    a, b = get_the_same(res)
    if a != None and b != None:
        return [get_center_point(rect_array[a]), get_center_point(rect_array[b])]
    return []
    


