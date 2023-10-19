import numpy as np
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

# def checkInner(a, b):
#     xa, ya, wa,ha = a
#     xb, yb, wb,hb = b
#     area_a = wa * ha
#     area_b = wb * hb
#     if area_a >= area_b:
#         return False
    
#     # if area_a > area_b:
#     #     if xb >= xa and xb <= xa + wa and yb >= ya and yb <= ya + ha:
#     #         return -1
#     #     return 0
#     if xa >= xb and xa <= xb + wb and ya >= yb and ya <= yb + hb:
#         return True
#     return False


# def removeInner(bounding_arr):
#     results = []
#     looper = range(0, len(bounding_arr))
#     for i in looper:
#         isInner = False
#         for j in looper:
#             isInner = checkInner(bounding_arr[i], bounding_arr[j])
#             if isInner:
#                 break
        
#         if isInner:
#             continue
#         results.append(bounding_arr[i])
#     return results

# model = create_model()

# model.summary()

# model.load_weights("checkpoints/cp.ckpt")
interpreter = tflite.Interpreter(model_path=TF_MODEL_FILE_PATH)
classify_lite = interpreter.get_signature_runner('serving_default')
print("model loaded")


def get_classify(img):
    # img = PIL.Image.fromarray(img)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img, 0) # Create a batch
    img_array = np.float32(img)

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

# def get_center_point(rect):
#     return [rect[0] + rect[2]/2, rect[1] + rect[3]/2]


def bypass(images):
    res = [get_classify(img) for img in images]
    if res == None:
        return []
    
    return get_the_same(res)
    
    


