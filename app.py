import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from PIL import Image, ImageOps
import numpy as np
#from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input
import numpy as np

class_names = open("labels.txt", "r").readlines()

@st.cache_resource()
def load_model():
    res_model = tf.keras.models.load_model('keras_model.h5')
    cnn_model = tf.keras.models.load_model('cnn.h5')
    #mobilenet_model = tf.keras.models.load_model('mobilenet.h5')
    return res_model, cnn_model


with st.spinner('Model is being loaded..'):
    res_model, cnn_model = load_model()

st.write("""
         # Image Classification
         """
         )

file = st.file_uploader(
    "Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


def upload_predict(upload_image,  res_model, cnn_model):


    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(upload_image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    #cnn_image = cv2.resize(image,(224, 224))
    #cnn_img = tf.keras.applications.resnet50.preprocess_input(cnn_image)
    #cnn_img = np.expand_dims(cnn_img, 0)
    #mob_image = cv2.resize(image,(224, 224))
    #mob_img = tf.keras.applications.resnet50.preprocess_input(mob_image)
    #mob_img = np.expand_dims(mob_img, 0)

    res_prediction = res_model.predict(data)
    cnn_prediction = cnn_model.predict(data)
    #mobilenet_prediction = mobilenet_model.predict(data)


    print(res_prediction)
    print(cnn_prediction)
    #print(mobilenet_prediction)

    return res_prediction, cnn_prediction

def class_resolver(pred):
    pred = pred.tolist()
    max_val = max(pred)
    index_res = pred.index(max_val)
    dict_class = {0:"Leaf Blight",1:"Brown Spot",2:"Leaf Smut",3:"Healthy"}
    pred = dict_class[index_res]
    return pred
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).convert("RGB")
    st.image(image)
    res_pred_class, cnn_pred_class= upload_predict(image,  res_model, cnn_model)


    index = np.argmax(res_pred_class)
    res_image_class = class_names[index]
    
    confidence_score = res_pred_class[0][index]

    cnn_image_class = class_resolver(cnn_pred_class[0])


    #mobilenet_image_class = class_resolver(mobilenet_pred_class[0])
    

    st.write("ResNet50 Classification", res_image_class)
   

    st.write("CNN Classification", cnn_image_class)


    #st.write("MobileNet Classification", mobilenet_image_class)



    #print("The image is classified as ", image_class,
    #      "with a similarity score of", score)
