
# coding: utf-8

# In[ ]:


from build_image import read_image,classify_decode
from build_models import build_classify,build_segment
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

# In[ ]:


class prediction():
    
    def __init__(self,classify_model_weights, segment_model_weights,input_shape=(3,256,256)):
    
        self.final_classify_model = build_classify()

        K.set_image_data_format('channels_first')
        self.final_segment_model = build_segment(input_shape)
    
        self.final_classify_model.load_weights(classify_model_weights)
        print('Classification Model Loaded....')
        self.final_segment_model.load_weights(segment_model_weights)
        print('Segmentation Model Loaded...') 
    
    def Predict(self,image_path):
        """
        This function predicts the classfication output and if it is greater than 0.5 it passes the image to the segmentation model.
        ------------------------------------------------------------------
        final_classify_model      : saved classification model instance
        final_segment_model       : saved segmentation model instance
        image_path                : image path
        ------------------------------------------------------------------
        """

         # read the original image
        image_orig = classify_decode(image_path)
    
        #get the classification output
        classify_output = self.final_classify_model.predict(tf.expand_dims(image_orig, axis=0))
        confidence = classify_output

        # checking the threshold:
        if confidence > 0.5:
            print('Pneumothorax Found..!!')
            print('Classifier Prediction Confidence : {}%'.format(classify_output*100))

            # read the original image
            image_seg = read_image(image_path)

            # reshape image and mask as first channel image format
            image = tf.transpose(image_seg, [2,0,1])

            # get the segmented output
            predicted_mask = self.final_segment_model.predict(tf.expand_dims(image, axis=0))

            #Re-arranging the dimensions for displaying the mask
            predicted_mask = tf.transpose(predicted_mask, [0,2,3,1])
            #plt.imshow(image_seg)      
            #plt.imshow(np.squeeze(predicted_mask[:,:,:,1]), cmap='Reds', alpha = 0.3)
        
            return confidence, predicted_mask
        else:
            image_seg = read_image(image_path)
            return confidence, image_seg
       
