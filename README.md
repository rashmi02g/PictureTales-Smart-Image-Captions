# <u>PictureTales-Smart Image Captions</u>
This project generates descriptive captions for images using deep learning. The model takes an input image 
and returns a relevant caption, aiming to aid visually impaired individuals and enhance image accessibility

## <u>Overview</u>
The system leverages a pretrained VGG16 model for feature extraction and a custom captioning model which was trained using LSTM for generating captions. The model is trained on the Flickr8k dataset  

The key components of the project include:
- Image feature extraction using a pretrained VGG16 model
- Caption preprocessing and tokenization
- Custom captioning model architecture
- Model training and evaluation
  <br>
## Model Anatomy
![model](screenshots/Model-Anatomy.png)
