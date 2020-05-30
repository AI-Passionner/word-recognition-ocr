Deep-Learning-Based Word Recognition OCR Developement

### Overview
This repo is directly inspired by Baoguang Shi et al.'s CRNN (CNN+LSTM) [paper](https://arxiv.org/abs/1507.05717) published in 2015. The novel neural
 network architecture
 introduced in this paper could be the foundation of the modern OCR technology. There are numerous text recognition repos in GitHub, which are the
  adaption of Shi et al.'s CRNN architecture more or less, including this repo. 
  
I have been using both conventional OCR (like OpenText and Nuance) and deep-learning based OCR (Tesseract and AWS Textract) quite long time. The
 former, I like to call it as a conventional OCR, simply because I want to distinguish it from the modern deep-learning based OCR. 

The CNN-based OCR actually
  outperforms the conventional one with higher accuracy and less image pre-processing. Think about the famous MNIST handwritten digit recognition
   problem. If you build a Logistic Regression model (softmax), probably you will get an accuracy around 93%. Applying a Feed-Forward Nerual Network
    will boost the accuracy up to 98%. However, a convolutional neural network could push the accuracy up to >99% easily.  
    
The conventional OCR actually extracts characteristics out of each isolated shape and then assigns a symbol. With feature extraction, the bitmap of
 each symbol was broken up into a set of characteristics, such as lines, strokes, curves, loops, etc. Rules were then applied to find the closest
  symbol.  The attached is an example of a detailed terminology available to describe the "geography" of a letter form. 
    
    
<img width="964" alt="Font Anatomy" src="https://github.com/AI-Passionner/word-recognition-ocr/blob/master/images/letter-anatomy.png?raw=true">

One big benefit using the convolutional neural network is about the automated feature extraction. This works very well in image-related
 recognition and classification. 
 
However, before the actual character recognition, there is a very challenging part, called character segmentation, separating the various letters of a
 word. If you look at the next two snapshots, you will see what I mean. The character recognition accuracy highly counts on whether the individual
 letters separated out from a word. 
 
    ![Touching Characters](/images/touching_characters.png)
    Format: ![Alt Text](https://github.com/AI-Passionner/word-recognition-ocr/blob/master/images/touching_characters.png?raw=true)
 
    ![Degraded Characters](/images/degraded_characters.png)
    Format: ![Alt Text](https://github.com/AI-Passionner/word-recognition-ocr/blob/master/images/degraded_characters.png?raw=true)
    
The character segmentation can be avoided if the OCR engine uses word recognition with an artificial neural network. After all, separating
 a word out of the text line is much easier than separating individual letters out of a word.  But why word recognition, rather than character
  recognition?  It is because of the particular advantages of the novel CRNN architecture mentioned in the paper. The network architecture is 
   specifically designed for sequence-like object recognition in images. It can learn directly words without detailed character annotations or
    character segmentation. 
 




#### Reference
1. https://arxiv.org/abs/1507.05717
2. https://github.com/bgshih/crnn
3. https://github.com/sbillburg/CRNN-with-STN/blob/master/Batch_Generator.py
4. https://github.com/weinman/cnn_lstm_ctc_ocr
5. https://www.tsgrp.com/2019/02/12/amazon-textract-and-opentext-capture-recognition-engine-recostar-comparison/
6. https://www.how-ocr-works.com/intro/intro.html
