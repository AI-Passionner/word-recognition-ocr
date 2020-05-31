## Deep-Learning-Based Word Recognition OCR

### Background
This repo is directly inspired by Baoguang Shi et al.'s CRNN (CNN+LSTM) [paper](https://arxiv.org/abs/1507.05717) published in 2015. The novel
 neural network architecture introduced in this paper could be the foundation of modern OCR technology. There are numerous text recognition repos
  in GitHub after this paper, which are more or less the adaption of the CNN+LSTM architecture, including this one. 

I have been using both conventional OCR (OpenText) and deep-learning-based OCR (Tesseract and AWS Textract) quite a long time. The former, I like to call it as a conventional OCR, simply because I want to distinguish it from the modern deep-learning-based OCR. 
 The CNN-based OCR outperforms the conventional one with [higher accuracy](https://www.tsgrp.com/2019/02/12/amazon-textract-and-opentext-capture-recognition-engine-recostar-comparison) and less image pre-processing. 

Think about the famous MNIST handwritten digit recognition problem. If you build a Logistic Regression model (softmax), probably you will get an
 accuracy of around 93%. Applying a Feed-Forward Neural Network will boost the accuracy of up to 98%. However, a convolutional neural network could push the accuracy up to >99% easily. 
    
The conventional OCR extracts characteristics out of each isolated shape and then assigns a symbol. With feature extraction, the bitmap of each symbol was broken up into a set of characteristics, such as lines, strokes, curves, loops, etc. Rules were then applied to find the closest symbol. The attached is an example of a detailed terminology available to describe the "geography" of a letter form. 
    
<img width="400" alt="Font Anatomy" src="https://github.com/AI-Passionner/word-recognition-ocr/blob/master/images/fontology_anatomy.gif?raw=true">

One big benefit of using the convolutional neural network is the automated feature extraction. This works very well in image-related recognition and classification. 

However, before the actual character recognition, there is a very challenging part, called character segmentation, separating the various letters of a word. If you look at the next two snapshots, you will see what I mean. Some letters are touching and even degraded. It is not a easy to segment individual letters out. It is also mission impossible to recognize those degraded letters! 
 
<img width="400" alt="Degraded Characters" src="https://github.com/AI-Passionner/word-recognition-ocr/blob/master/images/degraded_characters.png?raw=true">

However, the character segmentation can be avoided if the OCR engine uses word recognition with an artificial neural network. After all, separating a word of the text line is much easier than separating individual letters of a word. But why word recognition, rather than character recognition? It is because of the particular advantages of the novel CRNN architecture mentioned in the paper. The CNN+LSTM architecture is specifically designed for sequence-like object recognition in images. It can learn directly words without detailed character annotation or segmentation. 

My philosophy to Machine Learning and Artificial Intelligence is that if you want the machine to predict the data more accurately, you had better
  let it “see” it. This sounds a little bit of “cheating”. But it is the truth. In machine learning, it is very common that the new model works pretty well at the beginning after the deployment. However, it becomes worse and worse as time going. There is nothing wrong with the model. It is the data because new data are not similar to the training data pool. Back to the text recognition, I developed a word recognition model first, trained on millions of synthetic word images. It achieves >99% accuracy and works pretty well on regular text images (like book pages, newspaper, etc.). When I applied the model on business documents, its performance drops. Why? Because those training synthetic word images are obtained from regular and clean text images. 

The text recognition is relatively static. You won’t see big changes in text styles. The training data are cheap and accessible, no matter synthetic or real text images. Developing a new OCR model won’t take a long time. This is why I conduct this research, developing a customized OCR for some business documents. My goal is to achieve a comparable and even higher recognition rate on some business documents than the AWS Textract. The word recognition is the first step in this research. The next step is to conduct document layout analysis, including font style, fonts size, line, cell, box, table, and block. All of these could be very indicative and discriminative features, used for building robust models in downstream. 

Mind you, machine learning is not about the machine’s “intelligence”, it is all about automation.   


#### Reference
1. https://arxiv.org/abs/1507.05717
2. https://github.com/bgshih/crnn
3. https://github.com/sbillburg/CRNN-with-STN/blob/master/Batch_Generator.py
4. https://github.com/weinman/cnn_lstm_ctc_ocr
5. https://www.tsgrp.com/2019/02/12/amazon-textract-and-opentext-capture-recognition-engine-recostar-comparison
6. https://www.how-ocr-works.com/intro/intro.html
