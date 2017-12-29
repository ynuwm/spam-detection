# spam-dectection
Implement of paper Learning to Represent Review with Tensor Decomposition for Spam Detection,emnlp,2016
# results
RE(50:50):
          precision    recall  f1-score   support
          0       0.75      0.83      0.79        77
          1       0.81      0.73      0.77        77

avg / total       0.78      0.78      0.78       154
 
 ('RE,50:50,accuracy:', 0.7792207792207793)


RE(ND):
             precision    recall  f1-score   support

          0       0.29      0.83      0.44        77
          1       0.96      0.69      0.81       500

avg / total       0.87      0.71      0.76       577
 
 ('RE,ND,accuracy:', 0.7123050259965338)


<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
RE+PE(50:50)

             precision    recall  f1-score   support

          0       0.78      0.87      0.82        77
          1       0.85      0.75      0.80        77

avg / total       0.82      0.81      0.81       154
0.811688311688


RE+PE(ND):

             precision    recall  f1-score   support

          0       0.32      0.83      0.46        77
          1       0.97      0.72      0.83       500

avg / total       0.88      0.74      0.78       577
0.738301559792

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
RE+PE+BiGram(50:50)
             precision    recall  f1-score   support

          0       0.72      0.82      0.76        77
          1       0.79      0.68      0.73        77

avg / total       0.75      0.75      0.75       154
0.746753246753


RE+PE+BiGram(ND)
             precision    recall  f1-score   support

          0       0.35      0.83      0.49        77
          1       0.97      0.76      0.85       500

avg / total       0.88      0.77      0.80       577
0.769497400347



