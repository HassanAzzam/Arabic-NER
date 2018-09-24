# Arabic-NER

Model uses Named Entity Recognition concept to tag words in arabic text.

## Architecture

It consists of Bi-directional GRU units(One forward and the other is backward) and a CRF layer
![Architecture](https://raw.githubusercontent.com/HassanAzzam/Arabic-NER/master/arch.png)
<sup>referenced from https://arxiv.org/pdf/1508.01991v1.pdf<sup>

## Specifications

Model is trained on [ANERCorp dataset](http://users.dsic.upv.es/~ybenajiba/downloads.html).<sup>[more](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp).</sup> And uses FastText's Arabic vectors for word embedding.

No. epochs: 20

Accuracy: 94.2%

Classification report:

                  precision    recall  f1-score   support

             LOC       0.99      0.99      0.99     11055
            PERS       0.74      0.65      0.69       824
             ORG       0.64      0.46      0.54       503
            MISC       0.63      0.38      0.47       237
     avg / total       0.95      0.94      0.94     12619

F1_score: 95.0%

## Sample
#### Input
ماذا يفعل طلال عبد الهادي في دبي بعد ما رجع من برلين؟ كان يعمل هناك في شركة فولكسفاجن، صحيح؟
#### Output
![Example](https://raw.githubusercontent.com/HassanAzzam/Arabic-NER/master/example.png)
