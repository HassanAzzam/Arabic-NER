# Arabic-NER

Model uses Named Entity Recognition concept to tag words in arabic text.

## Architecture

It consists of Bi-directional GRU units(One forward and the other is backward) and a CRF layer
![Architecture](https://raw.githubusercontent.com/HassanAzzam/Arabic-NER/master/arch.png)
<sup>referenced from https://arxiv.org/pdf/1508.01991v1.pdf<sup>

## Specifications

Model is trained on [ANERCorp dataset](http://users.dsic.upv.es/~ybenajiba/downloads.html).<sup>[more](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp).</sup> And uses FastText's Arabic vectors for word embedding.

No. epochs: 10
Accuracy: 99.2%

## Example
#### Input
ماذا يفعل طلال عبد الهادي في دبي بعد ما رجع من برلين؟ كان يعمل هناك في شركة فولكسفاجن، صحيح؟
#### Output
![Example](https://raw.githubusercontent.com/HassanAzzam/Arabic-NER/master/example.png)
