# TTPDrill 0.3
TTPDrill focuses on developing automated and context-aware analytics of cyber threat intelligence to accurately learn attack patterns (TTPs) from commonly available CTI sources in order to timely implement cyber defense actions. It implements data and text mining approach that combines enhanced techniques of Natural Language Processing (NLP) and Information Retrieval (IR) to extract threat actions based on semantic rather than syntactic relationships. 

# Requirements
* Python 3
* stanford-corenlp jar

# Installation

* Clone this repository
  [GitHub](https://github.com/mpurba1/TTPDrill-0.3.git)  
* Add stanford-corenlp jar

# Notice
Copyright 2020 CyberDNA Center, UNC Charlotte

Please cite paper: https://dl.acm.org/doi/pdf/10.1145/3134600.3134646

# Problems when executing
If it is not able to instantiate the CoreNLPServer, just use the following command in the directory where you downlaod
the zip file containing the core-nlp jars (https://nlp.stanford.edu/software/stanford-corenlp-4.5.7.zip).
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```
