# CardiffNLP-metaphors-for-Semeval-2022-Task2-Subtask-A-idiomaticity
Script used by the CardiffNLP-metaphor team to run the experiments of Semeval 2022 Task 2 Subtask A : This is a binary classification task that requires classifying sentences into either "Idiomatic" or "Literal".

## Data preprocessing 

The data are reformatted into a json file containing the zero shot and one shot data settings in a dictionary.

### Instance Format
- corpus :
- id :
- expression : target MWE
- position : list of offsets of the MWE occurring within the context of the core sentence
- context : core sentence

- additionalInformation:
  - context : nextSentence, previousSentence
  - broadContext : concatenated string of the previous sentence, core sentence and next sentence.
  - broadContextPosition :
  - broadContextPositionFirst :


## Classification scripts

### Script Options

Input options :
- tagged : boolean
-
- occurrences :
- pair input :

Model parameters :
- max sequence length
- 

Model id :
-
-
-


### Examples 

- Monolingual :
- Multilingual :
