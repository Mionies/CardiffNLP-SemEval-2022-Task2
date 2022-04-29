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

### Example
```
{
 
}
```


## Classification scripts

### Script Options

Model parameters :
- number of epochs
- batch size
- max sequence length
- learning rate

Model:
- --lingsplit : train one classifier per language, <boolean>. If false, main model is used.
- -m : main model name
- -id : main model id
- -eid : english model name
- -pid : portuguese model name
- -gid : galician model name


Input options :
- -p :pair input <boolean>
- -t : tagged <boolean>
- -o : occurrences <multiple, first>




### Examples 


 
```
python3 classify.py 4 16 512 -m roberta -id xlm-roberta-large -p -c sentence -o first -t
python3 classify.py 3 8 350 -m bert -id bert-base-cased -p -c sentence -o first -t --lingsplit
```
- Experiment 1
```
python3 completed_classif.py 3 8 512 4e-05 -m xlmroberta -id xlm-roberta-base -p -c sentence -o multiple  --seeds 1,2,3 --shuffle 1
python3 completed_classif.py 3 8 512 4e-05 -m xlmroberta -id xlm-roberta-base -p -c sentence -o multiple  --seeds 1,2,3 --shuffle 2
python3 completed_classif.py 3 8 512 4e-05 -m xlmroberta -id xlm-roberta-base -p -c sentence -o multiple  --seeds 1,2,3 --shuffle 3

python3 completed_classif.py 1 64 128 4e-05 -m bert -id bert-base-multilingual-cased -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 1    
python3 completed_classif.py 1 64 128 4e-05 -m bert -id bert-base-multilingual-cased -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 2 
python3 completed_classif.py 1 64 128 4e-05 -m bert -id bert-base-multilingual-cased -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 3  
```
- Experiment 2
```
```
- Experiment 3
```
```


