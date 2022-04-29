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
			"id": "train_zero_shot.EN.47.1",
			"corpus": "semevalZeroShot",
			"context": "Even though there were no judges, parents, or spotlight of being on the stage, Hasbrouck Heights Middle School sixth grader Enric Soni won the annual spelling bee by answering 39 of 40 questions correctly.",
			"expression": "spelling bee",
			"label": "1",
			"position": [
				[
					150,
					162
				]
			],
			"dataSplit": "train",
			"expression_cased": "spelling bee"
			"additionalInformation": {
				"context": {
					"nextSentence": "He will move on to represent Hasbrouck Heights in the North Jersey Spelling Bee.The North Jersey Spelling Bee will include the counties of Bergen and Passaic only, according to the event's website.",
					"previousSentence": "This year's spelling bee was done virtually as an online test through the Scripps National Spelling Bee program, according to Hasbrouck Heights Middle School Social Studies teacher James Muska, who organizes the event."
				},
				"language": "EN",
				"setting": "zero_shot",
				"semID": "train_zero_shot.EN.47.1",
				"broadContext": "This year's spelling bee was done virtually as an online test through the Scripps National Spelling Bee program, according to Hasbrouck Heights Middle School Social Studies teacher James Muska, who organizes the event. Even though there were no judges, parents, or spotlight of being on the stage, Hasbrouck Heights Middle School sixth grader Enric Soni won the annual spelling bee by answering 39 of 40 questions correctly. He will move on to represent Hasbrouck Heights in the North Jersey Spelling Bee.The North Jersey Spelling Bee will include the counties of Bergen and Passaic only, according to the event's website.",
				"positionBroadContext": [
					[
						12,
						24
					],
					[
						91,
						103
					],
					[
						369,
						381
					],
					[
						492,
						504
					],
					[
						522,
						534
					]
				],
				"firstPositionBroadContext": [
					369,
					381
				]
			},
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


- Experiment 1
```
python3 classifier.py 1 8 128 4e-05 -m xlmroberta -id xlm-roberta-base -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 1
python3 classifier.py 1 8 128 4e-05 -m xlmroberta -id xlm-roberta-base -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 2
python3 classifier.py 1 8 128 4e-05 -m xlmroberta -id xlm-roberta-base -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 3

python3 classifier.py 1 64 128 4e-05 -m bert -id bert-base-multilingual-cased -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 1    
python3 classifier.py 1 64 128 4e-05 -m bert -id bert-base-multilingual-cased -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 2 
python3 classifier.py 1 64 128 4e-05 -m bert -id bert-base-multilingual-cased -p -c sentence -o multiple -t --seeds 1,2,3 --shuffle 3  
```
	
- Experiment 2
	
```
python3 classifier.py 3 8 512 4e-05 -m xlmroberta -id xlm-roberta-base -p -c sentence -o multiple  --seeds 1,2,3 --shuffle 1 
python3 classifier.py 3 8 512 4e-05 -m xlmroberta -id xlm-roberta-base -p -c paragraph -o multiple  --seeds 1,2,3 --shuffle 1    
python3 classifier.py 3 8 512 4e-05 -m xlmroberta -id xlm-roberta-base -c sentence -o multiple  --seeds 1,2,3 --shuffle 1 
python3 classifier.py 3 8 512 4e-05 -m xlmroberta -id xlm-roberta-base -c paragraph -o multiple --seeds 1,2,3 --shuffle 1    
	
```
- Experiment 3
	
```
python3 classifier.py	
	
```
	
- Experiment 4
	
```
python3 classifier.py	
```
