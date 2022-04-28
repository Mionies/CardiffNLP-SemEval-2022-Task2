# -*- coding: utf-8 -*-




import torch
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
#from sklearn.model_selection import train_test_split
import json
from simpletransformers.classification import ClassificationModel
import random
import copy

torch.cuda.empty_cache()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



def add_idiom_tokens(context,position):
    st_tok,end_tok = '<idiom>','</idiom>'
    cl = list(context)
    n=0
    for start,end in position:
        cl.insert(start+n,st_tok)
        n+=1
        cl.insert(end+n,end_tok)
        n+=1
    return ''.join(cl)

## Extract relevant fields from datafiles 

### Dataframes for classification experiments


def make_dataframe_from_json(data,occurrence ="multiple",context="sentence", label=True):
    datarows = []  
    for d in data:
        row = {
          'corpus':d["corpus"],
          'dataSplit':d["dataSplit"],
          'expression':d["expression_cased"],
          'language':d["additionalInformation"]["language"],
          'id':d["additionalInformation"]["semID"]
            }
        if label==True:
            row['label']=d["label"]
        if context=="sentence":
            row["context"]=d["context"]
            if occurrence=="first":
                p = d["position"][:1]
            elif occurrence=="multiple":
                p = d["position"]  
        elif context=="paragraph":
            row["context"]=d["additionalInformation"]["broadContext"]
            if occurrence=="first":
                p=[d["additionalInformation"]["firstPositionBroadContext"]]
            elif occurrence=="multiple":
                p=d["additionalInformation"]["positionBroadContext"]         
        row['tagged_context'] = add_idiom_tokens(row["context"],p)                
        datarows.append(row) 
    print(f'Loaded {len(datarows)} instances')
    df = pd.DataFrame(datarows)
    return df

## Format data for the classifier

def prepare_df_for_st(df, is_seq_pair = True, use_tagged_context=True):
    if is_seq_pair:
        if use_tagged_context==True:
            df = df[['tagged_context','expression','label']]
            df.rename(columns={"tagged_context": "text_a", "expression": "text_b", 'label':'labels'},inplace=True)
        else: 
            df = df[['context','expression','label']]
            df.rename(columns={"context": "text_a", "expression": "text_b", 'label':'labels'},inplace=True)
    else:
        if use_tagged_context==True:
            df = df[['tagged_context', 'label']]
            df.rename(columns={"tagged_context": "text", 'label':'labels'},inplace=True)
        else:
            df = df[['context', 'label']] 
            df.rename(columns={"context": "text", 'label':'labels'},inplace=True)
    labelmap = {'1':1,'0':0}
    df.labels = df.labels.apply(lambda x:labelmap[x])
    return df


def prepare_df_for_st_prediction(df, is_seq_pair = True, use_tagged_context=True):
    if is_seq_pair:
        if use_tagged_context==True:
            df = df[['tagged_context','expression']]
            df.rename(columns={"tagged_context": "text_a", "expression": "text_b"},inplace=True)
        else: 
            df = df[['context','expression']]
            df.rename(columns={"context": "text_a", "expression": "text_b"},inplace=True)
    else:
        if use_tagged_context==True:
            df = df[["tagged_context"]]
            df.rename(columns={"tagged_context": "text"},inplace=True)
        else:
            df = df[["context"]]
            df.rename(columns={"context": "text"},inplace=True)
    return df


## Code for Classification


### Run the classifier for unique sequence input or pair input




def Classification_one(df,dfe,dfall,params,training_args,dataset_names,language):
    results = {}
    dataset_names = ["semevalOneShot","semevalZeroShot"]
    for dataset in dataset_names:
        results[dataset]={}
        filtered_df = df[df.corpus==dataset]
        filtered_dfe = dfe[dfe.corpus==dataset]
        #split train-dev
        #if dataset=="semevalZeroShot" and language=="GL":
        if language=='GL':
            special = dfall[dfall.corpus==dataset]
            filtered_df_train = special[special.dataSplit=="train"]
            # Dangerous modification, bad implementation but not much time : it becomes sensitive to the order of dataset_names elements
            params["model_name"]["GL"]=params["model_name"]["ML"]
            params["model_id"]["GL"]=params["model_id"]["ML"]
        else:
            filtered_df_train = filtered_df[filtered_df.dataSplit=="train"]
        filtered_df_dev = filtered_df[filtered_df.dataSplit=="dev"]
        filtered_df_test_gl = filtered_df[filtered_df.dataSplit=="test_gl"]
        #split 
        filtered_df_eval = filtered_dfe[filtered_dfe.dataSplit=="eval"]
        filtered_df_test = filtered_dfe[filtered_dfe.dataSplit=="test"]
        filtered_df_eval = filtered_df_eval.sample(frac=1).reset_index(drop=True)
        filtered_df_test = filtered_df_test.sample(frac=1).reset_index(drop=True)
        filtered_df_dev = filtered_df_dev.sample(frac=1).reset_index(drop=True)
        filtered_df_test_gl = filtered_df_test_gl.sample(frac=1).reset_index(drop=True)
        filtered_df_train = filtered_df_train.sample(frac=1).reset_index(drop=True)
        # prepare columns for simpletransformers
        df_dev = prepare_df_for_st(filtered_df_dev, is_seq_pair = params["pair"], use_tagged_context=params["tagged"])
        df_train = prepare_df_for_st(filtered_df_train, is_seq_pair = params["pair"], use_tagged_context=params["tagged"])
        df_test_gl = prepare_df_for_st(filtered_df_test_gl, is_seq_pair = params["pair"], use_tagged_context=params["tagged"])
        df_eval = prepare_df_for_st_prediction(filtered_df_eval, is_seq_pair = params["pair"], use_tagged_context=params["tagged"])
        df_test = prepare_df_for_st_prediction(filtered_df_test, is_seq_pair = params["pair"], use_tagged_context=params["tagged"])
        print(f'=== For dataset {dataset} | Train with {len(df_train)} and test on {len(df_dev)} instances ===')
        for SEED in params["seeds"]:#,2,3]:
            # manually set random seed
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            # output file with results
            parameters = {"model_name_ml":params["model_name"]["ML"],"model_id_ml":params["model_id"]["ML"],
                          "model_name_en":params["model_name"]["EN"],"model_id_en":params["model_id"]["EN"],
                          "model_name_pt":params["model_name"]["PT"],"model_id_pt":params["model_id"]["PT"],
                          "model_name_gl":params["model_name"]["GL"],"model_id_gl":params["model_id"]["GL"],
                          "epoch":training_args['num_train_epochs'],"batch_size":training_args['train_batch_size'] ,
                          "learning_rate":training_args["learning_rate"], "tagged":params["tagged"], "context":params["context"],
                          "occurrences":params["occurrences"],"mask_finetuning":params["mask_finetuning"],
                          "max_seq_length":training_args['max_seq_length'], "split_by_language": params["split_by_language"],
                          "pair_input": params["pair"],
                                                    }
            RESULTS_PATH = os.path.join(dir,params["output_file"])
            print("===== Training for language = "+language+' ===============')
            print(params["model_name"][language])
            print(params["model_id"][language])
            model = ClassificationModel(params["model_name"][language], 
                                        params["model_id"][language], 
                                        num_labels=len(set(df_train.labels)), 
                                        use_cuda=True, 
                                        args=train_args)
            # Train the model
            model.train_model(df_train,  output_dir=dir+'/'+dataset) # eval_df=feval_df,
            if params["pair"]==False:
                dev_set = df_dev["text"].tolist()
                y_dev = df_dev.labels.tolist()
                test_gl_set = df_test_gl["text"].tolist()
                y_test_gl = df_test_gl.labels.tolist()
                #predict eval set and test set and store into results
                eval_set = df_eval.text.tolist()
                test_set = df_test.text.tolist()
            else:
                dev_set = df_dev[["text_a","text_b"]].values.tolist()
                y_dev = df_dev.labels.values
                test_gl_set = df_test_gl[["text_a","text_b"]].values.tolist()
                y_test_gl = df_test_gl.labels.values
                eval_set = df_eval[["text_a","text_b"]].values.tolist()
                test_set = df_test[["text_a","text_b"]].values.tolist()
            if len(dev_set)>0:   
                predictions, raw_outputs = model.predict(dev_set)
                res = classification_report(y_dev,predictions,digits=4)
                print(f'=== FOR SEED {SEED} === DEV SET')
                print(res)
                dev_set_ids = filtered_df_dev.id.tolist()
            else:
                predictions=[]
                dev_set_ids=[]
            if len(test_gl_set)>0:
                #For Galician
                predictions_gl, raw_outputs_gl = model.predict(test_gl_set)
                res = classification_report(y_test_gl,predictions_gl,digits=4)
                print(f'=== FOR SEED {SEED} === GALICIAN SET')
                print(res)
                f1_macro_gl = f1_score( y_test_gl, predictions_gl, average='macro' )
                test_gl_set_ids = filtered_df_test_gl.id.tolist()
            else:
                predictions_gl=[]
                f1_macro_gl = '-'
                test_gl_set_ids=[]
            if len(eval_set)>0:
                eval_set_ids = filtered_df_eval.id.tolist()
                predict_eval, raw_outputs = model.predict(eval_set)
            else:
                eval_set_ids = []
                predict_eval= []
            test_set_ids = filtered_df_test.id.tolist()
            predict_test, raw_outputs = model.predict(test_set)
            results[dataset][SEED]={"parameters":parameters,"dev":{"ids":dev_set_ids,"predictions":list(predictions)},"test_gl":{"ids":test_gl_set_ids,"predictions":list(predictions_gl), "f1_macro":f1_macro_gl},"eval":{"ids":eval_set_ids,"predictions":list(predict_eval)},"test":{"ids":test_set_ids,"predictions":list(predict_test)} }
    return results


## Split datasets by language, run the models and merge the results

def Classification_per_language(df,dfe,params,training_args,dataset_names):
    print("Classification with one model per language")
    print(params)
    dl = {}
    for l in ["EN","PT","GL"]:
        dl[l]=[ df[df.language==l],dfe[dfe.language==l]]
    for l in dl:
        print(l,len(dl[l][0]),len(dl[l][1]))
    res_ling = {}
    for l in dl:
        print(l)
        res_ling[l]=Classification_one(dl[l][0],dl[l][1],df,params,training_args,dataset_names,l)
    r =copy.deepcopy(res_ling["EN"])
    for shot in r:
        for seed in params["seeds"]:
            for dset in ["dev","eval","test"]:
                for i in res_ling["PT"][shot][seed][dset]["ids"]:
                    r[shot][seed][dset]["ids"].append(i)
                for i in res_ling["PT"][shot][seed][dset]["predictions"]:
                    r[shot][seed][dset]["predictions"].append(i)
            for dset in ["test","test_gl"]:
                r[shot][seed][dset]["ids"].extend(res_ling["GL"][shot][seed][dset]["ids"])
                for i in res_ling["GL"][shot][seed][dset]["predictions"]:
                    r[shot][seed][dset]["predictions"].append(i)
            r[shot][seed]["test_gl"]["f1_macro"]=res_ling["GL"][shot][seed]["test_gl"]["f1_macro"]
    return r



# Select the correct data according to the input format and context options

def TrainClassifier(params,classif_train_args): 
    if params["context"]=="paragraph":
        if params["occurrences"]=="multiple":
            df = df_paragraph_multiple
            df_eval= df_paragraph_multiple_eval
        elif params["occurrences"] == "first":
            df = df_paragraph_first
            df_eval= df_paragraph_first_eval 
    elif params["context"]=='sentence':
        if params["occurrences"]=="multiple":
            df = df_sentence_multiple
            df_eval= df_sentence_multiple_eval
        elif params["occurrences"] == "first":
            df = df_sentence_first
            df_eval= df_sentence_first_eval            
    dataset_names = list(df.corpus.unique()) 
    print('datasets: ',dataset_names)
    #store results for 2 corpora and 3 seeds
    if params["split_by_language"]==False:
        return Classification_one(df,df_eval,df,params,train_args,dataset_names, "ML")
    else:
        return Classification_per_language(df,df_eval,params,train_args,dataset_names)

# Set all the parameters
# not implemented : mask_finetuning, 3 sentences embedded with one embedding per sentence + concatenation.


models_ids = ["bert-base-cased",'bert-base-multilingual-cased', "bert-large-cased",'bert-large-multilingual-cased',"longformer-base-4096", "longformer-large-4096", "xlm-roberta-base","xlm-roberta-large"]
models_per_lg = {"gl":["dvilares/bertinho-gl-base-cased" ],
            "en":["bert-base-cased","bert-large-cased","longformer-base-4096", "longformer-large-4096"],
            "pt":["neuralmind/bert-base-portuguese-cased","neuralmind/bert-large-portuguese-cased"],
            "ml":['bert-base-multilingual-cased','bert-large-multilingual-cased',"xlm-roberta-base","xlm-roberta-large"]
            }

# training opts
training_parameters={
    "model_name" :{ "EN":"bert","ML":"bert","GL":"bert","PT":"bert"},
    "model_id" :{ "EN":"bert-base-cased","ML":"bert-base-multilingual-cased","GL":"dvilares/bertinho-gl-base-cased","PT":"neuralmind/bert-base-portuguese-cased"},
    "context" : "sentence",
    "occurrences":"multiple",
    "split_by_language":True,
    "pair":True,
    "mask_finetuning":False,
    "tagged":True,
    "output_file":"out_fifi",
    "english_model_name":"bert",
    "portuguese_model_name":"bert",
    "galician_model_name":"bert",
    "english_model_id":"bert-base-cased",
    "portuguese_model_id":"neuralmind/bert-base-portuguese-cased",
    "galician_model_id":"dvilares/bertinho-gl-base-cased"
}

train_args={
    # clean tensorboard events saving
    'tensorboard_dir':os.path.join('.','tensorboard_events'),
    # disable model saving
    'save_eval_checkpoints':False,
    'save_model_every_epoch':False,
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'special_tokens':["<idiom>","</idiom>"],
    'num_train_epochs': 1,
    'train_batch_size': 8,
    'learning_rate':2e-05,
    'tensorboard_dir':'events', # don't save tensorboard events in the Colab VM
    'max_seq_length':300, #512
    #'sliding_window':True
}
#train_args["special_tokens"].extend(q)


def add_labels_to_submission_table(t,dlab,dset):
  for x in t[1:]:
    if x[2]=="zero_shot":
      x[3]=dlab["semevalZeroShot"][dset][x[0]]
    elif x[2]=="one_shot":
      x[3]=dlab["semevalOneShot"][dset][x[0]]
  return t


def write_submission_file(t, filename):
  with open(filename, mode='w',encoding='utf8') as submissionf:
    subwriter = csv.writer(submissionf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for x in t:
      subwriter.writerow(x)



# Code for bash script


if __name__ == '__main__':
    import argparse 
    import csv
    import SubTask1Evaluator
    
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch",type=int)
    parser.add_argument("batch",type=int)
    parser.add_argument("length", help="max sequence length",type=int)
    parser.add_argument("lr", help="learning rate",type=float)
    parser.add_argument("-m","--model", default='bert')
    parser.add_argument("-id","--modelid", default='bert-base-multilingual-cased')
    parser.add_argument("-p","--pair", action="store_true")
    parser.add_argument("-c","--context",choices=["sentence","paragraph"], default="sentence")
    parser.add_argument("-o","--occurrences",choices=["first","multiple"], default="multiple")
    parser.add_argument("-t","--tagged",help="mark the target expression in the sequence",action="store_true")
    parser.add_argument("--lingsplit", help="uses 3 language models for the 3 languages of the corpus",action="store_true")
    parser.add_argument("--mask", help="finetune with mask training on unlabelled data",action="store_true")
    parser.add_argument("-output", default='out_fifi')
    parser.add_argument("--enid",help="English model id" ,default='bert-base-cased')
    parser.add_argument("--ptid",help="Portuguese model id" ,default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--glid",help="Galician model id", default='dvilares/bertinho-gl-base-cased')
    parser.add_argument("--men",help="English model type" ,default='bert')
    parser.add_argument("--mpt",help="Portuguese model type" ,default='bert')
    parser.add_argument("--mgl",help="Galician model type", default='bert')
    parser.add_argument("--seeds",help="seeds sepatated by ','", default='1')
    parser.add_argument("--shuffle",help="choose a set", default='1')


    args = parser.parse_args()
    #pass to the classifier dictionary
    if args.lingsplit == True:
        training_parameters["model_name"]={"ML":args.model,"EN":args.men,"PT":args.mpt,"GL":args.mgl}
        training_parameters["model_id"]={"ML":args.modelid,"EN":args.enid,"PT":args.ptid,"GL":args.glid}
    else:
        training_parameters["model_name"]={"ML":args.model,"EN":"-","PT":"-","GL":"-"}
        training_parameters["model_id"]={"ML":args.modelid,"EN":"-","PT":"-","GL":"-"}
    training_parameters["pair"]=args.pair
    training_parameters["context"]=args.context
    training_parameters["split_by_language"]=args.lingsplit
    training_parameters["occurrences"]=args.occurrences
    training_parameters["tagged"]=args.tagged
    training_parameters["mask_finetuning"]=args.mask
    training_parameters["seeds"]=args.seeds.split(',')
    training_parameters["seeds"] = [int(x) for x in training_parameters["seeds"]]
    train_args['num_train_epochs']=args.epoch
    train_args['train_batch_size']=args.batch
    train_args['max_seq_length']=args.length
    train_args['learning_rate']=args.lr
    
    print(training_parameters)
    print(train_args)

    dir = '.'
    json_file = "../semevalTaskA_dev_eval_shuffle_"+args.shuffle+".json"
    dati = json.load(open(os.path.join(dir, json_file),encoding='utf8'))
    print(dati.keys())
    print(dati["semevalZeroShot"][0])

    data = []
    test_eval=[]
    for x in dati:
        for y in dati[x]:
            if y["dataSplit"] in ["train","dev"]:
                data.append(y)
            else:
                test_eval.append(y)

    j = "semevalTaskA.json"
    da = json.load(open(os.path.join(dir, j),encoding='utf8'))


    gl = [x for x in da["semevalOneShot"] if x["additionalInformation"]["language"]=="GL" and x["dataSplit"]=="train"]

    for x in gl:
        kt1 = copy.deepcopy(x)
        kt1["dataSplit"]='test_gl'
        data.append(kt1)
        kt0=copy.deepcopy(kt1)
        kt0["corpus"]="semevalZeroShot"
        data.append(kt0)

    #random.shuffle(data)



    """## Format data
    Tags the first occurrence of the expression or multiple occurrence in the sentence context or the paragraph (Previous and Next sentence) context.
    """

    df_sentence_first = make_dataframe_from_json(data, occurrence="first",context="sentence",label=True)
    df_paragraph_first = make_dataframe_from_json(data, occurrence="first",context="paragraph",label=True)
    df_sentence_multiple = make_dataframe_from_json(data, occurrence="multiple",context="sentence",label=True)
    df_paragraph_multiple = make_dataframe_from_json(data, occurrence="multiple",context="paragraph",label=True)



    """### Process test and evaluation sets"""

    df_sentence_first_eval = make_dataframe_from_json(test_eval, occurrence="first",context="sentence",label=False)
    df_paragraph_first_eval = make_dataframe_from_json(test_eval, occurrence="first",context="paragraph",label=False)
    df_sentence_multiple_eval = make_dataframe_from_json(test_eval, occurrence="multiple",context="sentence",label=False)
    df_paragraph_multiple_eval = make_dataframe_from_json(test_eval, occurrence="multiple",context="paragraph",label=False)


    res = TrainClassifier(training_parameters, train_args)

    
    for seed in training_parameters["seeds"]:
        """### Link labels to ids"""
        dlab = {} 
        for shot in res:
          dlab[shot]={"dev":{},"eval":{},"test":{},"test_gl":{}}
          for e in dlab[shot]:
            for i,x in enumerate(res[shot][seed][e]["ids"]):
              dlab[shot][e][x]=res[shot][seed][e]["predictions"][i]
    
        """## Generate submission files, formated dev, test and eval predictions"""
        datadir = '../semeval_data_files'
        with open(datadir+"/dev_submission_format.csv", newline='',encoding='utf8') as csvfile:
            dataset = csv.reader(csvfile, delimiter=',', quotechar='"')
            dev = list(dataset)
        with open(datadir+"/eval_submission_format.csv", newline='',encoding='utf8') as csvfile:
            dataset = csv.reader(csvfile, delimiter=',', quotechar='"')
            evali = list(dataset)
        with open(datadir+"/test_submission_format.csv", newline='',encoding='utf8') as csvfile:
            dataset = csv.reader(csvfile, delimiter=',', quotechar='"')
            test = list(dataset)

        dev = add_labels_to_submission_table(dev,dlab,'dev')
        evali = add_labels_to_submission_table(evali,dlab,'eval')
        test = add_labels_to_submission_table(test,dlab,'test')
    
        write_submission_file(dev, "gen_submission_files_shuffle_"+args.shuffle+"/dev_"+str(train_args['num_train_epochs'])+'_'+str(train_args["train_batch_size"])+"_"+str(train_args["max_seq_length"])+"_"+training_parameters["occurrences"]+"_"+training_parameters["context"]+"_"+str(training_parameters["pair"])+"_"+str(training_parameters["tagged"])+"_"+str(training_parameters["split_by_language"])+"_"+training_parameters["model_id"]["ML"]+"_"+training_parameters["model_id"]["EN"]+'_'+str(train_args["learning_rate"])+"_"+str(seed)+".csv")
        write_submission_file(evali, "gen_submission_files_shuffle_"+args.shuffle+"/eval_"+str(train_args['num_train_epochs'])+'_'+str(train_args["train_batch_size"])+"_"+str(train_args["max_seq_length"])+"_"+training_parameters["occurrences"]+"_"+training_parameters["context"]+"_"+str(training_parameters["pair"])+"_"+str(training_parameters["tagged"])+"_"+str(training_parameters["split_by_language"])+"_"+training_parameters["model_id"]["ML"]+"_"+training_parameters["model_id"]["EN"]+'_'+str(train_args["learning_rate"])+"_"+str(seed)+".csv")
        write_submission_file(test, "gen_submission_files_shuffle_"+args.shuffle+"/test_"+str(train_args['num_train_epochs'])+'_'+str(train_args["train_batch_size"])+"_"+str(train_args["max_seq_length"])+"_"+training_parameters["occurrences"]+"_"+training_parameters["context"]+"_"+str(training_parameters["pair"])+"_"+str(training_parameters["tagged"])+"_"+str(training_parameters["split_by_language"])+"_"+training_parameters["model_id"]["ML"]+"_"+training_parameters["model_id"]["EN"]+'_'+str(train_args["learning_rate"])+"_"+str(seed)+".csv")

        """
        ### Write 6 lines of csv file per run (PT,EN and no GL in devset)
        """
    
        submission_file = "gen_submission_files_shuffle_"+args.shuffle+"/dev_"+str(train_args['num_train_epochs'])+'_'+str(train_args["train_batch_size"])+"_"+str(train_args["max_seq_length"])+"_"+training_parameters["occurrences"]+"_"+training_parameters["context"]+"_"+str(training_parameters["pair"])+"_"+str(training_parameters["tagged"])+"_"+str(training_parameters["split_by_language"])+"_"+training_parameters["model_id"]["ML"]+"_"+training_parameters["model_id"]["EN"]+'_'+str(train_args["learning_rate"])+"_"+str(seed)+".csv"


        gold_file = "../semeval_data_files/dev_gold.csv"
        reval = SubTask1Evaluator.evaluate_submission( submission_file, gold_file )
        reval2 = copy.deepcopy(reval)
        paramlist = ["seed",'epoch', 'batch_size', 'learning_rate', 'tagged', 'context', 'occurrences', 'mask_finetuning', 'max_seq_length', 'split_by_language', 'pair_input', "model_name_en", "model_name_pt" , "model_name_gl", "model_name_ml","model_id_en", "model_id_pt", "model_id_gl", "model_id_ml"]
        reval.append(["zero_shot","GL",res["semevalZeroShot"][seed]["test_gl"]["f1_macro"]])
        reval.append(["one_shot","GL",res["semevalOneShot"][seed]["test_gl"]["f1_macro"]])
        reval2 = copy.deepcopy(reval)
        reval[0].extend(paramlist)
        #reval[0].extend(["dev_size","galician_test_size"])
    
        if training_parameters["split_by_language"]==True:
            '''Fix of the dirty implementation of the case zero_shot split by language for Galician ->
            the galician perf are tested on a multiglingual model trained on english and portuguese
            but for the one_shot, we have galician training data so we train using te monolingual galician model'''
            res["semevalOneShot"][seed]['parameters']["model_id_gl"]="dvilares/bertinho-gl-base-cased"
    
        for x in reval[1:]:
            if x[0]=="zero_shot":
                for p in paramlist:
                    if p=='seed':
                        x.append(seed)
                    else:
                        x.append(res["semevalZeroShot"][seed]['parameters'][p])
            elif x[0]=="one_shot":
                for p in paramlist:
                    if p=='seed':
                        x.append(seed)
                    else:
                        x.append(res["semevalOneShot"][seed]['parameters'][p])
    
    
        k = reval[-1][3:]
        k = [str(x) for x in k]
        k = "_".join(k[:-2])
        k = k.replace("/",'_')
    
    
        s =open("result_csv_shuffle_"+args.shuffle+"/"+k+".csv",'w')
    
        for row in reval :
            s.write( '\t'.join( [ str( i ) for i in row ] )+'\n' ) 
            print( '\t'.join( [ str( i ) for i in row ] )+'\n' ) 
    
        s.close()
    
        heads = []
        for x in reval2[1:]:
          heads.append("_".join([x[0],x[1],"f1_macro"]))
    
        val = [x[2] for x in reval2[1:]]
    
        oneline = [[],[]]
        paramlist = ["seed",'epoch', 'batch_size', 'learning_rate', 'tagged', 'context', 'occurrences', 'mask_finetuning', 'max_seq_length', 'split_by_language', 'pair_input', "model_name_en", "model_name_pt" , "model_name_gl", "model_name_ml","model_id_en", "model_id_pt", "model_id_gl", "model_id_ml"]
        oneline[0].extend(heads)
        oneline[1].extend(val)
        oneline[0].extend(paramlist)
        for x in paramlist:
            if x=="model_id_gl" and training_parameters["split_by_language"]==True:
                '''Fix of the dirty implementation of the case zero_shot split by language for Galician ->
                the galician perf are tested on a multiglingual model trained on english and portuguese
                but for the one_shot, we have galician training data so we train using te monolingual galician model'''
                i = res["semevalOneShot"][seed]['parameters']["model_id_ml"]+"|"+"dvilares/bertinho-gl-base-cased"
                oneline[1].append(i)
            elif x=='seed':
                oneline[1].append(seed)
            else:
                oneline[1].append(res["semevalOneShot"][seed]['parameters'][x])
    
    
        s = open("results_on_1_line_shuffle_"+args.shuffle+"/"+k+".csv",'w')
        for row in oneline :
            print( '\t'.join( [ str( i ) for i in row ] )+'\n' ) 
            s.write( '\t'.join( [ str( i ) for i in row ] )+'\n' ) 
        s.close()
