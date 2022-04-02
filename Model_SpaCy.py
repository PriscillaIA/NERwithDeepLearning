#!/usr/bin/env python
# coding: utf-8

# # Importações
from pickletools import TAKEN_FROM_ARGUMENT1
import re
import spacy
import pickle
import os
import pandas as pd
import csv
import glob
import csv
from spacy.training import offsets_to_biluo_tags
from spacy.training import biluo_tags_to_spans
from spacy.training import offsets_to_biluo_tags, biluo_tags_to_offsets, biluo_tags_to_spans
import itertools
from sklearn.model_selection import train_test_split
import sys
# importações para o bloco de código do treinamento do modelo de NER
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.tokens import Doc
from spacy.training import Example
import warnings
import datetime as date
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY, LIST_HYPHENS
from spacy.lang.char_classes import LIST_ICONS, HYPHENS, CURRENCY
from spacy.lang.char_classes import CONCAT_QUOTES, ALPHA_LOWER, ALPHA_UPPER, ALPHA


# ### Carregando os Dados
# Na marcação IOB o prefixo B- antes de uma tag indica que a tag é o início de um pedaço, e um prefixo I- antes de uma tag indica que a tag está dentro de um pedaço. A tag B é usada somente quando uma tag é seguida por uma tag do mesmo tipo sem tokens O entre elas. Uma tag O indica que um token não pertence a nenhuma entidade/pedaço.
# O dataset possui 5 atributos: Palavras (são os tokens que frmam a base de dados rotulada manualmente); Rotulo (são as categorias de entidades nomeadas seguidas pelo prefixo do tipo de marcação IOB; Sentenca (um número que represeta a sentença a que determinado token pertence); Inicio (número que representa o inicio de um token dentro de sua respctiva sentença); Fim (número que representa o fim de um token dentro de sua respctiva sentença).

#Carrega o dataset IOB, ou seja, os tokens são rotulados com tags do tipo marcação IOB.
dataSet = pd.read_csv("Dataset_Padrao_IOB_Versao4.csv", usecols=['Palavras', 'Rotulo', 'Sentenca', 'Inicio', 'Fim'])
print("Quantidade de tokens no DataSet_IOB",len(dataSet))

##Carrega o dataset utilizada anteriormente na rotulação manual das categorias de entidades nomeadas. 
##Esse arquivo é utilizado para extrairmos as setenças a que cada token rotulado no dataset IOB pertence. Essa informação é...
#utilizada no código de conversão do dados no formato de marcação IOB para o formato compreendido pelo spaCy.
with open("Dados_Utilizados_Para_RotulacaoManual.txt", "r", encoding='utf-8') as file:
    textos = file.read().splitlines()
print("Quantidade de sentenças do dataSet utilizado para rotulação manual e que originou o dataset IOB:", len(textos))



# ### Convertendo os dados do formato IOB para o formato reconhecido pelo spaCy
sMarker = 0 #Marcador que referência a qual setença a entidade pertence, ou seja, serve pra pegar o texto da referencia dos tokens no dataset
s = textos[sMarker]
dataSetEntrada = []
entities = []
for index, row in dataSet.iterrows():
    
    if(row['Sentenca']-1 != sMarker): # Nova sentença
        if entities:
            dataSetEntrada.append((s, {"entities": [tuple(e) for e in entities]})) # Salva sentença anterior
        entities = [] # esvazia entidades
        sMarker = row['Sentenca']-1 # atualiza o marcador de sentença
        
        if (sMarker < len(textos)): # Limite de textos
            s = textos[sMarker]
            
    if(row['Rotulo'][0] == 'O'):
         entities.append([row['Inicio'], row['Fim'], row['Rotulo'][2:]])
            
    if(row['Rotulo'][0] == 'B'):
         entities.append([row['Inicio'], row['Fim'], row['Rotulo'][2:]])
            
    if(row['Rotulo'][0] == 'I'):
        if (entities): 
            entities[-1][1] = row['Fim']
        else:
            # print(index)
            entities.append([row['Inicio'], row['Fim'], row['Rotulo'][2:]])
            
    if index == dataSet.index[-1]: # Ultimo elemento
        if entities:
            dataSetEntrada.append((s, {"entities": [tuple(e) for e in entities]}))
        entities = [] 

print('Quantidade de sentenças rotuladas e não rotuladas utilizadas no treinamento do modelo:', len(dataSetEntrada))



# ### Dividindo os dados entre treino e teste
# https://www.kaggle.com/slavaz/spacy-ner-ensemble

# ## Importando modelo
# Carregando o modelo linguistico em português do spaCy (salvo em nlp). Esse modelo já foi baixado anteriormente
nlp = spacy.load('pt_core_news_sm')
print ("Modelo carregado '% s'"% nlp)

# ## Customizando o Tokenizador para o modelo de nlp em português
# Essa função cria um objeto customizado da classe Tokenizer do spacy, para alinhamento dos tokens com a rotulações em alguns casos especiais por exemplo: '-' e '$'.
def custom_tokenizer(nlp):    
    infixes = (
        spacy.lang.char_classes.LIST_ELLIPSES
        + spacy.lang.char_classes.LIST_ICONS
        + ["\.", "\/", ":", "\(", "\)"]
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[\.])[\.](?=[\.])",
            r"(?<=[{a}{q}])\.(?=[{a}{q}])".format(
                a = ALPHA, q = CONCAT_QUOTES),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}])(?:{h}|\/)(?=[{a}])".format(a=ALPHA, h=LIST_HYPHENS),
            r"(?<=[{a}])(?:{h}|\/)(?=[0-9])".format(a=ALPHA, h=LIST_HYPHENS),
            r"(?<=[0-9])(?:{h}|\/)(?=[{a}])".format(a=ALPHA, h=LIST_HYPHENS),
            r"(?<=[0-9])(?:{h}|\/|,|\.)(?=[0-9])".format(a=ALPHA, h=LIST_HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}\"]),(?=[0-9{a}])".format(a=ALPHA),
        ]
    )
    list_quotes = ["\\'", '"', '`',
                     '‘', '´', '’', '‚', ',',
                     '„', '»', '«', '「', '」',
                     '『', '』', '（', '）', '〔',
                     '〕', '【', '】', '《', '》',
                     '〈', '〉']
    suffixes = (
        LIST_PUNCT
        + LIST_HYPHENS
        + list_quotes
        + LIST_ICONS
        + [ "\/", ":", "\(", "\)"]
        + ["'s", "'S", "’s", "’S", "—", "–"]
        + [
            r"(?<=[0-9])\+",
            r"(?<=[0-9])(?:{c})".format(c=CURRENCY),
            r"(?<=[0-9a-z(?:{q})])\.".format(
            al=ALPHA_LOWER, q=CONCAT_QUOTES),
            r"(?<=[{au}])\.".format(au=ALPHA_UPPER),
        ] 
    )
    
    prefixes = (
        LIST_PUNCT
        + [ "\/", ":", "\(", "\)"]
        + LIST_HYPHENS
        + list_quotes
        + LIST_CURRENCY
        + LIST_ICONS
    )

    infix_re = spacy.util.compile_infix_regex(infixes)
    suffix_re = spacy.util.compile_suffix_regex(suffixes)
    prefix_re = spacy.util.compile_prefix_regex(prefixes)
    custom = Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=None)
    special_case = [{ORTH: "."}, {ORTH: "."}, {ORTH: "."}]        # Adicionando uma regra especial
    custom.add_special_case("...", special_case)
    return custom



# atribuindo o tokenizador customizado ao modelo nlp
nlp.tokenizer = custom_tokenizer(nlp)
# ### Configurando parâmetros para o treinamento do modelo
# Obtendo o componente do pipeline (ner) para trabalhar com reconhecimentos de entidades nomeadas
ner = nlp.get_pipe('ner')
# Adicionando as categorias de entidade nomeada ao pipeline 'ner' manualmente
ner.add_label('CARACTERISTICA_SENSORIAL_AROMA')
ner.add_label('CARACTERISTICA_SENSORIAL_CONSISTÊNCIA')
ner.add_label('CARACTERISTICA_SENSORIAL_SABOR')
ner.add_label('CARACTERISTICA_SENSORIAL_COR')
ner.add_label('RECIPIENTE_ARMAZENAMENTO')
ner.add_label('EQUIPAMENTO_DESTILACAO')
ner.add_label('CLASSIFICACAO_BEBIDA')
ner.add_label('TEMPO_ARMAZENAMENTO')
ner.add_label('GRADUACAO_ALCOOLICA')
ner.add_label('TIPO_MADEIRA')
ner.add_label('NOME_BEBIDA')
ner.add_label('VOLUME')
ner.add_label('NOME_LOCAL')
ner.add_label('NOME_ORGANIZACAO')
ner.add_label('NOME_PESSOA')
ner.add_label('PRECO')
ner.add_label('TEMPO')
#print("Categorias de entidade que o modelo contém:", '\n\n', ner.label_data)
# outra maneira mais rápida de adicionar as categorias de entidades ao 'ner'
#ner = nlp.get_pipe('ner')
#for texto, annotations in TRAIN_DATA:
    #for ent in annotations.get("entities"):
        #ner.add_label(ent[2])     
        #print(ent[2])
#print("Categorias de entidade que o modelo contém:", '\n\n', ner.label_data) #verificando as categorias de entidades nomeadas contidas no pipeline
# Obtendo os nomes dos componentes que queremos desativar durante o treinamento
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

#Dividindo a base em treino e teste
#TRAIN_DATA,TEST_DATA = train_test_split(dataSetEntrada, train_size=0.80,shuffle=True)
#print('Quantidade de dados para TREINO do modelo:',len(TRAIN_DATA))
#print('Quantidade de dados para TESTE do modelo:',len(TEST_DATA))
TRAIN_DATA = dataSetEntrada[0:100]
TEST_DATA = dataSetEntrada[101:200]

# ### Treinando o modelo de reconhecimento de entidades nomeadas
# O código abaixo realiza o treinamento do modelo de reconhecimento de entidades nomeadas
primeiraEntradaTerminal =  int(sys.argv[1])
segundaEntradaTerminal =  float(sys.argv[2])
terceiraEntradaTerminal =  float(sys.argv[3])
quartaEntradaTerminal =  float(sys.argv[4])
quintaEntradaTerminal =  float(sys.argv[5])
listValuesOfLost = [] #lista para guardar o somatório das percas (losses) de cada lote (batch) treinado e testado internamente pelo algorimto 
epochs = primeiraEntradaTerminal
optimizer = nlp.create_optimizer()

with nlp.disable_pipes(*unaffected_pipes), warnings.catch_warnings():#inicia a execução do algoritmo desabilitando os pipelines que não devem sofrer alterações
    warnings.filterwarnings("once", category=UserWarning, module='spacy') #mositra os erros identificados pelo algoritmo
    start=str(date.datetime.now())
    print(date.datetime.now())
    sizes = compounding(segundaEntradaTerminal, terceiraEntradaTerminal, quartaEntradaTerminal) 
    count=0
    for epoch in range(epochs):
        examples = TRAIN_DATA
        random.shuffle(examples)
        batches = minibatch(examples, size=sizes) #agrupa os exemplos usando o minibatc do spaCy
        losses = {}   
        count_1 = 0
        count=count+1
        print("EPOCH: ", count) 
        for batch in batches: #percorre cada lote de dados para executar o treinamento com os dados rotulados (TRAIN_DATA)
            texts, annotations = zip(*batch)
            count_1=count_1+1
            print('LOTE EM  QUESTÃO', count_1)
            print('QUANTIDADE DE DADOS NO LOTE', len(batch))
            print(batch)
            example = []
            # Atualize o modelo com a iteração de cada texto
            count_2 = 0
            for i in range(len(texts)):
                doc = nlp.make_doc(texts[i])
                count_2=count_2+1
                print('NÚMERO DA SENTENÇA SOB ANALISE:', count_2)
                print('TEXTO SOB ANALISE: ', doc)
                #print("Rotulacao DOC--->", doc)
                #print('----------------------------------BREAK------------------------------------------------')
                example.append(Example.from_dict(doc, annotations[i]))
                print('TAMNHO DO EXEMPLO', len(example))
                print(example)
                # Update the model
                nlp.update(example, drop=quintaEntradaTerminal, losses=losses,sgd=optimizer)
            #print('EXEMPLO--->', example)
            #print('----------------------------------BREAK------------------------------------------------')
            print('Update realizado...')
        print("Losses ({}/{})" .format(epoch + 1, epochs), '-', date.datetime.now(), losses)        
        #esta lista é utilizada para salvar as informações das percas por epoch
        numeracaoEpoca="Epoch ({}/{})" .format(epoch + 1, epochs)
        perda='Some of losses'+str(losses)
        listValuesOfLost.append((numeracaoEpoca,perda))
    end=str(date.datetime.now())
    print(date.datetime.now())
    print('Processo de treinamento finalizado!')




# ### Comparação Rotulação Manual e do Modelo
# Compara item por item como o modelo classificou as sentenças e como elas foram classificadas manual (ponto de referencia).
#nlp = spacy.load(output_dir)
examples = []
for text, annots in TEST_DATA:
    #print(f'Manual Anotation: {annots}')
    #doc = nlp(text)
    doc = nlp.make_doc(text)
    #spacy.displacy.render(nlp(text))  
    #for item in doc.ents:
        #print('Model prediction: ', 'entities: ', [(item.start_char, item.end_char,item.label_)])                
    examples.append(Example.from_dict(doc, annots)) 
    #print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

# ### Métricas Precision, Recall e f-measure: 
# Apresenta o resultados das métricas para o modelo como um todo, e as métricas do modelo para cada categoria de entidade
calculatedBySpacy = nlp.evaluate(examples)
print(calculatedBySpacy)

# ### Avaliação do Modelo pelas Métrica Acuracia
# dictionary which will be populated with the entities and result information
entity_evaluation = {}
# helper function to udpate the entity_evaluation dictionary
def update_results(entity, metric):
    if entity not in entity_evaluation:
        entity_evaluation[entity] = {"correct": 0, "total": 0}  
    entity_evaluation[entity][metric] += 1
# same as before, see if entities from test set match what spaCy currently predicts
for data in TEST_DATA:
    sentence = data[0]
    entities = data[1]["entities"]
    for entity in entities:
        doc = nlp(sentence)
        correct_text = sentence[entity[0]:entity[1]]
        for ent in doc.ents:
            if ent.label_ == entity[2] and ent.text == correct_text:
                update_results(ent.label_, "correct")
                break
        update_results(entity[2], "total")
print("Resultado para cada categoria:",'\n', entity_evaluation)

sum_total = 0
sum_correct = 0
accuracy_by_category = {}
for entity in entity_evaluation:
    total = entity_evaluation[entity]["total"]
    correct = entity_evaluation[entity]["correct"]
    sum_total += total
    sum_correct += correct
    percentual_acertos_modelo = correct/total*100
    accuracy_by_category[entity] = {str(percentual_acertos_modelo)+'%'}
    print("{} | {:.2f}%".format(entity, correct / total * 100))
sum_accuracy_total = sum_correct / sum_total * 100
sum_accuracy_total = str(sum_accuracy_total)+'%'
print("Accuracy by Category: {:.2f}%".format(sum_correct / sum_total * 100))


#Salvando informações geradas relevantes em um arquivo.txt
Caminho_e_nome_arquivo = sys.argv[6]+'.txt'

with open (Caminho_e_nome_arquivo,"w", encoding='utf-8')  as output:
    quantidadeDados = str(len(dataSetEntrada))
    output.write('Quantidade de Sentenças a serem utilizada para treino e teste: '+quantidadeDados+'\n')
    #output.write('\n'.join(textos))
    output.write('\n') 
    for line in listValuesOfLost: #escreve no documento as informações sobre a perca do modelo.
        output.writelines(line)
        output.write('\n') 
    #Métricas geradas pelo método 'evaluat', disponibilizado pelo próprio spaCy.
    token_acc = '\n'+"Accuracy geral dos tokens classificados : "+str(calculatedBySpacy.get("token_acc"))+'\n'
    token_p = "Precisão geral dos tokens classificados : "+str(calculatedBySpacy.get("token_p"))+'\n'
    token_r = "Revocaçãodo geral dos tokens classificados : "+str(calculatedBySpacy.get("token_r"))+'\n'
    token_f = "F-measure geral dos tokens classificados : "+str(calculatedBySpacy.get("token_f"))+'\n'
    ents_p = "Precisão geral das entidade classificadas : "+str(calculatedBySpacy.get("ents_p"))+'\n'
    ents_r = "Revocação geral das entidade classificadas : "+str(calculatedBySpacy.get("ents_r"))+'\n'
    ents_f = "F-measure geral das entidade classificadas : "+str(calculatedBySpacy.get("ents_f"))+'\n'
    ents_per_type = calculatedBySpacy.get("ents_per_type")
    output.write(token_acc), output.write(token_p), output.write(token_r), output.write(token_f)
    output.write(ents_p),output.write(ents_r),output.write(ents_f)
    #output.writelines(';'.join(str(x) for x in (token_acc,token_p)))
    output.writelines('\n'+"|ABAIXO TEMOS AS MÉTRICAS CALCULADAS PARA CADA CATEGORIA:|"+'\n')
    for key in ents_per_type:
        ents_por_tipo = str(key)+str(ents_per_type[key])+'\n'
        output.write(ents_por_tipo) 
    #Total de acertos e erros do modelo por categoria.
    output.writelines('\n'+"|ABAIXO TEMOS O TOTAL DE ERROS E ACERTOS DO MÉTODO PARA CADA CATEGORIA:|"+'\n')
    for key in entity_evaluation:
        acertos_erros = str(key)+str(entity_evaluation[key])+'\n'
        output.write(acertos_erros)
    #Acuracia do modelo por categoria.
    output.writelines('\n'+"|ABAIXO TEMOS A ACURÁCIA CALCULADA PARA CADA CATEGORIA:|"+'\n')
    for key in accuracy_by_category:
        entity_evaluation_accuracy = str(key)+str(accuracy_by_category[key])+'\n'
        output.write(entity_evaluation_accuracy)
    output.write(str(sum_accuracy_total))
    #Parametros digitados no terminaç pelo usuario
    output.write('\n'+"|ABAIXO TEMOS OS PARAMETROS INFORMADOS NO TEMRINAL:|"+'\n')
    output.writelines(';'.join(str(x) for x in (primeiraEntradaTerminal,segundaEntradaTerminal,terceiraEntradaTerminal,quartaEntradaTerminal,quintaEntradaTerminal)))
    #Salvando hora inicio e fim execução do algoritmo
    output.writelines('\n'+"|HORA DE INICIO E FIM DE EXECUÇÃO DO ALGORITMO:|"+'\n')
    output.write(start+'\n')
    output.write(end)


# ## Salvando o Modelo
#from pathlib import Path
output_dir=Path(Caminho_e_nome_arquivo+'modelo')
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

# ## Importando o Modelo
#print("Loading from", output_dir)
#nlp_updated = spacy.load(output_dir)
#doc = nlp_updated("Eu gostei da Cachaça Artesanal criada em Minas Gerais" )
#print("Entidades", [(ent.text, ent.label_) for ent in doc.ents])