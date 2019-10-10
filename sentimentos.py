import nltk

nltk.download()


base = [('Essa capinha de celular é muito boa', 'alegria'),
        ('Gostei muito desta capinha de celular', 'alegria'),
        ('Capinha maravilhosa', 'alegria'),
        ('Que capinha bonita', 'alegria'),
        ('Abaixo do esperado', 'raiva'),
        ('Não gostei', 'raiva'),
        ('Desbotou na primeira semana', 'raiva'),
        ('Olha só essa capinha! ', 'alegria'),
        ('Material de baixa resistência', 'raiva'),
        ('Custo beneficio excelente', 'alegria'),
        ('A foto é diferente do produto', 'raiva')]


stopwordsnltk = nltk.corpus.stopwords.words('portuguese')


def fazstemmer(texto):
    """
    Deixa apenas os radicais das palavras
    """
    stemmer = nltk.stem.RSLPStemmer()
    frasessstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p))
                       for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming


frasescomstemming = fazstemmer(base)
#print(frasescomstemming)


def buscapalavras(frases):
    """
    Busca as palavras nas frases e separa das emoções
    """
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras


palavras = buscapalavras(frasescomstemming)
print(palavras)


def buscafrequencia(palavras):
    """
    Define a frequência que as palavras aparecem no banco de dados
    """
    palavras = nltk.FreqDist(palavras)
    return palavras


frequenciatreinamento = buscafrequencia(palavras)
#print(frequencia.most_common(50))


def busca_palavrasunicas(frequencia):
    """
    Cria um dicionário de palavras únicas
    """
    freq = frequencia.keys()
    return freq


palavrasunicas = busca_palavrasunicas(frequenciatreinamento)
#print(palavrasunicas)

#print(palavrasunicas)


def extraipalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicas:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas


caracteristicasfrase = extraipalavras(['am', 'nov', 'dia'])
print(caracteristicasfrase)

basecompleta = nltk.classify.apply_features(extraipalavras, frasescomstemming)
#print(basecompleta[15])

# constroi a tabela de probabilidade
classificador = nltk.NaiveBayesClassifier.train(basecompleta)
print(classificador.labels())
print(classificador.show_most_informative_features(5))
"""
Most Informative Features
                     cap = False           raiva : alegri =      2.6 : 1.0
                   celul = False           raiva : alegri =      1.4 : 1.0
                       a = False          alegri : raiva  =      1.2 : 1.0
                   difer = False          alegri : raiva  =      1.2 : 1.0
                   mater = False          alegri : raiva  =      1.2 : 1.0

Explicando o most Informative Features:

Primeira linha quando a palavra cap= false, ou seja, não aparece significa que tem a probabilidade de 2.6 vezes mais de ser uma frase de raiva
do que de alegria.

Segunda linha quando a palavra celul= false, não aparece significa que tem a probabilidade de 1.4 vezes mais de ser uma frase de raiva
do que de alegria
"""
