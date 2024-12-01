import os
import chromadb
import requests
import json

from sentence_transformers import SentenceTransformer

from ..utils.utils import FuncaoEmbeddings
from ..environment.environment import environment
URL_LLAMA = 'http://localhost:11434/api/generate'
MODELO_LLAMA='llama3.1'
URL_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../conteudo"))
EMBEDDING_INSTRUCTOR="hkunlp/instructor-xl"
URL_BANCO_VETORES=os.path.join(URL_LOCAL,"bancos_vetores/banco_vetores_regimento_resolucoes_rh")
NOME_COLECAO='regimento_resolucoes_rh'
DEVICE='cuda'

class GeradorPerguntas:
    def __init__(self,
                 url_llama=URL_LLAMA,
                 modelo_llama=MODELO_LLAMA,
                 url_local=URL_LOCAL,
                 nome_modelo=EMBEDDING_INSTRUCTOR,
                 url_banco_vetores=URL_BANCO_VETORES,
                 nome_colecao=NOME_COLECAO,
                 device=DEVICE):
        self.URL_LLAMA = url_llama
        self.MODELO_LLAMA = modelo_llama
        self.URL_LOCAL = url_local
        self.NOME_MODELO = nome_modelo
        self.URL_BANCO_VETORES = url_banco_vetores
        self.NOME_COLECAO = nome_colecao
        self.DEVICE = device

    def gerar_perguntas(self, artigo, contexto):
        prompt = '''Considere o artigo abaixo. Crie pelo menos 5 perguntas que possam ser respondidas com fragmentos do artigo. A saída deve ser uma lista de objetos JSON, com os atributos {{"pergunta": "Texto da pergunta Gerada", "resposta": "fragmento do artigo que responde a pergunta"}}. Não adicione nada na resposta, exceto a lista de objetos JSON, sem qualquer comentário adicional. ARTIGO: {}'''.format(artigo)
        payload = {
            "model": MODELO_LLAMA,
            "prompt": prompt,
            "temperature": 0.0,
            "context": contexto
        }
        resposta = requests.post(self.URL_LLAMA, json=payload, stream=True)
        resposta.raise_for_status()
        texto_resposta = ''
        for fragmento in resposta.iter_content(chunk_size=None):
            if fragmento:
                dados = json.loads(fragmento.decode())
                texto_resposta += dados['response']
        return texto_resposta

    def run(self, url_arquivo_saida='documentos_perguntas.json', carregar_arquivo=False):
        if carregar_arquivo:
            print(f'Carregando {url_arquivo_saida}')
            with open(url_arquivo_saida, 'r', encoding='utf-8') as arq:
                documentos = json.load(documentos)
        else:
            print(f'Consultando documentos do banco de vetores')
            client = chromadb.PersistentClient(path=self.URL_BANCO_VETORES)
            funcao_de_embeddings_sentence_tranformer = FuncaoEmbeddings(nome_modelo=self.NOME_MODELO, tipo_modelo=SentenceTransformer, device=self.DEVICE)
            collection = client.get_collection(name=self.NOME_COLECAO, embedding_function=funcao_de_embeddings_sentence_tranformer)
            registros = collection.get()
            registros = zip(registros['ids'], registros['documents'], registros['metadatas'])
            documentos = [ {
                    "id": r[0],
                    "page_content": r[1],
                    "metadata": r[2]
                } for r in registros]

            print(f'Salvando resultados em {url_arquivo_saida}')
            with open(url_arquivo_saida, 'w', encoding='utf-8') as arq:
                arq.write(json.dumps(documentos, indent=4, ensure_ascii=False))

        qtd_docs = len(documentos)
        
        print(f'Gerando perguntas para {qtd_docs} documentos')
        for idx in range(qtd_docs):
            print(f'\rProcessando documento {idx+1} de {qtd_docs}', end='')
            doc = documentos[idx]    
            perguntas = self.gerar_perguntas(artigo=doc['page_content'], contexto=[])
            doc['perguntas'] = perguntas
            
            with open(url_arquivo_saida, 'w', encoding='utf-8') as arq:
                arq.write(json.dumps(documentos, indent=4, ensure_ascii=False))
                
if __name__ == "__main__":
    print('Iniciando gerador de perguntas')
    gerador_banco_perguntas = GeradorPerguntas()
    gerador_banco_perguntas.run()