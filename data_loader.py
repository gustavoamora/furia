import os
import openai

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

# ===========================
#   Clients 
# ===========================

# Obter as chaves do ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Inicializa os clientes
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Defina a chave da API OpenAI
openai.api_key = OPENAI_API_KEY

print("Clientes inicializados com sucesso!")

# ===========================
#   Collection Qdrant 
# ===========================
collection_name = "furia_matches_24"

# Se já estiver criada, comentar este bloco a partir daqui:
collection_config = VectorParams(
    size=1536,
    distance=Distance.COSINE
)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=collection_config
)

# ===========================
#   Embeddings
# ===========================
file_path = "furia_matches_24.txt"

# Inicializar variáveis
points = []
current_content = ""

# Processar o arquivo linha por linha
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        # Identificar início de um novo bloco com infos da partida
        if line.startswith("#"):
            if current_content:  # Consolidar o bloco anterior
                # Gerar embedding
                response = openai.Embedding.create(
                    input=current_content.strip(),
                    model="text-embedding-3-small"
                )
                embedding = response["data"][0]["embedding"]

                # Adicionar o ponto consolidado
                points.append(
                    PointStruct(
                        id=len(points) + 1,
                        vector=embedding,
                        payload={"content": current_content.strip()}
                    )
                )
                current_content = ""  # Reiniciar o conteúdo

        # Adicionar linha ao conteúdo atual
        current_content += line

    # Processar o último jogo no final do arquivo
    if current_content:
        response = openai.Embedding.create(
            input=current_content.strip(),
            model="text-embedding-3-small"
        )
        embedding = response["data"][0]["embedding"]

        points.append(
            PointStruct(
                id=len(points) + 1,
                vector=embedding,
                payload={"content": current_content.strip()}
            )
        )

# Inserir no Qdrant
qdrant_client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"{len(points)} jogos processados e inseridos no Qdrant com sucesso!")
