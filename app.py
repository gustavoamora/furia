import os
import openai
import streamlit as st
from qdrant_client import QdrantClient
from datetime import datetime
from dotenv import load_dotenv

# ======================================
#   Carregar variáveis de ambiente
# ======================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

openai.api_key = OPENAI_API_KEY

# ======================================
#   Inicializar QdrantClient
# ======================================
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "furia_matches_24"

# ======================================
#   Config. Modelos
# ======================================
llm = "gpt-4o-mini"
temperature = 0.5
embeddings = "text-embedding-3-small"

# ======================================
#   Funções auxiliares
# ======================================
def get_embedding(text: str, model=embeddings) -> list[float]:
    """Gera embedding de um texto usando endpoint OpenAI Embeddings."""
    response = openai.Embedding.create(input=text, model=model)
    return response["data"][0]["embedding"]


def retrieve_context(query: str, top_k=10) -> list[str]:
    """Faz similarity search no Qdrant e retorna lista de conteúdos relevantes."""
    embedding = get_embedding(query)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k
    )
    contexts = []
    for i, point in enumerate(search_result, start=1):
        content = point.payload.get("content", "")
        contexts.append(content)
        print(f"[DEBUG] Doc {i} (score={point.score}): {content[:60]}...")
    return contexts


def generate_answer(system_prompt: str, messages_history: list[dict], temperature=temperature, model=llm):
    """
    Recebe:
      - system_prompt: string base que define o "contexto" e regras do sistema
      - messages_history: histórico do chat (lista de dicionários {"role": "...", "content": "..."})
    Retorna:
      - texto gerado pelo ChatCompletion
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages_history   # expande a lista de mensagens (user e assistant)
        ],
        temperature=temperature
    )
    return response["choices"][0]["message"]["content"]


def build_system_prompt(context_joined: str) -> str:
    """
    Cria o prompt do 'system' com estilo e instruções da FURIA + contexto de busca.
    """
    today_date = datetime.now().strftime("%d de %B de %Y")  # Data atual

    return (
        f"Você é um assistente **oficial** da FURIA, especializado em eSports, "
        f"conversando com torcedores pelo WhatsApp. "
        f"Hoje é {today_date}. "
        "Seu tom de voz deve ser **amigável, engajado e empolgado**, representando "
        "o espírito competitivo e a energia da FURIA.\n\n"
        "Suas respostas podem incluir gírias de eSports, expressões de motivação, "
        "e até emojis relacionados ao time. Você pode mencionar a pantera (símbolo da FURIA) e "
        "usar as cores preto e branco para destacar alguma informação se achar relevante.\n\n"
        "### Regras e estilo:\n"
        "1. Seja **focado** na FURIA e no assunto eSports.\n"
        "2. Use **detalhes** do CONTEXTO quando disponíveis para responder. Se não houver "
        "informação suficiente, seja transparente e sugira ao usuário buscar mais detalhes "
        "em fontes oficiais ou aguardar atualizações.\n"
        "3. Considere a data atual ({today_date}) para responder perguntas temporais como 'último jogo'.\n"
        "4. Mantenha um tom **respeitoso** e **positivamente competitivo**.\n"
        "5. Pode usar até **2 emojis** na resposta, especialmente temáticos (pantera, troféu, etc.).\n\n"
        "6. Você só tem informações sobre os jogos do time principal de CS2 masculino da FURIA e somente do ano de 2024"

        "A seguir está o contexto que você deve usar para responder:\n\n"
        f"=== CONTEXTO ===\n{context_joined}\n=== FIM DO CONTEXTO ===\n\n"
        "Responda de forma **clara** e **objetiva**, mas sem perder a empolgação do mundo gamer!\n"
    )

# ======================================
#   Construir a interface Streamlit
# ======================================
def main():
    st.title("FURIA Matches Chatbot")

    # Inicializa a memória de mensagens na sessão, se ainda não existir
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # lista de dicts: [{"role": "user"/"assistant", "content": "..."}]

    # Campo de texto para o usuário
    user_input = st.text_input("Olá! Pergunte algo sobre os jogos do time principal de CS2 masculino da FURIA em 2024:", "")

    if st.button("Enviar"):
        if user_input.strip():
            # 1) Adiciona a mensagem do usuário no histórico
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # 2) Recupera contexto do Qdrant para essa nova pergunta
            top_contexts = retrieve_context(user_input, top_k=10)
            context_joined = "\n\n".join(top_contexts)

            # 3) Monta o system_prompt com esse contexto
            system_prompt = build_system_prompt(context_joined)

            # 4) Gera a resposta levando em conta TODAS as mensagens do histórico
            answer = generate_answer(system_prompt, st.session_state["messages"], model=llm)
            
            # 5) Adiciona resposta do assistente ao histórico
            st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Exibir todo o histórico de mensagens
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"**Você:** {msg['content']}")
        else:
            st.markdown(f"**FURIA Bot:** {msg['content']}")

if __name__ == "__main__":
    main()
