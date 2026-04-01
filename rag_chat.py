if authentication_status:


    import streamlit as st
    import streamlit_authenticator as stauth

    # -------------------------
    # USUÁRIOS
    # -------------------------
    names = ["Claudio", "Equipe"]
    usernames = ["claudio", "equipe"]
    passwords = ["123456", "senha123"]

    hashed_passwords = stauth.Hasher(passwords).generate()

    authenticator = stauth.Authenticate(
        names,
        usernames,
        hashed_passwords,
        "chat_cookie",
        "abc123",
        cookie_expiry_days=1
    )

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status == False:
        st.error("Usuário ou senha incorretos")
        st.stop()

    if authentication_status == None:
        st.warning("Digite usuário e senha")
        st.stop()

    if authentication_status:
        authenticator.logout("Logout", "sidebar")
        st.sidebar.write(f"Bem-vindo, {name}")
    # Fim da autenticação

    #Inicio do codigo
    import streamlit as st
    import requests
    import os
    import re
    from dotenv import load_dotenv

    load_dotenv()

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from sentence_transformers import CrossEncoder, SentenceTransformer, util

    # =========================
    # CONFIG
    # =========================
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    MODEL = "deepseek/deepseek-chat"

    CHROMA_DIR = "chroma_db"
    MEMORY_DIR = "memory_db"

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    st.set_page_config(layout="wide")

    # =========================
    # CSS
    # =========================
    st.markdown("""
    <style>
    div[data-baseweb="input"] input {
        border: 2px solid black !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # BASES
    # =========================
    @st.cache_resource
    def carregar_base():
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    @st.cache_resource
    def carregar_memoria():
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        return Chroma(persist_directory=MEMORY_DIR, embedding_function=embeddings)

    vectorstore = carregar_base()
    memory_store = carregar_memoria()

    # =========================
    # STATE MULTI-CHAT
    # =========================
    if "chats" not in st.session_state:
        st.session_state["chats"] = {}

    if "chat_atual" not in st.session_state:
        st.session_state["chat_atual"] = "Chat 1"

    if st.session_state["chat_atual"] not in st.session_state["chats"]:
        st.session_state["chats"][st.session_state["chat_atual"]] = {
            "mensagens": [],
            "memoria_resumo": ""
        }

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.title("💬 Conversas")

    if st.sidebar.button("➕ Novo Chat"):
        novo_nome = f"Chat {len(st.session_state['chats']) + 1}"
        st.session_state["chat_atual"] = novo_nome
        st.session_state["chats"][novo_nome] = {
            "mensagens": [],
            "memoria_resumo": ""
        }

    for nome in list(st.session_state["chats"].keys()):
        if st.sidebar.button(nome):
            st.session_state["chat_atual"] = nome

    st.sidebar.markdown("---")
    st.sidebar.write(f"Chat atual: **{st.session_state['chat_atual']}**")

    # =========================
    # FUNÇÃO NOME DO CHAT
    # =========================
    def gerar_nome_chat(pergunta):
        pergunta = pergunta.strip().replace("\n", " ")
        return pergunta[:40] + ("..." if len(pergunta) > 40 else "")

    # =========================
    # MEMÓRIA
    # =========================
    def pergunta_dependente(pergunta):
        gatilhos = ["explique", "melhor", "isso", "aquilo", "detalhe", "como assim"]
        return any(g in pergunta.lower() for g in gatilhos)

    def salvar_memoria(pergunta, resposta):
        texto = f"Pergunta: {pergunta}\nResposta: {resposta}"
        memory_store.add_texts([texto])

    def atualizar_resumo(pergunta, resposta):
        chat_data = st.session_state["chats"][st.session_state["chat_atual"]]
        memoria_atual = chat_data["memoria_resumo"]

        prompt = f"""
    Resuma a conversa mantendo apenas o essencial.

    MEMÓRIA:
    {memoria_atual}

    NOVO:
    Pergunta: {pergunta}
    Resposta: {resposta}

    Resumo curto:
    """

        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "Resuma contexto."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )

            resumo = r.json()["choices"][0]["message"]["content"]
            chat_data["memoria_resumo"] = resumo[:800]

        except:
            chat_data["memoria_resumo"] = (memoria_atual + " " + pergunta)[:800]

    def buscar_memoria(pergunta):
        docs = memory_store.similarity_search(pergunta, k=3)
        return "\n".join([d.page_content for d in docs])

    # =========================
    # RAG
    # =========================
    def validar_contexto(pergunta, docs_scores,
                        threshold_vector=0.6,
                        threshold_reranker=0.3):

        if not docs_scores:
            return False

        if docs_scores[0][1] > threshold_vector:
            return False

        docs = [doc for doc, _ in docs_scores[:3]]
        pares = [(pergunta, d.page_content) for d in docs]
        scores = reranker.predict(pares)

        return max(scores) >= threshold_reranker

    def melhor_trecho(pergunta, texto):
        partes = texto.split(".")
        if len(partes) < 2:
            return texto[:200]

        emb_q = embedder.encode(pergunta, convert_to_tensor=True)
        emb_s = embedder.encode(partes, convert_to_tensor=True)

        scores = util.cos_sim(emb_q, emb_s)[0]
        return partes[scores.argmax().item()]

    # =========================
    # PROMPT
    # =========================
    def montar_prompt(contexto_rag, memoria, pergunta):
        chat_data = st.session_state["chats"][st.session_state["chat_atual"]]

        return f"""
    Responda em texto corrido.

    Prioridade:
    1. Memória da conversa
    2. Contexto dos documentos

    Se não souber, diga NÃO ENCONTRADO.

    MEMÓRIA RESUMIDA:
    {chat_data["memoria_resumo"]}

    MEMÓRIA RECUPERADA:
    {memoria}

    CONTEXTO:
    {contexto_rag}

    PERGUNTA:
    {pergunta}
    """

    # =========================
    # LLM
    # =========================
    def gerar_resposta(prompt):
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Seja claro."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
        )

        return r.json()["choices"][0]["message"]["content"]

    # =========================
    # ENVIO
    # =========================
    def enviar():
        pergunta = st.session_state["input"]

        if not pergunta.strip():
            return

        nome_chat_atual = st.session_state["chat_atual"]
        chat_data = st.session_state["chats"][nome_chat_atual]

        # 🔥 RENOMEAR CHAT NA PRIMEIRA PERGUNTA
        if len(chat_data["mensagens"]) == 0:
            novo_nome = gerar_nome_chat(pergunta)

            contador = 1
            base_nome = novo_nome

            while novo_nome in st.session_state["chats"]:
                contador += 1
                novo_nome = f"{base_nome} ({contador})"

            st.session_state["chats"][novo_nome] = st.session_state["chats"].pop(nome_chat_atual)
            st.session_state["chat_atual"] = novo_nome

            chat_data = st.session_state["chats"][novo_nome]

        usar_memoria = pergunta_dependente(pergunta)

        docs_scores = vectorstore.similarity_search_with_score(pergunta, k=4)

        contexto_rag = ""
        fontes = []

        if docs_scores and not usar_memoria and validar_contexto(pergunta, docs_scores):
            docs = [doc for doc, _ in docs_scores]

            for doc in docs:
                conteudo = doc.page_content
                contexto_rag += conteudo + "\n"

                fontes.append({
                    "arquivo": os.path.basename(doc.metadata.get("source", "")),
                    "pagina": doc.metadata.get("page", 0),
                    "trecho": melhor_trecho(pergunta, conteudo)
                })

        memoria = buscar_memoria(pergunta)

        prompt = montar_prompt(contexto_rag, memoria, pergunta)

        resposta = gerar_resposta(prompt)

        salvar_memoria(pergunta, resposta)
        atualizar_resumo(pergunta, resposta)

        chat_data["mensagens"].append({
            "pergunta": pergunta,
            "resposta": resposta,
            "fontes": fontes
        })

        st.session_state["input"] = ""

    # =========================
    # UI
    # =========================
    st.title("🤖 Chat RAG com Memória Avançada")

    chat_data = st.session_state["chats"][st.session_state["chat_atual"]]

    for item in chat_data["mensagens"]:
        st.markdown(f"### 👤 {item['pergunta']}")
        st.markdown(item["resposta"])

        if item["fontes"]:
            with st.expander("📚 Fontes"):
                for f in item["fontes"]:
                    st.markdown(f"""
    **{f['arquivo']} (pág {f['pagina']+1})**

    > {f['trecho']}
    """)

        st.divider()

    # =========================
    # INPUT
    # =========================
    st.text_input(
        "Digite sua pergunta...",
        key="input",
        on_change=enviar
    )