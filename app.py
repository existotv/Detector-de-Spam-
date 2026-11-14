"""
Detector de Spam em E-mails - Aplicação Streamlit + TensorFlow
===============================================================
Uso interno: o funcionário cola a mensagem, a aplicação classifica
como SPAM ou NÃO SPAM e mostra o grau de confiança.

Rodar localmente:   streamlit run app.py
Rodar no Render:    streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Arquitetura (tudo em um único arquivo):
  1. CONFIGURAÇÕES  -> constantes que controlam o modelo
  2. DADOS          -> base de treino embutida (demonstração) ou CSV enviado
  3. MODELO         -> TextVectorization + Embedding + camadas densas
  4. INTERFACE      -> páginas e componentes do Streamlit
"""

# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------
import os

# Reduz o "barulho" de logs do TensorFlow (precisa vir antes do import dele)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------
# 1. CONFIGURAÇÕES
# ---------------------------------------------------------------
TAMANHO_VOCABULARIO = 5000   # quantidade máxima de palavras que o modelo "conhece"
TAMANHO_SEQUENCIA = 60       # nº de palavras analisadas por mensagem (corta/completa)
DIMENSAO_EMBEDDING = 16      # tamanho do vetor que representa cada palavra
EPOCAS = 60                  # quantas vezes o modelo percorre a base de treino
SEMENTE = 42                 # garante resultados reproduzíveis

# ---------------------------------------------------------------
# 2. DADOS
# Base pequena apenas para DEMONSTRAÇÃO (português e inglês).
# Em produção, envie o CSV da empresa pela barra lateral
# (colunas: "texto" e "rotulo", com rotulo = "spam" ou "ham").
# ---------------------------------------------------------------
EXEMPLOS_SPAM = [
    "Parabéns! Você ganhou um prêmio de 10 mil reais. Clique aqui agora para resgatar",
    "Oferta imperdível! Compre agora com 90% de desconto, só hoje, não perca",
    "Seu cartão foi bloqueado. Clique no link e informe sua senha para desbloquear",
    "Ganhe dinheiro rápido trabalhando de casa, sem experiência, renda de 5000 por semana",
    "URGENTE: sua conta será encerrada em 24 horas, confirme seus dados imediatamente",
    "Você foi selecionado para receber um iPhone grátis, responda esta mensagem",
    "Empréstimo aprovado sem consulta ao SPC, dinheiro na conta em minutos, clique já",
    "Emagreça 10 quilos em uma semana com este remédio milagroso, compre agora",
    "Clique aqui para ganhar bitcoin grátis, investimento com retorno garantido de 300%",
    "Última chance! Cupom exclusivo de desconto expira hoje, aproveite essa promoção grátis",
    "Aumente seu desempenho com pílulas naturais, entrega discreta, frete grátis",
    "Você ganhou na loteria internacional, envie seus dados bancários para receber o prêmio",
    "Atualize seu cadastro bancário agora pelo link abaixo ou sua conta será suspensa",
    "Renda extra garantida, invista hoje e multiplique seu dinheiro sem risco",
    "Casino online: ganhe 200 rodadas grátis e bônus de boas-vindas, cadastre-se já",
    "Aproveite! Compre um e leve três, promoção relâmpago, clique para comprar",
    "Congratulations! You have won a free gift card. Click here to claim your prize now",
    "Limited time offer: buy now and get 80% off, act fast, do not miss out",
    "Your account has been suspended. Verify your password immediately by clicking this link",
    "Make money fast from home, no experience needed, guaranteed income every week",
    "Free iPhone giveaway, you are the lucky winner, reply now to claim",
    "Urgent: confirm your bank details to receive your inheritance of 5 million dollars",
    "Lose weight fast with this miracle pill, order today and get free shipping",
    "Cheap meds online, no prescription needed, click here to order now",
    "You have been selected for a cash reward, click the link to receive your money",
    "Exclusive casino bonus, free spins, sign up today and win big",
]

EXEMPLOS_NAO_SPAM = [
    "Oi Carlos, segue em anexo a ata da reunião de ontem para sua revisão",
    "Lembrete: reunião de planejamento amanhã às 14h na sala 3",
    "Bom dia, poderia me enviar o relatório de vendas do mês de outubro quando possível?",
    "Obrigado pelo retorno. Vou analisar a proposta e respondo até sexta-feira",
    "Confirmo minha presença no treinamento de quinta-feira pela manhã",
    "Segue o contrato revisado pelo jurídico, qualquer dúvida estou à disposição",
    "Pessoal, o almoço de confraternização da equipe será na próxima terça-feira",
    "Preciso do seu apoio para fechar o orçamento da campanha até o final da semana",
    "O cliente aprovou o layout do anúncio, podemos seguir para a produção",
    "Estou com uma dúvida sobre a fatura do mês passado, podemos conversar amanhã?",
    "Bom dia equipe, o relatório de desempenho da campanha de e-mail está disponível na pasta",
    "Vou ficar fora do escritório na segunda-feira, retorno na terça com as pendências",
    "Pode me passar o contato da agência responsável pelo evento do próximo mês?",
    "Parabéns pelo excelente trabalho na apresentação de hoje, o cliente gostou muito",
    "Agendei a entrevista com o candidato para quarta-feira às 10h",
    "Anexei a planilha com as métricas de abertura e cliques da última newsletter",
    "Hi John, please find attached the minutes from yesterday's meeting",
    "Reminder: the project status call is tomorrow at 3 pm in the main conference room",
    "Thanks for your reply. I will review the proposal and get back to you by Friday",
    "Could you send me the sales report for last month when you have a chance?",
    "The client approved the final design, we can move on to production next week",
    "Team lunch is scheduled for Tuesday, please let me know about dietary restrictions",
    "I am out of office on Monday and will return on Tuesday to catch up on pending items",
    "Attached is the revised contract from legal, let me know if you have any questions",
    "Great job on today's presentation, the client was very pleased with the results",
    "Please confirm your attendance at the training session on Thursday morning",
]


def montar_base_demonstracao() -> pd.DataFrame:
    """Junta as listas de exemplos em um DataFrame com 'texto' e 'rotulo'."""
    spam = pd.DataFrame({"texto": EXEMPLOS_SPAM, "rotulo": "spam"})
    ham = pd.DataFrame({"texto": EXEMPLOS_NAO_SPAM, "rotulo": "ham"})
    return pd.concat([spam, ham], ignore_index=True)


def ler_csv_enviado(arquivo_bytes: bytes) -> pd.DataFrame:
    """Lê o CSV enviado e valida as colunas obrigatórias."""
    df = pd.read_csv(io.BytesIO(arquivo_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"texto", "rotulo"}.issubset(df.columns):
        raise ValueError("O CSV precisa ter as colunas 'texto' e 'rotulo'.")
    df = df[["texto", "rotulo"]].dropna()
    df["rotulo"] = df["rotulo"].astype(str).str.strip().str.lower()
    df = df[df["rotulo"].isin(["spam", "ham"])]
    if df["rotulo"].nunique() < 2 or len(df) < 20:
        raise ValueError(
            "O CSV precisa ter ao menos 20 linhas e conter exemplos de 'spam' e de 'ham'."
        )
    return df


# ---------------------------------------------------------------
# 3. MODELO
# @st.cache_resource: o treino roda UMA vez e o resultado fica em memória,
# evitando retreinar a cada clique do usuário (o Streamlit reexecuta o
# script inteiro a cada interação).
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def treinar_modelo(textos: tuple, rotulos: tuple):
    """Treina o classificador e devolve (vetorizador, modelo, acurácia_validação)."""
    keras.utils.set_random_seed(SEMENTE)

    textos = np.array(textos)
    y = np.array([1 if r == "spam" else 0 for r in rotulos], dtype="float32")

    # Separa 80% para treino e 20% para validação (medir se o modelo generaliza)
    rng = np.random.default_rng(SEMENTE)
    indices = rng.permutation(len(textos))
    corte = int(len(textos) * 0.8)
    idx_treino, idx_val = indices[:corte], indices[corte:]

    # TextVectorization: transforma texto em números (um id por palavra).
    # Aprende o vocabulário SOMENTE com os dados de treino.
    vetorizador = layers.TextVectorization(
        max_tokens=TAMANHO_VOCABULARIO,
        output_mode="int",
        output_sequence_length=TAMANHO_SEQUENCIA,
        standardize="lower_and_strip_punctuation",
    )
    vetorizador.adapt(textos[idx_treino])

    x_treino = vetorizador(tf.constant(textos[idx_treino])).numpy()
    x_val = vetorizador(tf.constant(textos[idx_val])).numpy()

    # Rede neural simples:
    #  - Embedding: cada palavra vira um vetor que o modelo aprende
    #  - GlobalAveragePooling1D: faz a média dos vetores (resume a mensagem)
    #  - Dense + Dropout: camadas de decisão (Dropout reduz overfitting)
    #  - Saída sigmoid: devolve um número entre 0 (não spam) e 1 (spam)
    modelo = keras.Sequential(
        [
            keras.Input(shape=(TAMANHO_SEQUENCIA,), dtype="int64"),
            layers.Embedding(TAMANHO_VOCABULARIO, DIMENSAO_EMBEDDING),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    modelo.fit(
        x_treino, y[idx_treino],
        epochs=EPOCAS, batch_size=8, verbose=0,
    )

    _, acuracia_val = modelo.evaluate(x_val, y[idx_val], verbose=0)
    return vetorizador, modelo, float(acuracia_val)


def classificar(mensagem: str, vetorizador, modelo) -> float:
    """Devolve a probabilidade (0 a 1) de a mensagem ser spam."""
    x = vetorizador(tf.constant([mensagem])).numpy()
    return float(modelo.predict(x, verbose=0)[0][0])


# ---------------------------------------------------------------
# 4. INTERFACE (Streamlit)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Detector de Spam | Marketing",
    page_icon="🛡️",
    layout="centered",
)

# Histórico da sessão (session_state guarda dados entre as reexecuções)
if "historico" not in st.session_state:
    st.session_state.historico = []

# ---- Barra lateral: configurações ----
with st.sidebar:
    st.header("⚙️ Configurações")

    limiar = st.slider(
        "Sensibilidade (limiar de spam)",
        min_value=0.30, max_value=0.90, value=0.50, step=0.05,
        help="Acima deste valor a mensagem é classificada como SPAM. "
             "Valores menores bloqueiam mais; valores maiores são mais permissivos.",
    )

    st.divider()
    st.subheader("Base de treinamento")
    arquivo = st.file_uploader(
        "Enviar CSV da empresa (opcional)",
        type=["csv"],
        help="Colunas obrigatórias: 'texto' e 'rotulo' (spam ou ham).",
    )

# ---- Carrega os dados e treina o modelo ----
try:
    if arquivo is not None:
        base = ler_csv_enviado(arquivo.getvalue())
        origem = f"CSV enviado ({len(base)} mensagens)"
    else:
        base = montar_base_demonstracao()
        origem = f"Base de demonstração ({len(base)} mensagens)"
except Exception as erro:
    st.sidebar.error(f"Erro no CSV: {erro}")
    base = montar_base_demonstracao()
    origem = f"Base de demonstração ({len(base)} mensagens)"

with st.spinner("Preparando o modelo de IA..."):
    vetorizador, modelo, acuracia = treinar_modelo(
        tuple(base["texto"]), tuple(base["rotulo"])
    )

with st.sidebar:
    st.caption(f"📚 {origem}")
    st.caption(f"🎯 Acurácia na validação: {acuracia:.0%}")
    if arquivo is None:
        st.warning(
            "Modelo de demonstração. Para uso real, envie uma base maior "
            "com mensagens da própria empresa."
        )

# ---- Área principal ----
st.title("🛡️ Detector de Spam")
st.markdown(
    "Cole abaixo o conteúdo do e-mail. A ferramenta analisa o texto e indica "
    "se a mensagem é **spam** ou **não spam**."
)

mensagem = st.text_area(
    "Mensagem do e-mail",
    height=200,
    placeholder="Cole aqui o texto do e-mail que deseja analisar...",
)

col1, col2 = st.columns([1, 1])
analisar = col1.button("🔍 Analisar mensagem", type="primary", use_container_width=True)

if analisar:
    if not mensagem.strip():
        st.warning("Digite ou cole uma mensagem para analisar.")
    else:
        prob_spam = classificar(mensagem, vetorizador, modelo)
        eh_spam = prob_spam >= limiar
        confianca = prob_spam if eh_spam else 1 - prob_spam

        st.divider()
        if eh_spam:
            st.error("🚨 **SPAM** — esta mensagem tem características de spam.")
        else:
            st.success("✅ **NÃO É SPAM** — esta mensagem parece legítima.")

        m1, m2 = st.columns(2)
        m1.metric("Probabilidade de spam", f"{prob_spam:.1%}")
        m2.metric("Confiança da classificação", f"{confianca:.1%}")
        st.progress(min(max(prob_spam, 0.0), 1.0), text="Nível de risco de spam")

        # Guarda no histórico da sessão
        st.session_state.historico.insert(0, {
            "Horário": datetime.now().strftime("%H:%M:%S"),
            "Mensagem": mensagem.strip()[:80] + ("..." if len(mensagem.strip()) > 80 else ""),
            "Resultado": "SPAM" if eh_spam else "NÃO SPAM",
            "Prob. spam": f"{prob_spam:.1%}",
        })

# ---- Histórico ----
if st.session_state.historico:
    st.divider()
    st.subheader("📋 Histórico desta sessão")
    st.dataframe(
        pd.DataFrame(st.session_state.historico),
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Limpar histórico"):
        st.session_state.historico = []
        st.rerun()

st.divider()
st.caption(
    "Ferramenta de apoio à decisão. Resultados de IA podem conter erros; "
    "em caso de dúvida, valide com a equipe de segurança."
)
