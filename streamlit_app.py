import re
import os

import openai
import streamlit as st
import numpy as np
import pandas as pd
from langchain.llms import OpenAIChat
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

def wiki_to_dataframe(wiki_text):
    # 正規表現パターンを定義する
    pattern = r'([^\d])\n'
    pattern_2 = r'(\b19[3-9]\d|20[0-4]\d|2050),(\d)'

    # 正規表現パターンに一致する文字列を置換する
    result = re.sub(pattern, r'\1', wiki_text.replace("\t", ",").replace("\n\n", "\n").replace("丨", "ー").replace(" ", ""))
    # 正規表現パターンに一致する文字列を置換する
    result = re.sub(pattern_2, r'\1,,\2', result)
    result = result.replace("WHIP", "WHIP\n")
    stats = result.split("\n")
    ind = np.argmax(["通算" in t for t in stats])

    stats = [t.split(",") for t in stats[:ind+1]]

    df_stats = pd.DataFrame(stats[1:-1], columns=stats[0])
    df_stats = df_stats.replace(r'^\s*$', np.nan, regex=True)
    df_stats = df_stats.fillna(method="ffill")

    return df_stats


# Streamlitアプリケーションのタイトルを設定する
st.title("Player history maker : Pitcher ver.")

# OpenAI APIキーを設定する
openai_api_key = st.text_input("Enter your OpenAI API key", value="", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key


# プレフィックスメッセージの準備
prefix_messages = [
    {
        "role": "system", 
        "content": """
あなたは野球のデータの専門家です。選手の将来性予測に関する専門家でもあります。入力表と予測表という二つの表について考えます。今から入力表としてある架空の選手の年度別成績表を与えます。次に予測表としてその入力表の続きを決められた年度まで作成して下さい。なお予測表の中の数字はリアリティのある数値を予測して記入しておいて下さい。必ず、最後に入力表の下に予測表を結合するような形で結果を表示して下さい。"
"""
}
]

# LLMの準備
llm = OpenAIChat(
    temperature=0, 
    prefix_messages=prefix_messages,
    streaming=False
)

conversation = ConversationChain(
    llm=llm, 
    verbose=False,
    memory=ConversationBufferMemory()
)
if openai_api_key:

    df_input_table = pd.DataFrame()
    # プロンプトを入力として受け取る
    input_table = st.text_area("Enter base stats copy&paste from wikipedia", height=100)
    if input_table:
        df_input_table = wiki_to_dataframe(input_table)
        st.dataframe(df_input_table)

    config = st.text_input("Enter your prompt here", value="この表の続きを2036年度まで生成して下さい。また、2026年度に怪我をしてしまい、その年度は欠場、それ以降成績が下降して下さい。欠場時の成績は全て0を入れて下さい。出力は入力表の下に予測表を結合して一つの表として結果のみを表示して下さい。")


    prompt = f"""
    {df_input_table.to_string(index=False)}

    {config}
    必ず入力表の下に予測表を結合するような形で結果を表示して下さい。
    """


    # プロンプトからレスポンスを生成し、逐次的に表示する
    if st.button("Generate"):
        text = conversation.predict(input=prompt)
        # st.write(text)
        print(text)
        p = text[text.find("\n\n")+2:]
        stats = p[:p.find("\n\n")]
        stats = [t.split() for t in stats.split("\n")]
        df_stats = pd.DataFrame(stats[1:], columns=stats[0])

        # 表を表示する
        st.dataframe(df_stats)