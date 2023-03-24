import re

import openai
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# セッション変数が存在しないときは初期化する
# ここでは 'counter' というセッション変数を作っている
if 'stats_generated' not in st.session_state:
    st.session_state['stats_generated'] = False
if 'history_generated' not in st.session_state:
    st.session_state['history_generated'] = False

def wiki_to_dataframe(wiki_text, player_type="pitcher"):
    if player_type == "pitcher":
        _word = "WHIP"
    elif player_type == "batter":
        _word = "OPS"
    else:
        _word = None

    # 正規表現パターンを定義する
    pattern = r'([^\d])\n'
    pattern_2 = r'(\b19[3-9]\d|20[0-4]\d|2050),(\d)'

    # 正規表現パターンに一致する文字列を置換する
    result = re.sub(pattern, r'\1', wiki_text.replace("\t", ",").replace("\n\n", "\n").replace("丨", "ー").replace(" ", ""))
    # 正規表現パターンに一致する文字列を置換する
    result = re.sub(pattern_2, r'\1,,\2', result)

    result = result.replace(_word, f"{_word}\n")
    
    stats = [t.split(",") for t in result.split("\n")[1:]]
    
    df_stats = pd.DataFrame(stats[1:-2], columns=stats[0])
    df_stats = df_stats.replace(r'^\s*$', np.nan, regex=True)
    df_stats = df_stats.fillna(method="ffill")

    return df_stats

def shaping_dataframe(df_wiki):
    df_wiki.columns = list(map(lambda x: x.replace(" ","").replace("丨", "ー"), list(df_wiki.columns)))
    df_wiki = df_wiki.loc[df_wiki["年度"].apply(lambda x: x.isdigit())]
    return df_wiki

def get_stats_table(url:str, player_type:str):
    tables = pd.read_html(url)
    for table in tables:
        if not "代 表" in list(table.columns):
            if "O P S" in list(table.columns):
                df_batter = shaping_dataframe(table)
            if "W H I P" in list(table.columns):
                df_pitcher = shaping_dataframe(table)
    
    if player_type == "batter":
        return df_batter
    elif player_type == "pitcher":
        return df_pitcher
    else:
        return (df_batter, df_pitcher)

@st.cache_data
def generate_stats(prompt):
    messages=[
        {"role": "assistant", "content": "あなたは野球のデータの専門家です。選手の将来性予測に関する専門家でもあります。入力表と予測表という二つの表について考えます。今から入力表としてある架空の選手の年度別成績表を与えます。次に予測表としてその入力表の続きを決められた年度まで作成して下さい。なお予測表の中の数字はリアリティのある数値を予測して記入しておいて下さい。必ず、最後に入力表の下に予測表を結合するような形で結果を表示して下さい。"},  # 役割設定（省略可）
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        # max_tokens=4096,
        stop=None,
        top_p=0.2,
        temperature=1.0,
    )
    
    text = response["choices"][0]["message"]["content"]
    print(text)
    p = text[text.find("\n\n")+1:]
    stats = p[:p.find("\n\n")]
    stats = [t.split() for t in stats.split("\n")]
    df_stats = pd.DataFrame(stats[1:], columns=stats[0])
    for col in stats[0]:
        try:
            df_stats[col] = pd.to_numeric(df_stats[col], errors="raise")
        except Exception as e:
            continue
    
    st.session_state["stats_generated"] = True
    return df_stats

def generate_history(detail, df_stats):

    prompt = f"""
    {df_stats.to_string(index=False)}

    上記の年度別投手成績成績表の各年度の成績表から選手のストーリーを描いて下さい。
    選手の背景情報は、{detail}
    """

    messages=[
        {"role": "system", "content": "あなたは野球好きの小説家であり素晴らしいストーリーテラーでもあります。選手の成長具合やもどかしさ苦しさを表現するような文章を生成できます。"},  # 役割設定（省略可）
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        # max_tokens=4096,
        stop=None,
        top_p=0.2,
        temperature=0.7,
    )
    st.session_state["history_generated"] = True

    return response["choices"][0]["message"]["content"]


# ボタンが押されたときに発火するコールバック
def generate_click(key):
    # ボタンが押されたらセッション変数の値を増やす
    st.session_state[key] = True

# セッション変数の値をリセットするボタン
def reset_click():
    st.session_state['stats_generated'] = False
    st.session_state["df_stats"] = ""
    st.session_state["history_generated"] = False

# Streamlitアプリケーションのタイトルを設定する
st.title("Player's history maker")

# OpenAI APIキーを設定する
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key

if openai_api_key:
    try:
        models = openai.Model.list()
        player_type = st.selectbox('Select player type.',('pitcher', 'batter'))
        if player_type:
            st.write('You selected:', player_type)

            df_input_table = pd.DataFrame()
            # プロンプトを入力として受け取る
            url = st.text_input("成績を予測したい選手のWikipediaのURLを入力して下さい。", value="https://ja.wikipedia.org/wiki/%E4%BD%90%E3%80%85%E6%9C%A8%E6%9C%97%E5%B8%8C")
            age = st.number_input(label="入団時の年齢を教えて下さい。", min_value=16, max_value=30, value=18)
            if url:
                df_input_table = get_stats_table(url, player_type=player_type)
                df_input_table.insert(1, "年齢", np.arange(age, age+df_input_table.shape[0]))
            st.dataframe(df_input_table)
    
            end_year = st.number_input(label="どの年度まで生成したいかを入力して下さい。", min_value=2023, max_value=2050, value=2030)
            config = st.text_input("選手の成績について発生させたいイベントを入力して下さい。", value=f"2026年度に怪我をしてしまい、その年度は欠場、それ以降成績が下降した。")

            prompt = f"""
            {df_input_table.to_string(index=False)}

            この表の続きを{end_year}年度まで生成して下さい。
            その際に次の5つのルールに必ず従って下さい。
            1.結果の出力は入力表の下に予測表を結合した一つの表のみを表示して下さい。
            2.次の文章を反映した成績表にして下さい。「{config}」
            3.各指標の数値はこれまでのプロ野球の現実的な値を考慮して入力して下さい。例えば、防御率0.20などは非現実的な数値であるためあまり良くありません。
            4.全く同じ数値が並ぶ年度があるのは非現実的なのでやめて下さい。
            5.万が一、欠場する場合の成績は全て0を入れて下さい。

            以下表
            """

            # プロンプトからレスポンスを生成し、逐次的に表示する

            if st.button("Generate Stats"):
                df_stats = generate_stats(prompt)
                st.session_state["df_stats"] = df_stats

            # 表を表示する
            if st.session_state["stats_generated"]:
                df_stats = st.session_state["df_stats"]
                st.dataframe(df_stats)

                y = st.selectbox('Select stats.', list(df_stats.columns)[1:], index=df_stats.shape[1]-2)
                st.line_chart(df_stats, x="年度", y=y)
                # fig, ax = plt.subplots()
                # df_stats.plot(x="年度", y=y, ax=ax, legend=False)
                # ax.set_ylabel(y)
                # ax.grid()
                # st.pyplot(fig)

                detail = st.text_input("選手の経歴や詳細について入力して下さい。", value=f"3人兄弟の末っ子。広島県出身。高校時代に甲子園で優勝しドラフト1位入団。")
                if st.button("Generate Story"):
                    history = generate_history(detail, df_stats)
                if st.session_state["history_generated"]:
                    st.text(history)

                st.button(label='Reset', on_click=reset_click)

    except Exception as e:
        st.error("Error: {}".format(e)) 
