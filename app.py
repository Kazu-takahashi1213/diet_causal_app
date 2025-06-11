import streamlit as st
import pandas as pd
import os
from datetime import date
from causalml.inference.meta import LRSRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

DATA_PATH = "data/diary.csv"
os.makedirs("data", exist_ok=True)

# ------------------ UI ------------------

st.title("ダイエット因果推論アプリ")
st.markdown("自分の行動（運動・睡眠・食事制限）が体重減少にどれだけ効いているか、因果的に分析します。")

st.header("今日のデータ入力")
with st.form("diary_form"):
    entry_date = st.date_input("日付", value=date.today())
    exercise_time = st.number_input("運動時間（分）", min_value=0, step=10)
    sleep_hours = st.number_input("睡眠時間（時間）", min_value=0.0, step=0.5)
    calorie_intake = st.number_input("摂取カロリー（kcal）", min_value=0, step=50)
    weight = st.number_input("今朝の体重（kg）", min_value=0.0, step=0.1)
    gender = st.selectbox("性別", ["男性", "女性"])
    age = st.number_input("年齢", min_value=10, max_value=100, step=1)
    submitted = st.form_submit_button("保存する")

if submitted:
    new_data = pd.DataFrame([{
        "date": entry_date,
        "exercise_min": exercise_time,
        "sleep_hr": sleep_hours,
        "calorie_kcal": calorie_intake,
        "weight_kg": weight,
        "gender": 1 if gender == "男性" else 0,
        "age": age
    }])

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_csv(DATA_PATH, index=False)
    st.success("保存完了")

# ------------------ 分析 ------------------

st.header("因果推論による効果分析")

if not os.path.exists(DATA_PATH):
    st.warning("データがまだありません。まずデータを入力してください。")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.write("過去のデータ", df.tail(10))

# 分析対象の行動を選択
st.subheader("どの行動の効果を分析したい？")
treatment_col = st.selectbox("介入（Treatment）を選んでください", [
    "exercise_min", "sleep_hr", "calorie_kcal"
])

# アウトカム（体重減少量）
df = df.sort_values("date")
df["weight_diff"] = df["weight_kg"].shift(1) - df["weight_kg"]
df = df.dropna()

# 介入のバイナリ化（行動の強弱によって変える）
median = df[treatment_col].median()
df["T"] = (df[treatment_col] > median).astype(int)

X = df[["gender", "age", "exercise_min", "sleep_hr", "calorie_kcal"]]
y = df["weight_diff"]

# 傾向スコア推定と回帰による因果推論
learner = LRSRegressor()
learner.fit(X, df["T"], y)
te, lb, ub = learner.estimate_ate(X, df["T"], y)

# ------------------ 結果 ------------------

st.subheader("因果効果の推定結果")
st.write(f"**{treatment_col} を増やすと、平均して体重が {round(te[0], 3)} kg 減る効果**が推定されました。")
st.write(f"信頼区間: [{round(lb[0], 3)}, {round(ub[0], 3)}]")

# グラフ表示
fig, ax = plt.subplots()
df.groupby("T")["weight_diff"].mean().plot(kind="bar", ax=ax)
ax.set_xticklabels(["低い", "高い"], rotation=0)
ax.set_title(f"{treatment_col} の多寡による体重減少の平均比較")
ax.set_ylabel("前日からの体重減少（kg）")
st.pyplot(fig)
