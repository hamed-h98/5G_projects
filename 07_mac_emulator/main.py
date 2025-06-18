from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
print("Flask app is starting...")

@app.route("/")
def index():
    df = pd.read_csv("queue_state.csv")
    users = df['user_id'].unique()

    fig, ax = plt.subplots(figsize=(10, 5))
    for uid in users:
        user_df = df[df['user_id'] == uid]
        ax.plot(user_df['slot'], user_df['queue_len'], label=f"User {uid}")

    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Queue Length")
    ax.set_title("MAC Queue Length vs Time")
    ax.legend()
    ax.grid(True)

    # Convert plot to PNG for embedding
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return render_template("plot.html", plot_data=img_base64)

if __name__ == "__main__":
    print("Flask app is starting...")
    import os
    print("Running on:", os.getcwd())
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)


