import gradio as gr
import csv, time

def save_row(prompt, answer, helpfulness, completeness, clarity, notes):
    with open("human_ratings.csv","a", newline="") as f:
        w = csv.writer(f)
        w.writerow([time.time(), prompt, answer, helpfulness, completeness, clarity, notes])
    return "Saved!"

with gr.Blocks() as demo:
    gr.Markdown("# Human Usefulness Ratings (1-5)")
    prompt = gr.Textbox(label="Prompt")
    answer = gr.Textbox(label="Answer")
    helpfulness = gr.Slider(1,5,step=1,value=3,label="Helpfulness")
    completeness = gr.Slider(1,5,step=1,value=3,label="Completeness")
    clarity = gr.Slider(1,5,step=1,value=3,label="Clarity")
    notes = gr.Textbox(label="Notes", lines=2)
    btn = gr.Button("Save Rating")
    out = gr.Label()
    btn.click(save_row, [prompt, answer, helpfulness, completeness, clarity, notes], out)

demo.launch()
