import gradio as gr
from src.inference import SentimentAnalyzer

def main():
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()

    # Create the Gradio interface
    interface = gr.Interface(
        fn=analyzer.predict,
        inputs=gr.Textbox(label="Enter financial text"),
        outputs=gr.Label(num_top_classes=3),
        title="Financial Sentiment Analysis",
        description="Analyze the sentiment of financial texts"
    )

    interface.launch()
    
if __name__ == "__main__":
    main()
