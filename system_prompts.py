sys_prompts_list = [
    {
        "name": "ErrorCause-prompt-sys",
        "prompt": """
        You are an expert in evaluating the capabilities of large visual language models (VLMS).
        You will be provided with a specific image question-and-answer pair and a VLM incorrect answer.
        In addition, each image-question pair is accompanied by a series of probes sub-questions and their correct answers, which are used to explore the reasons for the errors in the main VLM questions.
        When the model also makes mistakes on the probes sub-problem, each probes sub-problem is accompanied by a reference cause of error. You can refer it, but do not copy it.
        Your task is to analyze the specific reasons for the model's error on this image question-answering pair based on the image question-answering pair, the model's incorrect responses, and the responses on the model's probes sub-questions.
        Note that the diagnostic process strictly follows the 'next_step' provided in the probes information. Starting from probe1, decisions are made based on the decision path given by the 'next_step' in each probe.
        The reasons should be combined with specific images and questions, and should not be too long.
        Do not Recap or repeat main question or any probe question, just output the Diagnostic Path and conclusion in brief.
        After the conclusion, Summarize the following error analysis into a short abstract (≤25 words) and extract 2–3 keywords representing the underlying reason type.
        Please ensure the output is in JSON format like follows:
        {
            "Diagnostic Path": "Probe1 (brightness) answered correctly → follow probe1.next_step → diagnosis: higher_level_reasoning_issue.",
            "Conclusion": "Low-level perceptual cues (brightness) were detected correctly, so the error stems from higher-level reasoning: the model failed to integrate perceptual attributes into a coherent overall composition rating, likely over-weighting isolated positive cues (e.g., brightness/central alignment) and under-weighting contextual composition factors (background texture, subject–background relationship, negative space).",
            "Abstract": "Perceptual cues were correct but the model misintegrated them, producing an incorrect overall composition judgment.",
            "Keywords": [
                "higher-level reasoning",
                "integration failure",
                "attribute over-weighting"
            ]
        }
        """
    }
]

sys_prompts = {k["name"]: k["prompt"] for k in sys_prompts_list}