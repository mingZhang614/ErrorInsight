import os
import json
from PIL import Image
import base64
from tqdm import tqdm
from base_agent import BaseAgent
from system_prompts import sys_prompts_list
from google import genai
from google.genai import types
from io import BytesIO


# ----------------------------------------------------------------
# 1. Image and prompt pre-process
# ----------------------------------------------------------------

def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


def build_question_prompt(question):
    prompt = (
        "Only answer with one letter from A, B, C, or D. "
        "Do not explain."
        "You will see: the question to be answered, and the image.\n\n"
        "Rules:\n"
        "- Answer the question according to the image.\n"
        "- Only reply with one letter: A, B, C, or D.\n"
        "- Do not add any other text.\n\n"
        f"Question:\n{question}\n"
    )
    return prompt


# ----------------------------------------------------------------
# 2. Evaluated MLLM
# ----------------------------------------------------------------

class EvaluatedMLLM:
    """MLLM to be evaluated (ex. Gemini-2.5-Pro)"""

    def __init__(self, model_name="gemini-2.5-pro", api_key=None):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def inference(self, image_path, question):
        """
        Gemini-2.5-Pro inference
        """
        try:
            # 1. Prompt
            prompt_text = build_question_prompt(question)

            # 2. Image encode
            b64_data = encode_image_to_base64(image_path)
            image_part = types.Part.from_bytes(
                data=base64.b64decode(b64_data),
                mime_type="image/png"
            )

            # 3. MLLM for VQA inference
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt_text, image_part],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=5,
                    candidate_count=1
                )
            )

            # 4. Analysis output: Retain only the first letter or remove the extra Spaces/line breaks at the beginning and end
            ans = response.text.strip()
            if len(ans) > 1 and ans[0].isalpha():
                ans = ans[0]

            return ans

        except Exception as e:
            print(f"[Error] Gemini inference failed for {image_path}: {e}")
            return "ERROR"


# 3. ErrorInsight
# ----------------------------------------------------------------

class ErrorInsightFramework:
    def __init__(self, dataset_path, image_dir, diagnostic_agent):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        self.image_dir = image_dir
        self.agent = diagnostic_agent
        self.final_reports = []

    def run(self, target_model):
        print(f"[*] Starting diagnosis on ChallengeBench using {target_model.model_name}...")

        for entry in tqdm(self.dataset):
            img_path = os.path.join(self.image_dir, entry['image_file'])
            main_q = entry['main_question']

            # --- Step 1: Verify the main problem ---
            model_ans = target_model.inference(img_path, main_q['question'])
            is_correct = (model_ans.strip().upper() == main_q['answer'].upper())

            # Determine whether to complete or start the probe based on next_step
            next_step = main_q['next_step']['if_correct'] if is_correct else main_q['next_step']['if_wrong']

            if next_step == "Finish":
                continue

            # --- Step 2: Probing Trajectory ---
            trajectory = []
            current_pointer = next_step
            reference_label = ""

            # Perform probing in the order of the decision tree path
            while True:
                # Check whether "diagnosis" or "Finish" has been reached.
                if current_pointer == "Finish":
                    break
                if "diagnosis:" in current_pointer:
                    reference_label = current_pointer.split("diagnosis:")[1].strip()
                    break

                # Obtain the current probe question information
                probe = next((p for p in entry['probes'] if p['id'] == current_pointer), None)
                if not probe: break

                # MLLM inference the probe question
                p_model_ans = target_model.inference(img_path, probe['question'])
                p_is_correct = (p_model_ans.strip().upper() == probe['answer'].upper())

                # Record the probe trajectory: (question, correct answer, model answer, status)
                trajectory.append({
                    "id": probe['id'],
                    "purpose": probe['purpose'],
                    "question": probe['question'],
                    "ground_truth": probe['answer'],
                    "model_response": p_model_ans,
                    "result": "Correct" if p_is_correct else "Incorrect"
                })

                # Jump to the next node based on the probe results
                current_pointer = probe['next_step']['if_correct'] if p_is_correct else probe['next_step']['if_wrong']

            # --- Summarize the information and invoke the Agent for the final error diagnosis ---
            if trajectory:
                self.process_diagnosis(entry, model_ans, trajectory, reference_label)

    def process_diagnosis(self, entry, main_error_ans, trajectory, reference_label):
        """Construct a Prompt and request a diagnostic Agent"""

        # Construct the context information sent to the Agent
        context_msg = f"--- Main Question Analysis ---\n"
        context_msg += f"Question: {entry['main_question']['question']}\n"
        context_msg += f"Correct Answer: {entry['main_question']['answer']}\n"
        context_msg += f"Model Incorrect Answer: {main_error_ans}\n\n"

        context_msg += f"--- Probing Trajectory ---\n"
        for p in trajectory:
            context_msg += f"Probe ID: {p['id']} ({p['purpose']})\n"
            context_msg += f"Probe Question: {p['question']}\n"
            context_msg += f"Model Response: {p['model_response']} (Ground Truth: {p['ground_truth']}) -> {p['result']}\n"

        if reference_label:
            context_msg += f"\n--- Reference Info ---\n"
            context_msg += f"Terminal Decision Tree State: {reference_label}\n"

        # Generate structured diagnostic reports
        final_report = self.agent(context_msg, parse=True)

        # Record the results
        self.final_reports.append({
            "image_file": entry['image_file'],
            "ability": entry['ability'],
            "diagnosis": final_report
        })

        # save results
        with open("error_insight_results.json", "w", encoding="utf-8") as f:
            json.dump(self.final_reports, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    SYSTEM_PROMPT = sys_prompts_list[0]['prompt']
    diag_agent = BaseAgent(system_prompt=SYSTEM_PROMPT)
    test_model = EvaluatedMLLM(model_name="Gemini-2.5-Pro")

    framework = ErrorInsightFramework(
        dataset_path="ChallengeBench.json",
        image_dir="./images",
        diagnostic_agent=diag_agent
    )

    framework.run(test_model)