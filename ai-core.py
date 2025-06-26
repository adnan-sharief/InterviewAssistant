import os
import PyPDF2
import speech_recognition as sr
import google.generativeai as genai
import pyttsx3
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from scipy.io import wavfile

# === Configure Gemini API Key ===
genai.configure(api_key="YOUR-API-KEY")
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Speak the text using TTS (optional, not used in CLI)
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(text)
    engine.runAndWait()

# === Utility Functions ===
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def detect_domain_info(resume_text, job_text):
    prompt = f"""Analyze the resume and job description below.\nResume:\n{resume_text}\n\nJob Description:\n{job_text}\nWhat is the candidate's probable domain or area of expertise based on the resume and JD?"""
    response = model.generate_content(prompt)
    return response.text.strip()

def match_resume_to_jd(resume_text, job_text):
    prompt = f"""\nCompare the following resume and job description.\n\nResume:\n{resume_text}\n\nJob Description:\n{job_text}\n\nBased on skills, experience, and role alignment, provide a matching score between 0 to 100 indicating how well the resume fits the job description. Only return a number.\n"""
    response = model.generate_content(prompt)
    try:
        match_score = int("".join(filter(str.isdigit, response.text.splitlines()[0])))
        return max(0, min(match_score, 100))
    except:
        return 0

def generate_interview_questions(domain_info):
    prompt = f"""Given the following resume, generate 5 interview questions for the domain: {domain_info}. 
    Do not include any introductory text. Do not include any explanations. Do not include any context. 
    Just return the questions that are relevant to the candidate's experience, skills and domain."""
    response = model.generate_content(prompt)
    # Robustly extract questions: remove bullets, numbers, and empty lines
    questions = []
    for line in response.text.strip().split("\n"):
        q = line.strip().lstrip("•-0123456789. ")
        if q:
            questions.append(q)
    return questions

def analyze_answer_with_gemini(answer, question=None):
    prompt = f"""Analyze the following interview answer{f' to the question: {question}' if question else ''}. Provide a detailed, constructive analysis and suggestions for improvement. Do not return JSON or any structured format, just plain text feedback. Answer: {answer}"""
    response = model.generate_content(prompt)
    return response.text.strip()

def transcribe_audio_file(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError:
        return "Speech recognition service error."

def record_audio_live(filename, fs=16000):
    import sys
    print("Press Enter to start recording...")
    input()
    print("Recording... Press Enter to stop.")
    import sounddevice as sd
    import numpy as np
    import sys
    import select
    audio_buffer = []
    def callback(indata, frames, time, status):
        audio_buffer.append(indata.copy())
    stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback)
    stream.start()
    try:
        input()  # Wait for Enter to stop
    except KeyboardInterrupt:
        pass
    stream.stop()
    stream.close()
    if audio_buffer:
        full_recording = np.concatenate(audio_buffer, axis=0)
        write(filename, fs, full_recording)
        print(f"Recording saved to {filename}")
        # Playback for confirmation
        print("Playing back your recording...")
        sd.play(full_recording, fs)
        sd.wait()
        # Print file info
        rate, data = wavfile.read(filename)
        print(f"[DEBUG] WAV file info: Sample rate: {rate}, Channels: {data.shape[1] if len(data.shape) > 1 else 1}, Duration: {data.shape[0]/rate:.2f}s")
        if (rate != 16000):
            print("[WARNING] Sample rate is not 16kHz. Google STT works best with 16kHz mono WAV.")
        if (len(data.shape) > 1 and data.shape[1] != 1):
            print("[WARNING] Audio is not mono. Google STT expects mono audio.")
    else:
        print("No audio recorded.")

def generate_followup_question(main_question, previous_answers, last_answer):
    """
    Generate a follow-up question based on the main question, previous answers, and the last answer.
    """
    conversation = "\n".join([
        f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(previous_answers)
    ])
    prompt = f"""
You are conducting a technical interview. Given the main question: '{main_question}', and the following conversation so far:\n{conversation}\nThe candidate's last answer was: '{last_answer}'.\nAsk a relevant follow-up question to probe deeper. If no further follow-up is needed, reply with 'NO_FOLLOWUP'. Limit to 5 follow-ups per main question. Do not include any explanations, just the question or 'NO_FOLLOWUP'.
"""
    response = model.generate_content(prompt)
    followup = response.text.strip().split("\n")[0]
    return followup

def analyze_followup_answers(followup_qas, main_question):
    """
    Analyze all answers (main + follow-ups) for a main question and return a short, concise feedback.
    """
    conversation = "\n".join([
        f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(followup_qas)
    ])
    prompt = f"""
Given the following interview conversation for the main question: '{main_question}', provide a short, concise feedback (2-3 sentences) on the candidate's overall performance for this question and its follow-ups. Focus on strengths and areas for improvement. Do not include any explanations or context, just the feedback.\n\n{conversation}
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# === Terminal UI ===
def main():
    print("\n=== AI-Powered Interview Practice Assistant (Terminal Version) ===\n")
    # Resume
    print("Please provide your Resume PDF (must include 'resume' in filename). You can drag and drop the file into the terminal or paste the full path.")
    resume_path = input("Resume PDF path: ").strip().strip('"')
    if not os.path.isfile(resume_path) or "resume" not in resume_path.lower():
        print("[ERROR] Please provide a valid resume PDF file including 'resume' in the filename.")
        return
    # Job Description
    jd_mode = input("Provide Job Description as (1) PDF or (2) Text? Enter 1 or 2: ").strip()
    if jd_mode == "1":
        print("Please provide your Job Description PDF. You can drag and drop the file into the terminal or paste the full path.")
        job_path = input("Job Description PDF path: ").strip().strip('"')
        if not os.path.isfile(job_path):
            print("[ERROR] Invalid job description PDF path.")
            return
        job_description = extract_text_from_pdf(job_path)
    else:
        print("Paste the Job Description text below. End input with a blank line:")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        job_description = "\n".join(lines)
    # Extract resume text
    resume_text = extract_text_from_pdf(resume_path)
    print("\n[INFO] Calculating resume match score...")
    match_score = match_resume_to_jd(resume_text, job_description)
    print(f"Resume Match Score: {match_score}%")
    THRESHOLD = 60
    if match_score < THRESHOLD:
        print(f"[ERROR] Match score is too low to proceed (Required: {THRESHOLD}%). Please refine your resume.")
        return
    print("[INFO] Detecting domain info...")
    domain_info = detect_domain_info(resume_text, job_description)
    print(f"Detected Domain Info: {domain_info}\n")
    # Generate questions
    questions = generate_interview_questions(domain_info)
    print("Generated Interview Questions:")
    for idx, q in enumerate(questions):
        print(f"Q{idx+1}: {q}")
    print("\n--- Answer the questions below ---\n")
    answers_tree = []  # Store all Q&A for each main question and its follow-ups
    main_feedbacks = []  # Store full analysis for main answers
    followup_feedbacks = []  # Store short feedback for follow-ups
    for idx, q in enumerate(questions):
        print(f"Q{idx+1}: {q}")
        speak_text(q)
        print("Answer by (1) Typing, (2) Uploading WAV audio, or (3) Record live with your microphone")
        mode = input("Enter 1, 2, or 3: ").strip()
        if mode == "2":
            print("Please provide your WAV audio file. You can drag and drop the file into the terminal or paste the full path.")
            audio_path = input("WAV audio file path: ").strip().strip('"')
            if not os.path.isfile(audio_path):
                print("[ERROR] Invalid audio file path. Skipping to text input.")
                answer = input("Type your answer: ")
            else:
                answer = transcribe_audio_file(audio_path)
                print(f"[Transcribed]: {answer}")
        elif mode == "3":
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                record_audio_live(tmpfile.name)
                print(f"[DEBUG] Recorded file: {tmpfile.name}")
                try:
                    answer = transcribe_audio_file(tmpfile.name)
                    print(f"[Transcribed]: {answer}")
                except Exception as e:
                    print(f"[ERROR] Transcription failed: {e}")
                    print(f"[DEBUG] You can manually inspect or upload this file: {tmpfile.name}")
                    answer = "[Transcription failed]"
        else:
            answer = input("Type your answer: ")
        if not answer.strip():
            print("[INFO] No answer provided. Skipping to next main question.")
            answers_tree.append([])
            main_feedbacks.append(None)
            followup_feedbacks.append(None)
            continue
        # Full analysis for main answer
        main_feedback = analyze_answer_with_gemini(answer, q)
        main_feedbacks.append(main_feedback)
        # Start follow-up loop
        followup_qas = [(q, answer)]
        skip_followups = False
        for followup_num in range(5):
            followup_q = generate_followup_question(q, followup_qas, followup_qas[-1][1])
            if followup_q.strip().upper() == "NO_FOLLOWUP":
                break
            print(f"Follow-up Q{followup_num+1}: {followup_q}")
            speak_text(followup_q)
            print("Answer by (1) Typing, (2) Uploading WAV audio, or (3) Record live with your microphone")
            mode = input("Enter 1, 2, or 3: ").strip()
            if mode == "2":
                print("Please provide your WAV audio file. You can drag and drop the file into the terminal or paste the full path.")
                audio_path = input("WAV audio file path: ").strip().strip('"')
                if not os.path.isfile(audio_path):
                    print("[ERROR] Invalid audio file path. Skipping to text input.")
                    followup_a = input("Type your answer: ")
                else:
                    followup_a = transcribe_audio_file(audio_path)
                    print(f"[Transcribed]: {followup_a}")
            elif mode == "3":
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                    record_audio_live(tmpfile.name)
                    print(f"[DEBUG] Recorded file: {tmpfile.name}")
                    try:
                        followup_a = transcribe_audio_file(tmpfile.name)
                        print(f"[Transcribed]: {followup_a}")
                    except Exception as e:
                        print(f"[ERROR] Transcription failed: {e}")
                        print(f"[DEBUG] You can manually inspect or upload this file: {tmpfile.name}")
                        followup_a = "[Transcription failed]"
            else:
                followup_a = input("Type your answer: ")
            if not followup_a.strip():
                print("[INFO] No answer provided for follow-up. Skipping to next main question.")
                skip_followups = True
                break
            followup_qas.append((followup_q, followup_a))
        if skip_followups:
            answers_tree.append(followup_qas)
            followup_feedbacks.append(None)
            continue
        answers_tree.append(followup_qas)
        # Short feedback for all follow-ups (excluding main answer)
        if len(followup_qas) > 1:
            followup_feedback = analyze_followup_answers(followup_qas[1:], q)
        else:
            followup_feedback = None
        followup_feedbacks.append(followup_feedback)
    print("\n------------- Feedback -------------\n")
    for idx, qas in enumerate(answers_tree):
        if not qas:
            print(f"Q{idx+1}: No answer provided.")
        else:
            print(f"Q{idx+1} Main Answer Feedback: {main_feedbacks[idx]}\n")
            if followup_feedbacks[idx]:
                print(f"Q{idx+1} Follow-up Feedback: {followup_feedbacks[idx]}\n")

if __name__ == "__main__":
    main()
