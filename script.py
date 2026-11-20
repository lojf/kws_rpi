import time
import joblib
import numpy as np
import librosa
import sounddevice as sd

# ==============================================================================
# KONFIGURATION (Skal matche jeres script.py)
# ==============================================================================
SR = 16000               # Sample rate i Hz
DURATION = 1.0           # Buffer-størrelse (1 sekund)
SAMPLES = int(SR * DURATION)
N_MFCC = 13
CONFIDENCE_THRESHOLD = 0.75  # Krav: 75% sikkerhed før en kommando udføres
ENERGY_THRESHOLD = 0.015     # RMS energi-grænse (Energy Gate for at ignorere stilhed)
DEVICE_INDEX = 1          # Sæt til f.eks. 1 eller 2 hvis specifik USB-mic ønskes

KLASSER = ["Lager_1", "Lager_2", "Stop", "Unknown", "Noise"]

# ==============================================================================
# INDLÆS MODEL OG SCALER
# ==============================================================================
print("Indlæser kws_model.joblib...")
model_data = joblib.load('kws_model.joblib')
clf = model_data['model']
scaler = model_data['scaler']
print("Model succesfuldt indlæst!")

# Cirkulær buffer til 1 sekunds lyddata
audio_buffer = np.zeros(SAMPLES, dtype=np.float32)
is_robot_moving = False  # Sæt til True når robotten kører for at pausere lytning

# Callback-funktion som sounddevice kalder hver gang der kommer nye audio-samples
def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        print(f"Audio status advarsel: {status}")
    
    # Skub bufferen til venstre og tilføj de nyeste lyd-samples i slutningen
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

# ==============================================================================
# HOVED-LOOP (Sliding Window Inference)
# ==============================================================================
print("\n" + "="*50)
print("  KWS STEMMESTYRING ER NULSTILLET OG LYTTER PÅ MIKROFON")
print("="*50 + "\n")

# Start audio-stream fra USB-mikrofonen
with sd.InputStream(samplerate=SR, channels=1, callback=audio_callback, blocksize=2000, device=DEVICE_INDEX):
    last_detection_time = 0
    
    while True:
        # Tjek forudsigelser ca. hver 250 ms (4 gange i sekundet = sliding window)
        time.sleep(0.25)

        # Hvis robotten er i gang med at køre, springer vi lytning over (undgår motorstøj)
        if is_robot_moving:
            continue

        # Spærretid: Undgå dobbeltdetektering af det samme ord (Cooldown på 1.2 sekunder)
        if time.time() - last_detection_time < 1.2:
            continue

        # 1. Energy Gate: Udregn RMS energi i lydbufferen
        rms_energy = np.sqrt(np.mean(audio_buffer**2))
        if rms_energy < ENERGY_THRESHOLD:
            continue  # For stille i rummet - spring tung Machine Learning over

        # 2. Feature Ekstraktion (Exakt 1:1 med træningen i librosa)
        mfcc_feat = librosa.feature.mfcc(
            y=audio_buffer, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=250
        )
        features = mfcc_feat.flatten().reshape(1, -1)

        # 3. Skalering og Inferens med MLP
        scaled_features = scaler.transform(features)
        probabilities = clf.predict_proba(scaled_features)[0]
        
        best_class_idx = np.argmax(probabilities)
        max_prob = probabilities[best_class_idx]
        detected_class = KLASSER[best_class_idx]

        # 4. Evaluering af forudsigelse
        if max_prob >= CONFIDENCE_THRESHOLD:
            # Reager kun hvis det er en af jeres handlings-kommandoer
            if detected_class in ["Lager_1", "Lager_2", "Stop"]:
                print(f" >>> DETEKTERING: {detected_class} ({max_prob * 100:.1f}%) <<<")
                last_detection_time = time.time()

                # Udfør handlings-logik for robotten:
                if detected_class == "Lager_1":
                    print("[HANDLING] Robot navigering igangsat: Kører til Lager 1")
                    # Her kaldes ROS 2 Goal / Action eller Nav2 API
                
                elif detected_class == "Lager_2":
                    print("[HANDLING] Robot navigering igangsat: Kører til Lager 2")
                    # Her kaldes ROS 2 Goal / Action eller Nav2 API
                
                elif detected_class == "Stop":
                    print("[HANDLING] NØDSTOP AKTIVERET")
                    # Her kaldes ROS 2 Cancel Goal