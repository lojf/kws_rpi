import os
import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# ---------------------------------------------------------
# KONFIGURATION (Skal passe til jeres script.py)
# ---------------------------------------------------------
SR = 16000          # 16 kHz sample rate
DURATION = 1.0      # 1.0 sekund pr. optagelse
DEVICE_INDEX = 1 # Sæt til et tal (f.eks. 1 eller 2), hvis I vil tvinge en specifik USB-mikrofon

# Vælg den klasse I vil optage i øjeblikket:
# Muligheder: "Lager_1", "Lager_2", "Stop", "Unknown", "Noise"
VALGT_KLASSE = "Unknown"

# ---------------------------------------------------------
GEM_MAPPE = f"./lyd_data/{VALGT_KLASSE}"
os.makedirs(GEM_MAPPE, exist_ok=True)

print("=" * 50)
print(f"  OPTAGER DATA FOR KLASSE: '{VALGT_KLASSE}'")
print("=" * 50)
print("Instruktion:")
print(" - Tryk ENTER for at starte 1 sekunds optagelse.")
print(" - Skriv 'skift' og tryk ENTER for at skifte klasse.")
print(" - Tryk Ctrl+C for at afslutte programmet.\n")

count = 1

try:
    while True:
        bruger_input = input(f"[{VALGT_KLASSE}] Klar til optagelse #{count}? (ENTER = optag): ").strip()
        
        if bruger_input.lower() == 'skift':
            ny_klasse = input("Indtast ny klasse (f.eks. Lager_2, Stop, Noise): ").strip()
            if ny_klasse:
                VALGT_KLASSE = ny_klasse
                GEM_MAPPE = f"./lyd_data/{VALGT_KLASSE}"
                os.makedirs(GEM_MAPPE, exist_ok=True)
                count = 1
                print(f"\n---> Skiftede til klasse: '{VALGT_KLASSE}'\n")
            continue

        print("Optager NOW...")
        # Optag 1 sekund i float32
        audio = sd.rec(
            int(DURATION * SR), 
            samplerate=SR, 
            channels=1, 
            dtype='float32', 
            device=DEVICE_INDEX
        )
        sd.wait() # Vent til 1 sek er gået
        print("Optagelse færdig!")

        # Generer filnavn med tidsstempel
        timestamp = int(time.time())
        filnavn = os.path.join(GEM_MAPPE, f"{VALGT_KLASSE}_{timestamp}_{count}.wav")
        
        # Konverter float32 [-1.0, 1.0] til int16 [-32768, 32767] for ren standard wav-fil
        audio_int16 = (audio * 32767).astype(np.int16)
        write(filnavn, SR, audio_int16)

        print(f"Gemte: {filnavn}\n")
        count += 1

except KeyboardInterrupt:
    print("\n\nOptagelse stoppet. Klar til træning!")