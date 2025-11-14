import os
import pandas as pd
from audio_io import playback_audio
done = ["b","a","b","b","b","a","a","a","b","0","b","b","b","b","b","b","b","b","b","b","a","a","a","b","b","b","b","0","b","b","a","a","b","b","a","0","b","b",
"0","b","b","a","a","a","a","a","a","b","a","b","b","b","b","b","b","a","b","a","b","b","0","a","b","a","b","a","a","a","b","b",
"b","b","a","a","a","a","a","a","a","a","a","b","0","b","b","b","b","a","b","b","b","a","b","a","a","a","b","a","b","b","0","b",
"a","b","b","a","a","a","0","b","a","a","b","a","b",]
# Collect user input
def get_user_input():
    # code by ChatGPT
    while True:
        response = input("Enter your response (a / b / 0 for neutral): ").strip()
        if response in ["a", "b", "0"]:
            if response == "a":
                return 1
            elif response == "b":
                return -1
            else:
                return 0
        print("Invalid input. Please enter a, b, or 0.")
results = []
base_path = "a4_output/tts"
models = os.listdir(base_path)
files = os.listdir(f"{base_path}/{models[0]}")

# Iterate through models and files -- we are actually grouping by files not models which is the opposite of how i saved it. 
count = 0
for audio_file in files:
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            i_audio_file = f"{base_path}/{models[i]}/{audio_file}"
            j_audio_file = f"{base_path}/{models[j]}/{audio_file}"
            if count >=len(done):
                playback_audio(i_audio_file)
                playback_audio(j_audio_file)
                user_response = get_user_input()
                if user_response != 0:
                    if user_response == 1:
                        results.append({
                            "model": models[i],
                            "prompt": audio_file.split(".")[0]
                        })
                    else:
                        results.append({
                            "model": models[j],
                            "prompt": audio_file.split(".")[0]
                        })
            else:
                response = done[count]
                if response == "a":
                    response= 1
                elif response == "b":
                    response= -1
                else:
                    response= 0
                if response != 0:
                    if response == 1:
                        results.append({
                            "model": models[i],
                            "prompt": audio_file.split(".")[0]
                        })
                    else:
                        results.append({
                            "model": models[j],
                            "prompt": audio_file.split(".")[0]
                        })
                count += 1
# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("a4_output/ab_test_results.csv", index=False)
print("\nAll responses saved to ab_test_results.csv")