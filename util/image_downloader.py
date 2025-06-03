import os
import requests
import time

def download_faces(folder, st_stevilo):
    url = "https://thispersondoesnotexist.com/"

    for i in range(1, st_stevilo + 1):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                filename = os.path.join(folder, f"oseba_{i:04d}.jpg")
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"[{i}/{st_stevilo}] Slika shranjena: {filename}")
            else:
                print(f"Napaka pri prenosu slike {i}: Status {response.status_code}")
        except Exception as e:
            print(f"Napaka pri prenosu slike {i}: {e}")

        time.sleep(1)

if __name__ == '__main__':
    download_folder = "../data/raw"
    st_slik = 10

    download_faces(download_folder, st_slik)
