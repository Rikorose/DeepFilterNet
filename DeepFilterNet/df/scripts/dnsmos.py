import os
import sys

from df.enhance import load_audio
from df.evaluation_utils import dnsmos_api_req

URL_P808 = "https://dnsmos.azurewebsites.net/score"
__a_tol = 1e-4
__r_tol = 1e-4


def isclose(a, b) -> bool:
    return abs(a - b) <= (__a_tol + __r_tol * abs(b))


def main():
    file = sys.argv[1]
    target_value = float(sys.argv[2]) if len(sys.argv) > 2 else None
    key = os.environ["DNS_AUTH_KEY"]
    audio, _ = load_audio(file, sr=16000, verbose=False)
    dnsmos = dnsmos_api_req(URL_P808, key, audio)["mos"]
    print(dnsmos)
    if target_value is not None:
        if not isclose(dnsmos, target_value):
            print(f"Is not close to target: {target_value}")
            exit(2)
    exit(0)


if __name__ == "__main__":
    main()
