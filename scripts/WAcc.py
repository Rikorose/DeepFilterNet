import argparse
import glob
import os

import librosa
import numpy as np
import pandas
import pandas as pd
import requests
import soundfile as sf

WACC_SERVICE_URL = "https://wacc.azurewebsites.net/api/TriggerEvaluation?code=K2XN7ouruRN/2k1HNyS79ET39rEMZ9jOOCnFtodPDj42WJFjG9LWXg=="
SUPPORTED_SAMPLING_RATE = 16000


def main(args):
    audio_clips_list = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    transcriptions_df = pd.read_csv(
        args.transcription_file, sep="\t", names=["filename", "transcription"]
    )
    scores = []
    for fpath in audio_clips_list:
        if os.path.basename(fpath) not in transcriptions_df["filename"].unique():
            continue
        original_audio, fs = sf.read(fpath)
        if fs != SUPPORTED_SAMPLING_RATE:
            print("Only sampling rate of 16000 is supported as of now so resampling audio")
            audio = librosa.core.resample(original_audio, fs, SUPPORTED_SAMPLING_RATE)
            sf.write(fpath, audio, SUPPORTED_SAMPLING_RATE)

        try:
            with open(fpath, "rb") as f:
                resp = requests.post(WACC_SERVICE_URL, files={"audiodata": f})
            wacc = resp.json()
        except:  # noqa: E722
            print("Error occured during scoring")
            print("response is ", resp)
        sf.write(fpath, original_audio, fs)
        score_dict = {"file_name": os.path.basename(fpath), "wacc": wacc}
        scores.append(score_dict)

    df = pd.DataFrame(scores)
    print("Mean WAcc for the files is ", np.mean(df["wacc"]))

    if args.score_file:
        df.to_csv(args.score_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testset_dir",
        required=True,
        help="Path to the dir containing audio clips to be evaluated",
    )
    parser.add_argument("--transcription_file")
    parser.add_argument(
        "--score_file", help="If you want the scores in a CSV file provide the full path"
    )

    args = parser.parse_args()
    main(args)
