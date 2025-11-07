import sys
import argparse
from pathlib import Path

sys.path.append("orchestrator")

from tqdm import tqdm

from tts_api import TTSApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tts_system", type=str, required=True)
    parser.add_argument("--tts_version", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=-1)
    args = parser.parse_args()
    api = TTSApi({args.tts_system: 8000}, use_docker=True)
    dir_paths = sorted(list(Path(args.source_audio_dir).iterdir()))
    dir_paths = [p.name for p in dir_paths if p.is_dir()]
    ab = False
    if 'A' in dir_paths and 'B' in dir_paths:
        reference_dir = Path(args.source_audio_dir) / 'A'
        transcript_dir = Path(args.source_audio_dir) / 'B'
        ab = True
    else:
        reference_dir = Path(args.source_audio_dir)
        transcript_dir = Path(args.source_audio_dir)
    audio_files = sorted(list(Path(reference_dir).rglob("*.wav")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fail_count = 0
    for audio_file in tqdm(audio_files):
        try:
            if not ab:
                text_file = audio_file.with_suffix(".txt")
            else:
                if audio_file.parent.name not in ['A', 'B']:
                    text_file = transcript_dir / audio_file.parent.name / audio_file.name
                else:
                    text_file = transcript_dir / audio_file.name
                text_file = text_file.with_suffix(".txt")
                input_text = audio_file.with_suffix(".txt")
                input_text = input_text.read_text()
            with open(text_file, "r") as f:
                text = f.read()
            # respect file structure in source_audio_dir
            if not ab:
                relative_path = audio_file.relative_to(Path(args.source_audio_dir))
            else:
                relative_path = audio_file.name
            output_file = output_dir / relative_path
            output_file.parent.mkdir(exist_ok=True, parents=True)

            if output_file.with_suffix(".wav").exists() and output_file.with_suffix(".txt").exists():
                continue

            if args.timeout > 0:
                try:
                    audio_bytes = api.synthesize(text, args.tts_system, args.tts_version, audio_file, input_text=input_text, timeout=args.timeout)
                except Exception as e:
                    print(f"Timeout occurred for {audio_file}: {e}")
                    continue
            audio_bytes = api.synthesize(text, args.tts_system, args.tts_version, audio_file, input_text=input_text)
            with open(output_file.with_suffix(".wav"), "wb") as f:
                f.write(audio_bytes)
            with open(output_file.with_suffix(".txt"), "w") as f:
                f.write(text)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            fail_count += 1
            if fail_count > 10:
                raise e
        
if __name__ == "__main__":
    main()