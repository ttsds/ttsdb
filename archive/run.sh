sudo apt install ffmpeg
pip install -r requirements.txt

if [ $1 == "naturalspeech2" ]; then
    docker stop $(docker ps -a -q)
    cd containers/amphion && docker build -t amphion . && docker run -p 8000:8000 amphion &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/amphion_ns2/librittsr --tts_system amphion --tts_version "NaturalSpeech 2"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/amphion_ns2/emilia --tts_system amphion --tts_version "NaturalSpeech 2"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/amphion_ns2/librilatest --tts_system amphion --tts_version "NaturalSpeech 2"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/amphion_ns2/myst --tts_system amphion --tts_version "NaturalSpeech 2"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/amphion_ns2/torgo --tts_system amphion --tts_version "NaturalSpeech 2"
elif [ $1 == "valle" ]; then
    docker stop $(docker ps -a -q)
    cd containers/amphion && docker build -t amphion . && docker run -p 8000:8000 amphion &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/amphion_valle/librittsr --tts_system amphion --tts_version "VALL-E v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/amphion_valle/emilia --tts_system amphion --tts_version "VALL-E v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/amphion_valle/librilatest --tts_system amphion --tts_version "VALL-E v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/amphion_valle/myst --tts_system amphion --tts_version "VALL-E v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/amphion_valle/torgo --tts_system amphion --tts_version "VALL-E v1"
elif [ $1 == "bark" ]; then
    docker stop $(docker ps -a -q)
    cd containers/bark && docker build -t bark . && docker run -p 8000:8000 bark &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/bark/librittsr --tts_system bark --tts_version "Bark"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/bark/emilia --tts_system bark --tts_version "Bark"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/bark/librilatest --tts_system bark --tts_version "Bark"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/bark/myst --tts_system bark --tts_version "Bark"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/bark/torgo --tts_system bark --tts_version "Bark"
elif [ $1 == "e2" ]; then
    docker stop $(docker ps -a -q)
    cd containers/f5e2 && docker build -t f5e2 . && docker run -p 8000:8000 f5e2 &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/e2/librittsr --tts_system f5e2 --tts_version "E2-TTS"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/e2/emilia --tts_system f5e2 --tts_version "E2-TTS"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/e2/librilatest --tts_system f5e2 --tts_version "E2-TTS"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/e2/myst --tts_system f5e2 --tts_version "E2-TTS"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/e2/torgo --tts_system f5e2 --tts_version "E2-TTS"
elif [ $1 == "fish" ]; then
    docker stop $(docker ps -a -q)
    cd containers/fish && docker build -t fish . && docker run -p 8000:8000 fish &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/fish/librittsr --tts_system fish --tts_version "Fish"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/fish/emilia --tts_system fish --tts_version "Fish"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/fish/librilatest --tts_system fish --tts_version "Fish"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/fish/myst --tts_system fish --tts_version "Fish"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/fish/torgo --tts_system fish --tts_version "Fish"
elif [ $1 == "f5" ]; then
    docker stop $(docker ps -a -q)
    cd containers/f5e2 && docker build -t f5e2 . && docker run -p 8000:8000 f5e2 &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/f5/librittsr --tts_system f5e2 --tts_version "F5-TTS"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/f5/emilia --tts_system f5e2 --tts_version "F5-TTS"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/f5/librilatest --tts_system f5e2 --tts_version "F5-TTS"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/f5/myst --tts_system f5e2 --tts_version "F5-TTS"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/f5/torgo --tts_system f5e2 --tts_version "F5-TTS"
elif [ $1 == "gpt-sovits" ]; then
    docker stop $(docker ps -a -q)
    cd containers/gpt-sovits && docker build -t gpt-sovits . && docker run -p 8000:8000 gpt-sovits &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/gpt-sovits/librittsr --tts_system gpt-sovits --tts_version "GPT-SoVITS"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/gpt-sovits/emilia --tts_system gpt-sovits --tts_version "GPT-SoVITS"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/gpt-sovits/librilatest --tts_system gpt-sovits --tts_version "GPT-SoVITS"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/gpt-sovits/myst --tts_system gpt-sovits --tts_version "GPT-SoVITS"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/gpt-sovits/torgo --tts_system gpt-sovits --tts_version "GPT-SoVITS"
elif [ $1 == "hierspeechpp" ]; then
    docker stop $(docker ps -a -q)
    cd containers/hierspeechpp && docker build -t hierspeechpp . && docker run -p 8000:8000 hierspeechpp &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/hierspeechpp/librittsr --tts_system hierspeechpp --tts_version "v1.1"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/hierspeechpp/emilia --tts_system hierspeechpp --tts_version "v1.1"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/hierspeechpp/librilatest --tts_system hierspeechpp --tts_version "v1.1"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/hierspeechpp/myst --tts_system hierspeechpp --tts_version "v1.1"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/hierspeechpp/torgo --tts_system hierspeechpp --tts_version "v1.1"
elif [ $1 == "metavoice" ]; then
    docker stop $(docker ps -a -q)
    cd containers/metavoice && docker build -t metavoice . && docker run -p 8000:8000 metavoice &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/metavoice/librittsr --tts_system metavoice --tts_version "metavoice"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/metavoice/emilia --tts_system metavoice --tts_version "metavoice"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/metavoice/librilatest --tts_system metavoice --tts_version "metavoice"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/metavoice/myst --tts_system metavoice --tts_version "metavoice"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/metavoice/torgo --tts_system metavoice --tts_version "metavoice"
elif [ $1 == "openvoice" ]; then
    docker stop $(docker ps -a -q)
    cd containers/openvoice && docker build -t openvoice . && docker run -p 8000:8000 openvoice &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/openvoice/librittsr --tts_system openvoice --tts_version "OpenVoice_v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/openvoice/emilia --tts_system openvoice --tts_version "OpenVoice_v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/openvoice/librilatest --tts_system openvoice --tts_version "OpenVoice_v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/openvoice/myst --tts_system openvoice --tts_version "OpenVoice_v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/openvoice/torgo --tts_system openvoice --tts_version "OpenVoice_v1"
elif [ $1 == "parlertts" ]; then
    docker stop $(docker ps -a -q)
    cd containers/parlertts && docker build -t parlertts . && docker run -p 8000:8000 parlertts &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/parlertts/librittsr --tts_system parlertts --tts_version "Large-v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/parlertts/emilia --tts_system parlertts --tts_version "Large-v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/parlertts/librilatest --tts_system parlertts --tts_version "Large-v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/parlertts/myst --tts_system parlertts --tts_version "Large-v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/parlertts/torgo --tts_system parlertts --tts_version "Large-v1"
elif [ $1 == "pheme" ]; then
    docker stop $(docker ps -a -q)
    cd containers/pheme && docker build -t pheme . && docker run -e HF_TOKEN=$HF_TOKEN -p 8000:8000 pheme &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/pheme/librittsr --tts_system pheme --tts_version "Pheme"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/pheme/emilia --tts_system pheme --tts_version "Pheme"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/pheme/librilatest --tts_system pheme --tts_version "Pheme"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/pheme/myst --tts_system pheme --tts_version "Pheme" --timeout 180
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/pheme/torgo --tts_system pheme --tts_version "Pheme" --timeout 180
elif [ $1 == "speecht5" ]; then
    docker stop $(docker ps -a -q)
    cd containers/speecht5 && docker build -t speecht5 . && docker run -p 8000:8000 speecht5 &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/speecht5/librittsr --tts_system speecht5 --tts_version "SpeechT5"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/speecht5/emilia --tts_system speecht5 --tts_version "SpeechT5"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/speecht5/librilatest --tts_system speecht5 --tts_version "SpeechT5"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/speecht5/myst --tts_system speecht5 --tts_version "SpeechT5"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/speecht5/torgo --tts_system speecht5 --tts_version "SpeechT5"
elif [ $1 == "styletts2" ]; then
    docker stop $(docker ps -a -q)
    cd containers/styletts2 && docker build -t styletts2 . && docker run -p 8000:8000 styletts2 &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/styletts2/librittsr --tts_system styletts2 --tts_version "StyleTTS2"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/styletts2/emilia --tts_system styletts2 --tts_version "StyleTTS2"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/styletts2/librilatest --tts_system styletts2 --tts_version "StyleTTS2"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/styletts2/myst --tts_system styletts2 --tts_version "StyleTTS2"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/styletts2/torgo --tts_system styletts2 --tts_version "StyleTTS2"
elif [ $1 == "tortoise" ]; then
    docker stop $(docker ps -a -q)
    cd containers/tortoise && docker build -t tortoise . && docker run -p 8000:8000 tortoise &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/tortoise/librittsr --tts_system tortoise --tts_version "tortoise"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/tortoise/emilia --tts_system tortoise --tts_version "tortoise"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/tortoise/librilatest --tts_system tortoise --tts_version "tortoise"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/tortoise/myst --tts_system tortoise --tts_version "tortoise"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/tortoise/torgo --tts_system tortoise --tts_version "tortoise"
elif [ $1 == "voicecraft" ]; then
    docker stop $(docker ps -a -q)
    cd containers/voicecraft && docker build -t voicecraft . && docker run -p 8000:8000 voicecraft &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/voicecraft/librittsr --tts_system voicecraft --tts_version "830M_TTSEnhanced"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/voicecraft/emilia --tts_system voicecraft --tts_version "830M_TTSEnhanced"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/voicecraft/librilatest --tts_system voicecraft --tts_version "830M_TTSEnhanced"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/voicecraft/myst --tts_system voicecraft --tts_version "830M_TTSEnhanced"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/voicecraft/torgo --tts_system voicecraft --tts_version "830M_TTSEnhanced"
elif [ $1 == "whisperspeech" ]; then
    docker stop $(docker ps -a -q)
    cd containers/whisperspeech && docker build -t whisperspeech . && docker run -p 8000:8000 whisperspeech &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/whisperspeech/librittsr --tts_system whisperspeech --tts_version "Medium"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/whisperspeech/emilia --tts_system whisperspeech --tts_version "Medium"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/whisperspeech/librilatest --tts_system whisperspeech --tts_version "Medium"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/whisperspeech/myst --tts_system whisperspeech --tts_version "Medium"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/whisperspeech/torgo --tts_system whisperspeech --tts_version "Medium"
elif [ $1 == "xtts" ]; then
    docker stop $(docker ps -a -q)
    cd containers/xtts && docker build -t xtts . && docker run -p 8000:8000 xtts &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/xtts/librittsr --tts_system xtts --tts_version "v2"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/xtts/emilia --tts_system xtts --tts_version "v2"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/xtts/librilatest --tts_system xtts --tts_version "v2"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/xtts/myst --tts_system xtts --tts_version "v2"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/xtts/torgo --tts_system xtts --tts_version "v2"
fi