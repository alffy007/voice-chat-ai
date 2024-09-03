import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

ckpt_base = 'checkpoints_v2/base_speakers/EN'
ckpt_converter = 'checkpoints_v2/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'
print(f"Using device: {device}")
base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

source_se = torch.load(f'{ckpt_base}/en_style_se.pth').to(device)


reference_speaker = 'characters\missminutes\missminutes.mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)


save_path = f'{output_dir}/output_en_default.wav'

# Run the base speaker tts
text = "You are the most humblest person i have ever met!"
src_path = f'{output_dir}/tmp.wav'
base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=0.9)


# Run the tone color converter
encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path=src_path,
    src_se=source_se,
    tgt_se=target_se,
    output_path=save_path,
    message=encode_message)



# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.device_count())
