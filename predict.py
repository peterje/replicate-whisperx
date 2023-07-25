import os
os.environ['HF_HOME'] = '/src/hf_models'
os.environ['TORCH_HOME'] = '/src/torch_models'
from cog import BasePredictor, Input, Path
import torch
import whisperx
import json

compute_type="float16"
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.language_code = "en"
        self.model = whisperx.load_model("large-v2", self.device, language=self.language_code, compute_type=compute_type)
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code=self.language_code, device=self.device)

    def predict(
        self,
        audio_url: Path = Input(description="A url of a publicly available audio file."),
        lang: str = Input(description="The language code of the audio", default='en'),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
        word_level: bool = Input(description="Enable word-level transcription timestamps", default=False),
        only_text: bool = Input(description="Set if you only want to return text; otherwise, segment metadata will be returned as well.", default=False),
        debug: bool = Input(description="Print out memory usage information.", default=False)
    ) -> str:
        """Run a single prediction on the model"""
        with torch.inference_mode():
            result = self.model.transcribe(str(audio_url), batch_size=batch_size, lang=lang) 
            if word_level: result = whisperx.align(result['segments'], self.alignment_model, self.metadata, str(audio), self.device, return_char_alignments=False)
            if only_text: return ''.join([val.text for val in result['segments']])
            if debug: print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
        return json.dumps(result['segments'])

