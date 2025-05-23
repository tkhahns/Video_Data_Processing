"""
Core transcription functionality for speech-to-text processing.
"""

import os
import logging
import torch
import tempfile
import tqdm
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Handle imports differently when run as script vs. as module
if __name__ == "__main__" or os.path.basename(sys.argv[0]) == "__main__.py":
    # Add the parent directory to sys.path for direct script execution
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, parent_dir)
    
    # Import from utils package
    from utils import colored_logging, init_logging
else:
    # Try using both approaches for importing the logging modules
    try:
        # First try absolute imports
        from utils import colored_logging, init_logging
    except ImportError:
        # Fall back to relative imports if the module structure permits it
        try:
            from ...utils import colored_logging, init_logging
        except ImportError:
            # Last resort: try absolute imports with sys.path manipulation
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
            from utils import colored_logging, init_logging

# Get logger for this module
logger = init_logging.get_logger(__name__)

class TranscriptionModel:
    """Base class for transcription models."""
    
    def __init__(self, model_name: str, model_path: Path, device: str = "cuda"):
        """
        Initialize the transcription model.
        
        Args:
            model_name: Name of the model to use
            model_path: Path to the model directory
            device: Device to use for inference ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = "cpu" if not torch.cuda.is_available() and device == "cuda" else device
        self.model = None
        self.processor = None
        
    def load(self):
        """Load the model and processor."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def transcribe(self, audio_path: Path, language: str = "en", diarize: bool = False) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code for transcription
            diarize: Whether to perform speaker diarization
            
        Returns:
            Dictionary containing transcription results
        """
        raise NotImplementedError("Subclasses must implement this method")


class WhisperXModel(TranscriptionModel):
    """WhisperX model for speech-to-text transcription."""
    
    def load(self):
        """Load the WhisperX model."""
        try:
            import whisperx
            logger.info(f"Loading WhisperX model...")
            
            # Explicitly use float32 to avoid precision issues on devices without efficient float16 support
            self.model = whisperx.load_model(
                "large-v2", 
                self.device, 
                download_root=str(self.model_path),
                compute_type="float32"  # Force float32 precision to avoid float16 errors
            )
            logger.info(f"WhisperX model loaded successfully with float32 precision")
            
        except ImportError:
            logger.error("WhisperX not installed. Install with: pip install git+https://github.com/m-bain/whisperX.git")
            raise
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise
    
    def transcribe(self, audio_path: Path, language: str = "en", diarize: bool = False) -> Dict:
        """Transcribe audio using WhisperX."""
        try:
            import whisperx
            
            if self.model is None:
                self.load()
            
            # Check if diarization is requested
            if diarize:
                from . import diarization
                
                logger.info(f"Transcribing {audio_path} with speaker diarization...")
                
                # Step 1: Load the diarization model
                diar_model = diarization.load_diarization_model(device="cpu")  # CPU is often more reliable for pyannote
                
                # Check if diarization model loaded successfully - added additional check
                if diar_model is None:
                    logger.warning("Failed to load diarization model. Falling back to standard transcription.")
                    return self._transcribe_standard(audio_path, language)
                
                # Step 2: Perform diarization to get speaker segments
                speaker_segments = diarization.perform_diarization(audio_path, diar_model)
                if not speaker_segments:
                    logger.warning("No speaker segments found. Falling back to standard transcription.")
                    return self._transcribe_standard(audio_path, language)
                
                # Step 3: Segment the audio by speaker
                audio_segments = diarization.segment_audio(audio_path, speaker_segments)
                if not audio_segments:
                    logger.warning("Failed to segment audio. Falling back to standard transcription.")
                    return self._transcribe_standard(audio_path, language)
                
                # Step 4: Transcribe each segment
                segment_results = {}
                logger.info(f"Transcribing {len(audio_segments)} speaker segments...")
                
                with tqdm.tqdm(total=len(audio_segments), desc="Transcribing speaker segments") as pbar:
                    for idx, (segment_data, sr, start, end, speaker) in audio_segments.items():
                        # Save segment to a temporary file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            import soundfile as sf
                            sf.write(temp_file.name, segment_data, sr)
                            
                            # Transcribe the segment
                            try:
                                segment_result = self.model.transcribe(temp_file.name, language=language)
                                
                                # Extract text with proper error handling
                                if isinstance(segment_result, dict):
                                    # Get text from result dictionary
                                    segment_text = segment_result.get("text", "").strip()
                                    
                                    # If text is empty but there are segments, try to get text from them
                                    if not segment_text and "segments" in segment_result and segment_result["segments"]:
                                        segment_text = " ".join(seg.get("text", "") for seg in segment_result["segments"] if seg.get("text"))
                                else:
                                    # Handle unexpected result type
                                    segment_text = str(segment_result).strip()
                                
                                segment_results[idx] = (start, end, f"Speaker {speaker}", segment_text)
                            except Exception as e:
                                logger.warning(f"Error transcribing segment {idx}: {e}")
                                # Add empty text for failed segment to maintain sequence
                                segment_results[idx] = (start, end, f"Speaker {speaker}", "[Inaudible]")
                            
                            # Clean up the temporary file
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass
                        pbar.update(1)
                
                # Step 5: Merge the transcriptions
                result = diarization.merge_transcriptions(segment_results)
                return result
            else:
                # Standard transcription without diarization
                return self._transcribe_standard(audio_path, language)
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": str(e), "segments": []}
    
    def _transcribe_standard(self, audio_path: Path, language: str = "en") -> Dict:
        """Standard transcription without speaker diarization."""
        try:
            import whisperx
            
            logger.info(f"Transcribing {audio_path}...")
            
            # Transcribe audio
            result = self.model.transcribe(str(audio_path), language=language)
            
            # Try to align whisper output, with graceful fallback if it fails
            try:
                model_a, metadata = whisperx.load_align_model(language_code=language, device=self.device)
                result = whisperx.align(result["segments"], model_a, metadata, str(audio_path), self.device)
            except Exception as align_error:
                # Log the alignment error but continue with unaligned results
                logger.warning(f"Alignment failed, using unaligned results: {align_error}")
                if "segments" not in result:
                    result = {"segments": result["segments"], "text": " ".join(seg["text"] for seg in result["segments"])}
            
            logger.info(f"Transcription complete")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": str(e), "segments": []}


class XLSR_Model(TranscriptionModel):
    """XLSR model from Facebook for speech-to-text transcription."""
    
    def load(self):
        """Load the XLSR model."""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            logger.info(f"Loading XLSR model...")
            model_id = "facebook/wav2vec2-large-xlsr-53-english"
            
            # Load processor and model
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
            
            logger.info(f"XLSR model loaded successfully")
            
        except ImportError:
            logger.error("Transformers library not installed. Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load XLSR model: {e}")
            raise
    
    def transcribe(self, audio_path: Path, language: str = "en", diarize: bool = False) -> Dict:
        """Transcribe audio using XLSR."""
        try:
            import librosa
            import numpy as np
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            if self.model is None or self.processor is None:
                self.load()
            
            # Check if diarization is requested
            if diarize:
                from . import diarization
                
                logger.info(f"Transcribing {audio_path} with speaker diarization...")
                
                # Step 1: Load the diarization model
                diar_model = diarization.load_diarization_model(device="cpu")
                if diar_model is None:
                    logger.warning("Failed to load diarization model. Falling back to standard transcription.")
                    return self._transcribe_standard(audio_path, language)
                
                # Step 2: Perform diarization to get speaker segments
                speaker_segments = diarization.perform_diarization(audio_path, diar_model)
                if not speaker_segments:
                    logger.warning("No speaker segments found. Falling back to standard transcription.")
                    return self._transcribe_standard(audio_path, language)
                
                # Step 3: Segment the audio by speaker
                audio_segments = diarization.segment_audio(audio_path, speaker_segments)
                if not audio_segments:
                    logger.warning("Failed to segment audio. Falling back to standard transcription.")
                    return self._transcribe_standard(audio_path, language)
                
                # Step 4: Transcribe each segment
                segment_results = {}
                logger.info(f"Transcribing {len(audio_segments)} speaker segments...")
                
                with tqdm.tqdm(total=len(audio_segments), desc="Transcribing speaker segments") as pbar:
                    for idx, (segment_data, sr, start, end, speaker) in audio_segments.items():
                        # Resample to 16kHz for XLSR
                        if sr != 16000:
                            segment_data = librosa.resample(segment_data, orig_sr=sr, target_sr=16000)
                            sr = 16000
                            
                        # Process audio
                        inputs = self.processor(segment_data, sampling_rate=16000, return_tensors="pt").to(self.device)
                        
                        # Get logits and predicted transcription
                        with torch.no_grad():
                            logits = self.model(inputs.input_values).logits
                            predicted_ids = torch.argmax(logits, dim=-1)
                            transcription = self.processor.batch_decode(predicted_ids)[0]
                        
                        segment_results[idx] = (start, end, f"Speaker {speaker}", transcription)
                        pbar.update(1)
                
                # Step 5: Merge the transcriptions
                result = diarization.merge_transcriptions(segment_results)
                return result
            else:
                # Standard transcription without diarization
                return self._transcribe_standard(audio_path, language)
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": str(e), "segments": []}
    
    def _transcribe_standard(self, audio_path: Path, language: str = "en") -> Dict:
        """Standard transcription without speaker diarization."""
        try:
            import librosa
            import numpy as np
            
            logger.info(f"Transcribing {audio_path}...")
            
            # Load audio file
            speech_array, sampling_rate = librosa.load(str(audio_path), sr=16000)
            
            # Process audio
            inputs = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").to(self.device)
            
            # Get logits and predicted transcription
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Format result similar to Whisper for consistency
            result = {
                "text": transcription,
                "segments": [{"text": transcription, "start": 0, "end": len(speech_array) / sampling_rate}]
            }
            
            logger.info(f"Transcription complete")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": str(e), "segments": []}


def load_transcription_model(model_name: str, model_dir: Path) -> TranscriptionModel:
    """
    Load the specified transcription model.
    
    Args:
        model_name: Name of the model to load
        model_dir: Directory where models are stored
        
    Returns:
        Loaded transcription model
    """
    model_name = model_name.lower()
    
    if model_name == "whisperx":
        return WhisperXModel(model_name, model_dir)
    elif model_name == "xlsr":
        return XLSR_Model(model_name, model_dir)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    model_name: str = "whisperx",
    model_dir: Path = None,
    language: str = "en",
    output_format: str = "srt",
    diarize: bool = False
) -> Dict:
    """
    Transcribe an audio file and save the result.
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the transcription
        model_name: Name of the model to use
        model_dir: Directory containing models
        language: Language code for transcription
        output_format: Format for saving output: "srt", "txt", or "both"
        diarize: Whether to perform speaker diarization
        
    Returns:
        Dictionary containing transcription results
    """
    # Load the model
    model = load_transcription_model(model_name, model_dir)
    
    # Transcribe the audio
    result = model.transcribe(audio_path, language, diarize=diarize)
    
    # Save the transcription in the specified format(s)
    # Pass diarize flag as with_speakers parameter to ensure speaker labels are included
    save_transcription(result, output_path, output_format, with_speakers=diarize)
    
    return result


def save_transcription(result: Dict, output_path: Path, output_format: str = "srt", with_speakers: bool = False) -> None:
    """
    Save transcription results to a file.
    
    Args:
        result: Transcription results
        output_path: Path to save the transcription
        output_format: Format for saving output: "srt", "txt", or "both"
        with_speakers: Whether to include speaker labels in the output
    """
    # Create the directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Save outputs based on format selection
    saved_files = []
    
    if output_format in ["txt", "both"]:
        # Save as TXT file
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            if "error" in result:
                f.write(f"Transcription error: {result['error']}\n")
            else:
                f.write(result.get("text", "") + "\n\n")
                
                # Write segments with timestamps and speakers if available
                for segment in result.get("segments", []):
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "")
                    speaker = segment.get("speaker", "")
                    
                    # Format timestamp as [MM:SS.mmm]
                    start_str = f"{int(start_time // 60):02d}:{start_time % 60:06.3f}"
                    end_str = f"{int(end_time // 60):02d}:{end_time % 60:06.3f}"
                    
                    # Include speaker if available
                    if with_speakers and speaker:
                        f.write(f"[{start_str} --> {end_str}] {speaker}: {text}\n")
                    else:
                        f.write(f"[{start_str} --> {end_str}] {text}\n")
        saved_files.append(txt_path)
        
    if output_format in ["srt", "both"]:
        # Save as SRT subtitle file
        srt_path = output_path.with_suffix('.srt')
        save_as_srt(result, srt_path, with_speakers)
        saved_files.append(srt_path)
    
    # Log the saved files
    if saved_files:
        logger.info(f"Transcription saved as: {', '.join(str(path) for path in saved_files)}")
    else:
        logger.warning(f"No transcription files were saved. Invalid output format: {output_format}")


def save_as_srt(result: Dict, output_path: Path, with_speakers: bool = False) -> None:
    """
    Save transcription as SRT subtitle file.
    
    Args:
        result: Transcription results
        output_path: Path to save the SRT file
        with_speakers: Whether to include speaker labels in the output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        if "error" in result:
            f.write(f"1\n00:00:00,000 --> 00:00:05,000\nTranscription error: {result['error']}\n\n")
        else:
            for i, segment in enumerate(result.get("segments", []), 1):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "")
                speaker = segment.get("speaker", "")
                
                # Format timestamp for SRT as HH:MM:SS,mmm
                start_h, start_m = divmod(start_time // 60, 60)
                start_s, start_ms = divmod(start_time % 60, 1)
                start_str = f"{int(start_h):02d}:{int(start_m):02d}:{int(start_s):02d},{int(start_ms * 1000):03d}"
                
                end_h, end_m = divmod(end_time // 60, 60)
                end_s, end_ms = divmod(end_time % 60, 1)
                end_str = f"{int(end_h):02d}:{int(end_m):02d}:{int(end_s):02d},{int(end_ms * 1000):03d}"
                
                # Include speaker if available and with_speakers is True
                if with_speakers and speaker:
                    f.write(f"{i}\n{start_str} --> {end_str}\n{speaker}: {text}\n\n")
                else:
                    f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")


def perform_diarization(audio_path: Union[str, Path], pipeline) -> List[Tuple[float, float, str]]:
    """
    Perform speaker diarization on an audio file.
    """
    # Extract segments
    segments = []
    speakers = {}
    speaker_counter = 1
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        end = turn.end
        
        # Create consistent speaker mapping (e.g., "SPEAKER_00" -> "1")
        if speaker not in speakers:
            speakers[speaker] = str(speaker_counter)
            speaker_counter += 1
        
        segments.append((start, end, speakers[speaker]))
    
    return segments
