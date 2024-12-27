import sys
import os
from datetime import datetime
import warnings
import queue
import threading
import numpy as np
import torch
import sounddevice as sd
import whisper
from pydub import AudioSegment
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QComboBox, 
                           QTextEdit, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, QTimer
from PyQt6.QtGui import QIcon, QPainter, QColor, QPen, QLinearGradient, QPalette, QFont, QRadialGradient
from PyQt6.QtNetwork import QLocalSocket, QLocalServer

# Suppress warnings
warnings.filterwarnings('ignore')

class RecordingIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)  # Doubled the size
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(800)  # Blink every 800ms
        self.is_visible = True
        
    def paintEvent(self, event):
        if self.is_visible:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw the red dot with an enhanced glow effect
            painter.setPen(Qt.PenStyle.NoPen)
            
            # Larger glow effect
            glow_gradient = QRadialGradient(12, 12, 16)  # Centered and larger radius
            glow_gradient.setColorAt(0, QColor(239, 68, 68, 200))  # Brighter red glow
            glow_gradient.setColorAt(0.5, QColor(239, 68, 68, 100))
            glow_gradient.setColorAt(1, QColor(239, 68, 68, 0))
            painter.setBrush(glow_gradient)
            painter.drawEllipse(0, 0, 24, 24)
            
            # Larger core dot
            painter.setBrush(QColor(239, 68, 68))  # Solid red
            painter.drawEllipse(4, 4, 16, 16)  # Larger central dot
    
    def update_visibility(self):
        self.is_visible = not self.is_visible
        self.update()

class WaveformVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.waveform_data = np.zeros(100)
        self.animation_phase = 0
        
        # Setup animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)
        
        # Setup gradient colors
        self.gradient_colors = [
            QColor(37, 99, 235),  # Blue-600
            QColor(79, 70, 229)   # Indigo-600
        ]
        
    def update_waveform(self, audio_data):
        if len(audio_data) > 0:
            # Normalize and reshape audio data
            audio_flat = audio_data.flatten()
            chunk_size = max(1, len(audio_flat) // 100)
            chunks = [audio_flat[i:i + chunk_size] for i in range(0, len(audio_flat), chunk_size)]
            self.waveform_data = np.array([np.abs(chunk).mean() for chunk in chunks if len(chunk) > 0])
            
            # Normalize to 0-1 range
            if self.waveform_data.max() > 0:
                self.waveform_data = self.waveform_data / self.waveform_data.max()
            
            self.update()
    
    def update_animation(self):
        self.animation_phase += 0.1
        if self.animation_phase >= 2 * np.pi:
            self.animation_phase = 0
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create gradient
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, self.gradient_colors[0])
        gradient.setColorAt(1, self.gradient_colors[1])
        
        # Calculate dimensions
        width = self.width()
        height = self.height()
        center_y = height / 2
        bar_width = width / len(self.waveform_data)
        
        # Animation factor
        animation_factor = 0.7 + 0.3 * np.sin(self.animation_phase)
        
        # Draw waveform bars
        painter.setPen(Qt.PenStyle.NoPen)
        for i, amplitude in enumerate(self.waveform_data):
            x = i * bar_width
            # Apply animation factor to bar height
            bar_height = amplitude * height * 0.8 * animation_factor
            
            painter.setBrush(gradient)
            painter.drawRoundedRect(
                int(x + 1), int(center_y - bar_height/2),
                int(max(1, bar_width - 2)), int(bar_height),
                2, 2
            )

class ModernButton(QPushButton):
    def __init__(self, text, icon_name=None, is_primary=True):
        super().__init__(text)
        self.setFixedHeight(48)
        
        # Simplified style without backdrop-filter
        style = f"""
            QPushButton {{
                background-color: {('#2563eb' if is_primary else '#1f2937')};
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {('#1d4ed8' if is_primary else '#374151')};
                border: 1px solid rgba(255, 255, 255, 0.3);
            }}
            QPushButton:pressed {{
                background-color: {('#1e40af' if is_primary else '#4b5563')};
                border: 1px solid rgba(255, 255, 255, 0.4);
            }}
            QPushButton:disabled {{
                background-color: #6b7280;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
        """
        self.setStyleSheet(style)

class ModernComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(48)
        self.setStyleSheet("""
            QComboBox {
                background-color: #1f2937;
                color: white;
                border: 2px solid #374151;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 16px;
                min-width: 250px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 20px;
            }
            QComboBox::down-arrow {
                width: 20px;
                height: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #1f2937;
                color: white;
                selection-background-color: #2563eb;
                border: 2px solid #374151;
                border-radius: 12px;
                padding: 8px;
            }
        """)

class AudioInputThread(QThread):
    audio_captured = pyqtSignal(np.ndarray)
    
    def __init__(self, device_id):
        super().__init__()
        self.device_id = device_id
        self.running = True
        
    def run(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            if self.running:
                self.audio_captured.emit(indata.copy())
        
        try:
            with sd.InputStream(
                device=self.device_id,
                channels=1,
                callback=audio_callback,
                samplerate=16000,
                blocksize=512,
                dtype=np.float32
            ):
                while self.running:
                    sd.sleep(100)
        except Exception as e:
            print(f"Audio input error: {e}")
    
    def stop(self):
        self.running = False

class TranscriptionThread(QThread):
    text_ready = pyqtSignal(str)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = True
        self.audio_queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        self.accumulated_text = []
        
    def add_audio(self, audio_data):
        if self.running:
            self.buffer = np.append(self.buffer, audio_data.flatten())
            if len(self.buffer) >= 16000 * 2:  # Process every 2 seconds
                self.audio_queue.put(self.buffer.copy())
                self.buffer = np.array([], dtype=np.float32)
    
    def run(self):
        while self.running:
            try:
                if not self.audio_queue.empty():
                    audio = self.audio_queue.get()
                    if len(audio) > 0:
                        result = self.model.transcribe(
                            audio,
                            fp16=False,
                            language='en',
                            task='transcribe',
                            without_timestamps=True
                        )
                        
                        text = result["text"].strip()
                        if text:
                            self.accumulated_text.append(text)
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            formatted_text = self.format_text(' '.join(self.accumulated_text))
                            self.text_ready.emit(f"[{timestamp}]\n{formatted_text}")
                            
                QThread.msleep(10)
            except Exception as e:
                print(f"Transcription error: {e}")
                QThread.msleep(100)
    
    def format_text(self, text):
        # Add proper spacing after punctuation
        text = text.replace('.', '. ').replace('?', '? ').replace('!', '! ')
        
        # Split into sentences
        sentences = text.split('. ')
        
        # Group sentences into paragraphs (roughly 3-4 sentences per paragraph)
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_paragraph.append(sentence)
            
            if len(current_paragraph) >= 3:
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
        
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph) + '.')
        
        # Join paragraphs with double newline
        return '\n\n'.join(paragraphs)
    
    def stop(self):
        self.running = False

class FileTranscriptionThread(QThread):
    progress = pyqtSignal(int)
    chunk_transcribed = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, file_path, chunk_size=30000):  # 30 seconds chunks
        super().__init__()
        self.model = model
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.running = True

    def run(self):
        try:
            # Load audio file
            audio = AudioSegment.from_file(self.file_path)
            
            # Calculate number of chunks
            duration_ms = len(audio)
            num_chunks = math.ceil(duration_ms / self.chunk_size)
            
            # Process each chunk
            accumulated_text = []
            for i in range(num_chunks):
                if not self.running:
                    break
                    
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, duration_ms)
                
                # Extract chunk and export to temporary file
                chunk = audio[start:end]
                temp_path = f"temp_chunk_{i}.wav"
                chunk.export(temp_path, format="wav")
                
                # Transcribe chunk
                result = self.model.transcribe(
                    temp_path,
                    fp16=False,
                    language='en',
                    task='transcribe',
                    without_timestamps=True
                )
                
                # Clean up temp file
                os.remove(temp_path)
                
                # Format and emit chunk text
                text = result['text'].strip()
                if text:
                    accumulated_text.append(text)
                    formatted_text = self.format_text(' '.join(accumulated_text))
                    self.chunk_transcribed.emit(formatted_text)
                
                # Update progress
                progress = int((i + 1) / num_chunks * 100)
                self.progress.emit(progress)
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

    def format_text(self, text):
        # Add proper spacing after punctuation
        text = text.replace('.', '. ').replace('?', '? ').replace('!', '! ')
        
        # Split into sentences
        sentences = text.split('. ')
        
        # Group sentences into paragraphs (roughly 3-4 sentences per paragraph)
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_paragraph.append(sentence)
            
            if len(current_paragraph) >= 3:
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
        
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph) + '.')
        
        # Join paragraphs with double newline
        return '\n\n'.join(paragraphs)

    def stop(self):
        self.running = False

class TranscriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Add FFmpeg path to environment
        ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg_files', 'bin')
        if os.path.exists(ffmpeg_path):
            os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
            
        self.model = None
        self.audio_thread = None
        self.transcription_thread = None
        self.file_transcription_thread = None
        self.is_recording = False
        self.waveform_visualizer = None
        self.recording_indicator = None
        
        # Initialize UI first
        self.init_ui()
        # Then initialize model
        self.init_model()
    
    def init_ui(self):
        self.setWindowTitle("WCWS Murmur")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f172a;
                color: #f8fafc;
            }
        """)
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(32)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Header section with recording indicator
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(24)
        
        # Title container
        title = QLabel("WCWS Murmur")
        title.setStyleSheet("""
            font-size: 36px;
            font-weight: bold;
            color: #f8fafc;
            letter-spacing: -0.5px;
        """)
        header_layout.addWidget(title)
        
        # Status container with recording indicator
        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)  # Space between dot and text
        
        self.recording_indicator = RecordingIndicator()
        self.recording_indicator.hide()
        
        # Status label on the right
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            color: #94a3b8;
            font-size: 18px;
        """)
        
        status_layout.addWidget(self.recording_indicator)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        header_layout.addStretch()  # This pushes the status to the right
        header_layout.addWidget(status_container)
        
        layout.addWidget(header)
        
        # Add waveform visualizer
        self.waveform_visualizer = WaveformVisualizer()
        layout.addWidget(self.waveform_visualizer)
        
        # Controls section
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Device and model selection
        selection_layout = QHBoxLayout()
        device_label = QLabel("Input Device:")
        device_label.setStyleSheet("color: #e2e8f0; font-size: 16px;")
        self.device_combo = ModernComboBox()
        self.update_devices()
        selection_layout.addWidget(device_label)
        selection_layout.addWidget(self.device_combo)
        
        model_label = QLabel("Model:")
        model_label.setStyleSheet("color: #e2e8f0; font-size: 16px;")
        self.model_combo = ModernComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        self.model_combo.currentTextChanged.connect(self.change_model)
        selection_layout.addWidget(model_label)
        selection_layout.addWidget(self.model_combo)
        
        controls_layout.addLayout(selection_layout)
        controls_layout.addStretch()
        
        layout.addWidget(controls)
        
        # Action buttons
        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(16)
        
        self.record_button = ModernButton("Start Recording", "mic")
        self.record_button.clicked.connect(self.toggle_recording)
        buttons_layout.addWidget(self.record_button)
        
        self.file_button = ModernButton("Upload Audio", "upload", False)
        self.file_button.clicked.connect(self.process_file)
        buttons_layout.addWidget(self.file_button)
        
        self.stop_processing_button = ModernButton("Stop Processing", "stop", False)
        self.stop_processing_button.clicked.connect(self.stop_file_processing)
        self.stop_processing_button.hide()
        buttons_layout.addWidget(self.stop_processing_button)
        
        self.clear_button = ModernButton("Clear", "trash", False)
        self.clear_button.clicked.connect(self.clear_output)
        buttons_layout.addWidget(self.clear_button)
        
        self.copy_button = ModernButton("Copy Text", "clipboard", False)
        self.copy_button.clicked.connect(self.copy_output)
        buttons_layout.addWidget(self.copy_button)
        
        buttons_layout.addStretch()
        layout.addWidget(buttons)
        
        # Output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: #1f2937;
                border: 2px solid #374151;
                border-radius: 16px;
                padding: 24px;
                font-size: 18px;
                line-height: 1.8;
                color: #f8fafc;
                font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }
        """)
        layout.addWidget(self.output_text)
    
    def init_model(self):
        try:
            model_name = self.model_combo.currentText()
            self.status_label.setText(f"Loading {model_name} model...")
            
            torch.set_num_threads(8)
            self.model = whisper.load_model(
                model_name,
                device="cpu",
                download_root=os.path.expanduser("~/.cache/whisper")
            )
            self.model.eval()
            
            self.status_label.setText("Model ready")
            return True
        except Exception as e:
            self.status_label.setText(f"Model loading failed: {str(e)}")
            return False

    def change_model(self):
        if not self.is_recording:
            self.init_model()
        else:
            self.model_combo.setCurrentText(self.model_combo.currentText())
            self.status_label.setText("Cannot change model while recording")

    def update_devices(self):
        self.device_combo.clear()
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.device_combo.addItem(f"{dev['name']}", i)
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        device_id = self.device_combo.currentData()
        
        # Start audio capture
        self.audio_thread = AudioInputThread(device_id)
        
        # Start transcription
        self.transcription_thread = TranscriptionThread(self.model)
        self.audio_thread.audio_captured.connect(self.transcription_thread.add_audio)
        self.audio_thread.audio_captured.connect(self.waveform_visualizer.update_waveform)
        self.transcription_thread.text_ready.connect(self.update_output)
        
        # Start threads
        self.audio_thread.start()
        self.transcription_thread.start()
        
        # Show and start recording indicator
        self.recording_indicator.show()
        self.recording_indicator.animation_timer.timeout.connect(
            self.recording_indicator.update_visibility
        )
        
        # Update UI
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self.status_label.setText("Recording active")
        self.device_combo.setEnabled(False)
        self.file_button.setEnabled(False)
    
    def stop_recording(self):
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait()
            self.audio_thread = None
            
        if self.transcription_thread:
            self.transcription_thread.stop()
            self.transcription_thread.wait()
            self.transcription_thread = None
        
        # Hide recording indicator
        self.recording_indicator.hide()
        try:
            self.recording_indicator.animation_timer.timeout.disconnect(
                self.recording_indicator.update_visibility
            )
        except:
            pass
        
        # Update UI
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self.status_label.setText("Ready")
        self.device_combo.setEnabled(True)
        self.file_button.setEnabled(True)
    
    def process_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.mp4)"
        )
        
        if file_path:
            self.status_label.setText("Processing audio file...")
            self.file_button.setEnabled(False)
            self.record_button.setEnabled(False)
            self.stop_processing_button.show()  # Show stop button
            
            # Create and start transcription thread
            self.file_transcription_thread = FileTranscriptionThread(self.model, file_path)
            self.file_transcription_thread.progress.connect(self.update_progress)
            self.file_transcription_thread.chunk_transcribed.connect(self.update_output)
            self.file_transcription_thread.finished.connect(self.on_transcription_finished)
            self.file_transcription_thread.error.connect(self.on_transcription_error)
            self.file_transcription_thread.start()

    def update_progress(self, value):
        self.status_label.setText(f"Processing audio file... {value}%")

    def on_transcription_finished(self):
        self.status_label.setText("File processing complete")
        self.file_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.stop_processing_button.hide()  # Hide stop button
        self.file_transcription_thread = None

    def on_transcription_error(self, error_msg):
        self.status_label.setText(f"Processing failed: {error_msg}")
        self.file_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.stop_processing_button.hide()  # Hide stop button
        self.file_transcription_thread = None

    def stop_file_processing(self):
        if self.file_transcription_thread and self.file_transcription_thread.isRunning():
            self.file_transcription_thread.stop()
            self.file_transcription_thread.wait()
            self.status_label.setText("Processing stopped by user")
            self.file_button.setEnabled(True)
            self.record_button.setEnabled(True)
            self.stop_processing_button.hide()
            self.file_transcription_thread = None

    def update_output(self, text):
        self.output_text.setText(text)
        
        # Use a standard system font
        font = QFont("Arial", 18)
        self.output_text.setFont(font)
        
        # Scroll to bottom
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.output_text.setTextCursor(cursor)
    
    def clear_output(self):
        self.output_text.clear()
        self.status_label.setText("Output cleared")
    
    def copy_output(self):
        text = self.output_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText("Text copied to clipboard")
    
    def closeEvent(self, event):
        if self.is_recording:
            self.stop_recording()
        event.accept()

def main():
    # Configure environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    
    # Ensure only one instance is running
    socket = QLocalSocket()
    socket.connectToServer("WCWS-Murmur-App")
    
    if socket.waitForConnected(300):
        print("Application is already running")
        sys.exit(1)
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set up single instance mechanism
    server = QLocalServer()
    server.listen("WCWS-Murmur-App")
    
    # Set app icon if available
    if os.path.exists("icons/app-icon.png"):
        app.setWindowIcon(QIcon("icons/app-icon.png"))
    
    # Create and show main window
    window = TranscriptionApp()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
