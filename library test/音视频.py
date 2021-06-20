import ffmpeg
import subprocess
import av
import image
from ffmpy import FFmpeg
from moviepy.editor import *
from pydub import AudioSegment
from pydub.utils import make_chunks
import datetime, time

# attract audio from input video file by using "ffmpy"
ff = FFmpeg(inputs={'xxx.mp4': None},
            outputs={'xxx.WAV': None})
ff.run()

# attract audio from input video file by using "moviepy"
# video = VideoFileClip('test.mp4')
# audio = video.audio
# audio.write_audiofile('test.wav')

# cut audio into different segments by using "pydub"
# audio = AudioSegment.from_file("test.wav", "wav")
# size = 10000  # cut audip with length 10s=10000
# chunks = make_chunks(audio, size)
# for i, chunk in enumerate(chunks):
#    chunk_name = "seg-{0}.wav".format(i)
#      print(chunk_name)
#      chunk.export(chunk_name, format="wav")
#
# attract audio from input video file by using "ffmpeg"
#  (
#    ffmpeg
#       .input('test.mp4')
#       .output('output.mp3')
#       .run()
#  )
